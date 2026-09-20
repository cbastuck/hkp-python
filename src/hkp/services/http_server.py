from __future__ import annotations

# Service Documentation
# Service ID: http-server-subservices
# Service Name: HttpServerSubservices
# Runtime: hkp-python
# Modes: none — the entry points a board declares say what it is for
# Key Config: bypass/onProcess/onRequest (the endpoint is assigned, not configured)
# IO: in=request envelope -> out=response envelope
# Arrays: not primary
# Binary: depends on endpoint + nested services
# MixedData: not native in runtime
#
# **There are two ways in, and a board names the ones it uses.** `onRequest` is
# a caller arriving; `onProcess` is a pass of the board's own chain. Each is a
# pipeline of its own, because they are different jobs:
#
#     { "onRequest": [ … ] }                      requests; a pass goes through
#     { "onProcess": [ … ] }                      passes; the board answers
#     { "onProcess": [ … ], "onRequest": [ … ] }  both, separately
#     { "pipeline":  [ … ] }                      one pipeline, entered from both
#
# **Declaring `onRequest` is what takes the answer away from the chain.** With
# one, that pipeline is the handler and what it returns is what the caller gets;
# the services after this one still run — that is where a board acts on having
# served a request — but after the answer is decided. Without one, the request
# flows into the services after this one and whatever they return is the answer,
# which is the inversion of control this service is built around.
#
# So an endpoint can have something to run on a pass without silently becoming
# an HTTP handler, which is what a single unnamed pipeline could not express:
# having one at all decided who answered.
#
# **A value does not survive between the two on its own.** They are separate
# pipelines, and a pass ends where it ends — so an endpoint that publishes what
# the board last handed it holds that value in a slot (see `hold`), in cells
# this service owns and lends to both of its pipelines. Legacy boards say the
# same thing as `mode: "process_on_data"`, which is this arrangement built in
# and unnamed; `_entry_for` is where the older spellings are read.
#
# What the handler answers with is the envelope read backwards: a value carrying
# `meta.status` beside `body` or `binary` sets the status, the content type and
# the headers, and anything else is answered as JSON — which is what a board
# written before this got. Raw bytes are the one value whose own type decides.
# That is what lets one endpoint serve a feed as XML and the next serve audio as
# audio; without it an endpoint can only ever say `application/json`, whatever
# it is holding. Byte answers are seekable: `Range` is honoured against the
# bytes the handler produced.
#
# Mirrors hkp-node's `http-server-subservices`, which reads the same shapes.

import asyncio
import json
import re
import uuid as _uuid_mod
from typing import Any, Callable
from urllib.parse import parse_qsl, urlparse

from aiohttp import web

from ..mounts import MountContext, MountHandle, decode_body
from ..runtime import new_run
from ..mount import MOUNT_FIELD
from ..types import (
    ProcessContext,
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    RuntimeNotification,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
    SlotStore,
)
from .nested_pipeline import NestedPipeline
from .sub_service import _is_json_record

#: The two ways into this service, named.
#:
#: ``onProcess`` is a pass of the board's own chain arriving; ``onRequest`` is a
#: caller. They are declared as separate pipelines because they are separate
#: jobs — which is what the ``mode`` flag was standing in for, badly: one
#: unnamed list could not say what it was for, so a flag beside it had to.
ENTRY_NAMES = ("onProcess", "onRequest")

HTTP_SERVER_SUBSERVICES_DESCRIPTOR = ServiceRegistryEntry(
    service_id="http-server-subservices",
    service_name="HttpServerSubservices",
    capabilities=["subservices"],
)


def _filename_from_disposition(disposition: str | None) -> str | None:
    """Extract ``filename="…"`` from a Content-Disposition header, if present."""
    if not disposition:
        return None
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition, re.I)
    return match.group(1) if match else None


def _as_response_envelope(value: Any) -> dict[str, Any] | None:
    """A value read as a response envelope, or None when it is not one.

    **A status is what tells a response from a request.** Both are the same
    shape — ``meta`` beside ``body`` or ``binary`` — so a pipeline that passes
    its input through returns a request, and reading any ``meta`` as a response
    would answer the caller with the content type they sent. A request carries
    no status and a response always does, which makes ``meta.status`` the one
    field that cannot be a coincidence. It is also what ``http-client`` reports
    a call's response as, so proxying one takes no translation.
    """
    if not isinstance(value, dict):
        return None
    meta = value.get("meta")
    if not isinstance(meta, dict):
        return None
    if not isinstance(meta.get("status"), int) or isinstance(meta.get("status"), bool):
        return None
    return value if ("binary" in value or "body" in value) else None


def _envelope_headers(meta: dict[str, Any]) -> dict[str, str]:
    """Header map from an envelope's ``meta.headers``, names lower-cased."""
    declared = meta.get("headers")
    if not isinstance(declared, dict):
        return {}
    return {
        str(name).lower(): str(value)
        for name, value in declared.items()
        if value is not None
    }


def _to_answer(value: Any) -> tuple[int, dict[str, str], bytes]:
    """The status, headers and bytes a value stands for.

    Without an envelope the answer is JSON, which is what every board written
    before this got and still gets. The single exception is raw bytes, which are
    sent as bytes: there is no JSON encoding of them anybody wanted.
    """
    envelope = _as_response_envelope(value)
    if envelope is not None:
        meta = envelope.get("meta") or {}
        headers = _envelope_headers(meta)
        status = int(meta["status"])
        declared = meta.get("contentType")
        declared = declared if isinstance(declared, str) else None

        binary = envelope.get("binary")
        if isinstance(binary, (bytes, bytearray, memoryview)):
            headers["content-type"] = (
                declared or headers.get("content-type") or "application/octet-stream"
            )
            return status, headers, bytes(binary)

        body = envelope.get("body")
        if isinstance(body, str):
            headers["content-type"] = (
                declared or headers.get("content-type") or "text/plain; charset=utf-8"
            )
            return status, headers, body.encode("utf-8")

        headers["content-type"] = (
            declared or headers.get("content-type") or "application/json"
        )
        return status, headers, json.dumps(body, default=str).encode("utf-8")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return 200, {"content-type": "application/octet-stream"}, bytes(value)

    return (
        200,
        {"content-type": "application/json"},
        json.dumps(value if value is not None else None, default=str).encode("utf-8"),
    )


def _charset_of(content_type: str) -> str | None:
    """The charset a content type declares, which aiohttp takes separately."""
    for parameter in content_type.split(";")[1:]:
        name, _, value = parameter.partition("=")
        if name.strip().lower() == "charset":
            return value.strip() or None
    return None


def _requested_range(header: str | None, length: int) -> tuple[int, int] | None:
    """The range a ``Range: bytes=…`` header asks for, clamped to what there is.

    None when the header asks for nothing this can serve — absent, a unit other
    than bytes, several ranges, or a start past the end. A player seeking inside
    an audio file sends one of these, and a server that ignores it re-sends the
    whole file every time somebody drags the scrubber.
    """
    if not header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match:
        return None
    raw_start, raw_end = match.groups()
    if not raw_start and not raw_end:
        return None

    # "bytes=-500" is the last 500 bytes, not a range starting at zero.
    if not raw_start:
        span = int(raw_end)
        return (max(0, length - span), length - 1) if span else None

    start = int(raw_start)
    if start >= length:
        return None
    end = min(int(raw_end), length - 1) if raw_end else length - 1
    return (start, end) if end >= start else None


class HttpServerSubservicesService:
    service_id = HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id
    service_name = HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_name
    version: str | None = None
    capabilities = HTTP_SERVER_SUBSERVICES_DESCRIPTOR.capabilities

    def __init__(
        self,
        config: ServiceConfiguration,
        create_service: ServiceCreator,
        # Upper bound on a request body, in bytes; 0 disables the limit.
        # Supplied by the server because the endpoint is public and shared.
        max_body_bytes: int = 0,
    ) -> None:
        self._max_body_bytes = max_body_bytes
        self.uuid = config.uuid
        self._bypass = True
        self._mode: str = "process_on_session"
        self._latest_data: Any = None
        self._mount: MountHandle | None = None
        #: What this endpoint is called, which is what its public address is
        #: derived from. Empty falls back to the service's uuid, which is stable
        #: in a board file too — so an address only changes when a board
        #: deliberately renames it.
        self._mount_name = ""
        #: Which way the board declared its pipelines; see the module header.
        #:
        #: Kept because state reports what was declared rather than a canonical
        #: form: a board saved after being loaded has to come back out the way
        #: it went in, or every board on the older spelling rewrites itself the
        #: first time somebody saves it.
        self._form = "legacy"
        #: The one pipeline a legacy board declares, entered from whichever side
        #: its `mode` says.
        self._legacy: NestedPipeline | None = None
        self._entries: dict[str, NestedPipeline | None] = {
            "onProcess": None,
            "onRequest": None,
        }
        #: The cells this endpoint's pipelines hold values in.
        #:
        #: Owned here because the two entry points are pipelines that never
        #: meet: a value one of them produces has nowhere to live until the
        #: other runs. One store per endpoint is also what keeps the names in it
        #: private, so two endpoints on a runtime may both call a slot
        #: ``document``.
        self._slot_store = SlotStore()
        #: Which of a request's headers the pipeline is shown, or None for all.
        #:
        #: Headers are where a caller puts a credential, and ``meta`` goes
        #: wherever the pipeline takes it — including into a board, if a service
        #: is wired to write it there. Naming the ones a board actually reads is
        #: how it stops carrying the ones it does not: an empty list forwards
        #: none, and no list at all forwards everything, which is what a board
        #: that has not thought about it gets.
        self._forward_headers: list[str] | None = None
        self._create_service = create_service
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        # `port` is accepted and ignored: the endpoint is served by the shared
        # runtime server under an assigned path, so a service no longer picks a
        # port. Older boards still carry the field, and rejecting it would fail
        # them on load for a setting that no longer means anything.

        # An array is a decision, including an empty one. Anything else —
        # absent, None, a string — leaves the default of forwarding all of them.
        if "forwardHeaders" in config:
            names = config["forwardHeaders"]
            self._forward_headers = (
                [n.lower() for n in names if isinstance(n, str)]
                if isinstance(names, list)
                else None
            )

        if isinstance(config.get("mountName"), str):
            # Renaming rotates this endpoint's address, so an already-claimed
            # mount is released and claimed again under the new name rather than
            # left answering on the old one.
            renamed = config["mountName"] != self._mount_name
            self._mount_name = config["mountName"]
            if renamed and self._mount:
                self._release_mount()

        # How a board that predates named entry points says which side enters
        # the one pipeline it declares. Still accepted, and still reported back
        # to a board that arrived carrying it; _entry_for is the whole of what
        # it means now.
        if config.get("mode") in (
            "process_on_session",
            "process_on_data",
            "process_on_both",
        ):
            self._mode = config["mode"]

        # Declaring an entry point by name is what puts this endpoint in the
        # newer form, and from then on its state is reported that way.
        for name in ENTRY_NAMES:
            if isinstance(config.get(name), list):
                self._form = "entries"
                self._entry_pipeline(name).set_pipeline(config[name])

        # An edit aimed at one named entry. The unscoped verbs below cannot say
        # which pipeline they mean once there is more than one.
        if _is_json_record(config.get("configurePipeline")):
            payload = config["configurePipeline"]
            if payload.get("entry") in ENTRY_NAMES:
                self._form = "entries"
                self._edit_pipeline(self._entry_pipeline(payload["entry"]), payload)

        if (
            isinstance(config.get("pipeline"), list)
            or _is_json_record(config.get("appendService"))
            or isinstance(config.get("removeService"), str)
            or _is_json_record(config.get("configureService"))
        ):
            # The unscoped verbs belong to the one pipeline a legacy board
            # declares. Left working rather than redirected at an entry, because
            # which entry they would mean is exactly what the older form cannot
            # say.
            self._edit_pipeline(self._legacy_pipeline(), config)

        # Bypass toggle
        if isinstance(config.get("bypass"), bool) and config["bypass"] != self._bypass:
            self._bypass = config["bypass"]
            if self._bypass:
                self._release_mount()
            else:
                self._claim_mount()

        # Anything above may have left this without an endpoint it should have —
        # coming out of bypass, or a rename that released the old address. One
        # check covers them rather than one per cause.
        if not self._bypass and not self._mount:
            self._claim_mount()

        return self.get_state()

    def _request_headers(self, request: Any) -> JsonRecord:
        """The headers this pipeline is shown, lower-cased as HTTP names compare.

        A caller that has to prove who it is does so in a header — a shared
        secret, a signature, a bearer token — so a pipeline that cannot see them
        cannot check one. What a board does not name, it does not receive.
        """
        headers: JsonRecord = {}
        for name, value in request.headers.items():
            lowered = name.lower()
            if self._forward_headers is not None and lowered not in self._forward_headers:
                continue
            headers[lowered] = value
        return headers

    def find_nested(self, instance_id: str):
        """The nested service a scoped address names, searching every pipeline
        this endpoint owns. The two entries are separate pipelines rather than
        branches of one, so an instanceId used in both resolves to whichever
        ``_pipelines()`` lists first.
        """
        for pipeline in self._pipelines():
            found = pipeline.find(instance_id)
            if found is not None:
                return found
        return None

    def remount(self) -> None:
        """See HostedService.remount: the runtime can serve one now.

        Bypassed is the same answer it gives at set_host: an endpoint switched
        off holds no address, and gaining somewhere to claim one does not
        switch it back on.
        """
        if not self._bypass:
            self._claim_mount()

    def get_state(self) -> JsonRecord:
        common = {
            "bypass": self._bypass,
            # Name this endpoint is known by; see the field on the service.
            "mountName": self._mount_name,
            # Public endpoint assigned by the runtime; empty while bypassed.
            # Reserved name: generic board machinery reads and rewrites it (see
            # the frontend's runtime/board/mount).
            MOUNT_FIELD: self._mount.url if self._mount else "",
            "forwardHeaders": self._forward_headers,
        }

        # What was declared, not what it was understood as. A board that named
        # its entries gets them back; one that carries a `mode` keeps it,
        # because that is the version an older runtime can still load.
        if self._form == "entries":
            state: JsonRecord = dict(common)
            for name in ENTRY_NAMES:
                pipeline = self._entries[name]
                if pipeline is not None:
                    state[name] = pipeline.state()
            return state

        return {
            **common,
            "mode": self._mode,
            "pipeline": self._legacy.state() if self._legacy else [],
        }

    # ── Pipelines ──────────────────────────────────────────────────────────────

    def _entry_pipeline(self, name: str) -> NestedPipeline:
        """The pipeline behind one entry point, built on first use.

        Built lazily because an endpoint declaring only ``onRequest`` should not
        carry an empty runtime for the side it never uses — and because an empty
        pipeline and an absent one differ here: only the second leaves the board
        answering.
        """
        existing = self._entries[name]
        if existing is not None:
            return existing
        created = self._new_pipeline(name)
        self._entries[name] = created
        return created

    def _legacy_pipeline(self) -> NestedPipeline:
        if self._legacy is None:
            self._legacy = self._new_pipeline("pipeline")
        return self._legacy

    def _new_pipeline(self, label: str) -> NestedPipeline:
        pipeline = NestedPipeline(
            f"{self.uuid}:{label}",
            self._create_service,
            "HttpServerSubservices",
            self.uuid,
        )
        # Both entries hold in the same cells: that they can is the whole reason
        # for declaring them separately.
        pipeline.share_slots(self._slot_store)
        if self._host:
            pipeline.attach(self._host)
        return pipeline

    def _edit_pipeline(self, pipeline: NestedPipeline, payload: JsonRecord) -> None:
        """The four edit verbs, applied to whichever pipeline was named."""
        if isinstance(payload.get("pipeline"), list):
            pipeline.set_pipeline(payload["pipeline"])
        elif _is_json_record(payload.get("appendService")):
            pipeline.append(payload["appendService"])
        elif isinstance(payload.get("removeService"), str):
            pipeline.remove(payload["removeService"])
        elif _is_json_record(payload.get("configureService")):
            edit = payload["configureService"]
            if isinstance(edit.get("instanceId"), str) and _is_json_record(edit.get("state")):
                pipeline.configure_service(edit["instanceId"], edit["state"])

    def _pipelines(self) -> list[NestedPipeline]:
        """Every pipeline this endpoint owns, each one only once."""
        seen: list[NestedPipeline] = []
        for pipeline in (self._legacy, self._entries["onProcess"], self._entries["onRequest"]):
            if pipeline is not None and not any(p is pipeline for p in seen):
                seen.append(pipeline)
        return seen

    def _entry_for(self, name: str) -> NestedPipeline | None:
        """The pipeline one side enters through, or None where it has none.

        This is where a legacy ``mode`` is read, and the only place it is: a
        board that names its entries never reaches the table below.

        =====================  =========  =========
        declared               onProcess  onRequest
        =====================  =========  =========
        ``process_on_session``  —          the one
        ``process_on_both``     the one    the one
        ``process_on_data``     —          —
        =====================  =========  =========

        ``process_on_both`` answers with *the same instance* on both sides,
        never a second copy of the configuration: a pipeline holding a Hold, a
        timer or a mount is one running thing, and duplicating it would give a
        board two of each and a slot that never reaches itself.
        """
        if self._form == "entries":
            return self._entries[name]
        if self._mode == "process_on_data":
            return None
        if name == "onProcess" and self._mode != "process_on_both":
            return None
        return self._legacy

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host
        # A pipeline built in the constructor was built before there was a host
        # to ask what board it belongs to, or to report through.
        for pipeline in self._pipelines():
            pipeline.attach(host)
        # State is applied in the constructor, before the host exists, so a
        # service configured as already-active has nothing to claim its mount
        # from until now. Claiming here is what makes a board load into a live
        # endpoint.
        if not self._bypass and not self._mount:
            self._claim_mount()

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        # The legacy built-in slot: what the board hands this endpoint is what a
        # caller gets back. Expressible now as an ``onProcess`` that writes a
        # slot and an ``onRequest`` that reads it, and kept because boards carry
        # the older spelling and a board is a document people keep.
        if self._form == "legacy" and self._mode == "process_on_data":
            self._latest_data = input
            return input

        entry = None if self._bypass else self._entry_for("onProcess")
        if entry is None:
            return input

        # Routing only: what the pass's own pipeline returns carries on down the
        # chain. Whatever has to survive until a request arrives — a value this
        # side produces and the other reads — belongs in a slot, which is a
        # service's job and not this one's.
        return self._run_entry(entry, input)

    def destroy(self) -> None:
        self._release_mount()
        # Nested services hold the same things top-level ones do — timers,
        # sockets, mounts — and nothing else will ever reach them once this
        # service is gone.
        for pipeline in self._pipelines():
            pipeline.destroy()
        self._legacy = None
        self._entries["onProcess"] = None
        self._entries["onRequest"] = None

    # ── Mount ──────────────────────────────────────────────────────────────────

    def _claim_mount(self) -> None:
        if self._mount or not self._host:
            return
        mount = self._host.mount(self.uuid, self._handle_request, self._mount_name)
        if not mount:
            return
        self._mount = mount
        # A board reads the assigned endpoint from here (or from state), since
        # it is not knowable at design time.
        self._do_notify({MOUNT_FIELD: mount.url}, self.uuid)

    def _release_mount(self) -> None:
        if self._mount:
            self._mount.release()
        self._mount = None

    async def _handle_request(
        self, request: web.Request, context: MountContext
    ) -> web.Response:
        if self._bypass:
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"error": "http-server-subservices is bypassed"}),
            )

        # Serving a request is one run, however many pipelines it passes
        # through: the nested handler below descends from it, and the outer
        # chain afterwards continues it. Minting one here rather than letting
        # each leg mint its own is what keeps a request's trace joined up
        # instead of arriving as two unrelated runs sharing a timestamp.
        run_context = new_run()

        # Whether the answer is already decided here, or is whatever the rest
        # of the outer chain makes of what this service emitted.
        answered_here = False
        if self._form == "legacy" and self._mode == "process_on_data":
            process_input = self._latest_data
            output: Any = process_input
            # The mode's whole contract: what the board handed this endpoint is
            # what a caller gets back, verbatim. The services after it still run
            # — having served a request is something a board may want to act on
            # — but what they make of it is theirs, not the answer. Letting the
            # chain's tail answer instead would mean an endpoint could only ever
            # be the last service in its runtime, so a runtime could publish
            # only one document.
            answered_here = True
        else:
            process_input = await self._read_request(request, context)
            # A declared handler is what takes the answer away from the chain —
            # not the presence of a pipeline, which says only that this endpoint
            # has something to run, possibly on the other side. Without one the
            # board answers, which is the inversion of control this service is
            # built around.
            handler = self._entry_for("onRequest")
            answered_here = handler is not None and not handler.is_empty()
            output = (
                self._run_entry(handler, process_input, run_context)
                if handler is not None
                else process_input
            )

        # What the nested pipeline produced, before the outer runtime sees it.
        answer = output

        if self._host:
            # process_from reports this service's own call-process pair, so
            # there is no manual pair here — emitting one too would double every
            # request in the UI. It also reports the right value: what this
            # service emitted, rather than what the whole downstream chain
            # finally returned.
            #
            # The callback is a no-op: the runtime already fans notifications
            # out to its targets, and re-notifying would deliver each twice.
            output = self._host.process_from(
                self.uuid, output, lambda _n: None, run_context
            )
            self._host.emit_result(output)

        # With a nested pipeline configured, that pipeline is the handler and
        # what it returned is the answer; the outer runtime ran for its side
        # effects. Without one, the rest of the board is the handler — except in
        # `process_on_data`, where the stored document is the answer.
        response_value = answer if answered_here else output
        return self._answer(request, response_value)

    def _answer(self, request: web.Request, value: Any) -> web.Response:
        """Writes what the handler produced, honouring a range request when the
        answer is bytes a caller can seek inside."""
        status, headers, payload = _to_answer(value)
        content_type = headers.pop("content-type", "application/octet-stream")

        if status == 200:
            headers["accept-ranges"] = "bytes"
            wanted = _requested_range(request.headers.get("Range"), len(payload))
            if wanted:
                start, end = wanted
                headers["content-range"] = f"bytes {start}-{end}/{len(payload)}"
                return web.Response(
                    status=206,
                    headers=headers,
                    content_type=content_type.split(";")[0].strip(),
                    charset=_charset_of(content_type),
                    body=payload[start : end + 1],
                )

        return web.Response(
            status=status,
            headers=headers,
            content_type=content_type.split(";")[0].strip(),
            charset=_charset_of(content_type),
            body=payload,
        )

    async def _read_request(
        self, request: web.Request, context: MountContext
    ) -> JsonRecord:
        """Build the MixedData an incoming request becomes: JSON ``meta``
        describing it, plus the body in whichever single form is useful —
        decoded as ``body`` when the content type says what the bytes mean, raw
        as ``binary`` otherwise. Matches hkp-node's http-server-subservices so a
        pipeline written for one runtime works on the other.
        """
        # The mount prefix is transport addressing, not part of the route the
        # pipeline matches on, so the pipeline sees the path below the mount.
        parsed = urlparse(context.sub_path)
        meta: JsonRecord = {
            "method": request.method,
            "path": parsed.path or "/",
            "query": dict(parse_qsl(parsed.query)),
            "headers": self._request_headers(request),
        }

        content_type = request.headers.get("Content-Type")
        if content_type:
            meta["contentType"] = content_type
        filename = _filename_from_disposition(
            request.headers.get("Content-Disposition")
        )
        if filename:
            meta["filename"] = filename

        body = await self._read_body(request)

        # Exactly one representation of the body, or neither when there was none.
        decoded = decode_body(body, content_type)
        if decoded is not None:
            return {"meta": meta, "body": decoded}
        if body:
            return {"meta": meta, "binary": body}
        return {"meta": meta}

    async def _read_body(self, request: web.Request) -> bytes:
        """Read the request body, refusing anything past the configured cap.

        A mount is reachable without a token by design, so an unbounded read is
        a way for anyone holding the URL to exhaust the host — which on a shared
        instance is everyone else's problem too. The cap is enforced while
        reading rather than from Content-Length, which a client controls.
        """
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await request.content.readany()
            if not chunk:
                break
            total += len(chunk)
            if self._max_body_bytes and total > self._max_body_bytes:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=self._max_body_bytes, actual_size=total
                )
            chunks.append(chunk)
        return b"".join(chunks)

    # ── Pipeline helpers ───────────────────────────────────────────────────────

    def _run_entry(
        self,
        pipeline: NestedPipeline,
        input: Any,
        parent: ProcessContext | None = None,
    ) -> Any:
        """Runs one entry's pipeline as a run descended from ``parent``.

        Both entry points land here, and they differ only in what they descend
        from: a request brings the run its caller minted for the whole exchange,
        while data from the outer chain arrives mid-call and descends from
        whatever that call is running as.
        """
        context = parent
        if context is None and self._host:
            context = self._host.current_context()
        return pipeline.process(input, context)

    def _do_notify(self, payload: Any, instance_id: str | None = None) -> None:
        if self._host:
            self._host.notify(payload, instance_id or self.uuid)
