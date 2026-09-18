from __future__ import annotations

# Service Documentation
# Service ID: http-server-subservices
# Service Name: HttpServerSubservices
# Runtime: hkp-python
# Modes: session pipeline hosting
# Key Config: host/port/routes/subservices
# IO: in=request envelope -> out=response envelope
# Arrays: not primary
# Binary: depends on endpoint + nested services
# MixedData: not native in runtime
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
from ..runtime import HostedRuntime, child_run, new_run
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
)
from .sub_service import _is_json_record, _normalize_pipeline_array, _normalize_pipeline_entry

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
        self._pipeline_config: list[ServiceConfiguration] = []
        #: Which of a request's headers the pipeline is shown, or None for all.
        #:
        #: Headers are where a caller puts a credential, and ``meta`` goes
        #: wherever the pipeline takes it — including into a board, if a service
        #: is wired to write it there. Naming the ones a board actually reads is
        #: how it stops carrying the ones it does not: an empty list forwards
        #: none, and no list at all forwards everything, which is what a board
        #: that has not thought about it gets.
        self._forward_headers: list[str] | None = None
        self._pipeline: HostedRuntime | None = None
        self._release_pipeline_notifications: Callable[[], None] | None = None
        self._release_pipeline_logs: Callable[[], None] | None = None
        self._create_service = create_service
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        # `port` is accepted and ignored: the endpoint is served by the shared
        # runtime server under an assigned path, so a service no longer picks a
        # port. Older boards still carry the field, and rejecting it would fail
        # them on load for a setting that no longer means anything.

        # Where the nested pipeline is entered from:
        #
        # - process_on_session — requests only; data from the outer chain passes
        #   through untouched.
        # - process_on_data — data from the outer chain is stored and served
        #   back to requests verbatim; the nested pipeline is not used.
        # - process_on_both — both entry points run the nested pipeline. The
        #   pipeline is a single ordered list either way, so a service inside it
        #   that needs to tell a request from a data arrival has to do so from
        #   the input.
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

        if config.get("mode") in (
            "process_on_session",
            "process_on_data",
            "process_on_both",
        ):
            self._mode = config["mode"]

        # Pipeline replacement
        if isinstance(config.get("pipeline"), list):
            next_pipeline = _normalize_pipeline_array(config["pipeline"])
            if next_pipeline is None:
                raise ValueError("Invalid http-server-subservices pipeline format")
            self._pipeline_config = next_pipeline
            self._rebuild()
        elif _is_json_record(config.get("appendService")):
            appended = _normalize_pipeline_entry(config["appendService"])
            if not appended:
                raise ValueError("Invalid appendService payload")
            self._sync_states()
            self._pipeline_config.append(appended)
            self._rebuild()
        elif isinstance(config.get("removeService"), str):
            self._sync_states()
            target = config["removeService"]
            self._pipeline_config = [e for e in self._pipeline_config if e.uuid != target]
            self._rebuild()
        elif _is_json_record(config.get("configureService")):
            payload = config["configureService"]
            if (
                isinstance(payload.get("instanceId"), str)
                and _is_json_record(payload.get("state"))
                and self._pipeline
            ):
                self._pipeline.configure_service(payload["instanceId"], payload["state"])
                self._sync_states()

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

    def get_state(self) -> JsonRecord:
        return {
            "bypass": self._bypass,
            "mode": self._mode,
            # Name this endpoint is known by; see the field on the service.
            "mountName": self._mount_name,
            # Public endpoint assigned by the runtime; empty while bypassed.
            # Reserved name: generic board machinery reads and rewrites it (see
            # the frontend's runtime/board/mount).
            MOUNT_FIELD: self._mount.url if self._mount else "",
            "forwardHeaders": self._forward_headers,
            "pipeline": self._get_pipeline_state(),
        }

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host
        # A pipeline built in the constructor was built before there was a host
        # to ask, so what the board records reaches it here rather than never.
        self._apply_log_settings()

    def _apply_log_settings(self) -> None:
        """Hands the board's log settings to the nested pipeline, if any."""
        if not self._host or not self._pipeline:
            return
        settings = self._host.log_settings()
        self._pipeline.set_logging(settings["logging"])
        self._pipeline.set_log_data(settings["log_data"])
        self._pipeline.set_log_level(settings["log_level"])
        # State is applied in the constructor, before the host exists, so a
        # service configured as already-active has nothing to claim its mount
        # from until now. Claiming here is what makes a board load into a live
        # endpoint.
        if not self._bypass and not self._mount:
            self._claim_mount()

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        if self._mode == "process_on_data":
            self._latest_data = input
            return input

        # Routing only: the nested pipeline handles data arriving from the outer
        # chain exactly as it handles a request, and what it returns carries on
        # down the chain. Whatever has to survive between the two — a value one
        # side produces and the other reads — is a service's job, not this one's.
        if self._mode == "process_on_both" and not self._bypass:
            return self._process_session_input(input)

        return input

    def destroy(self) -> None:
        self._release_mount()
        self._release_notifications()
        # Nested services hold the same things top-level ones do — timers,
        # sockets, mounts — and nothing else will ever reach them once this
        # service is gone.
        if self._pipeline:
            self._pipeline.destroy()
        self._pipeline = None
        self._pipeline_config = []

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

        answered_by_subservices = False
        if self._mode == "process_on_data":
            process_input = self._latest_data
            output: Any = process_input
        else:
            process_input = await self._read_request(request, context)
            answered_by_subservices = self._has_subservices()
            output = self._process_session_input(process_input, run_context)

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
        # effects. Without one, the rest of the board is the handler.
        response_value = answer if answered_by_subservices else output
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

    def _has_subservices(self) -> bool:
        """Whether a nested pipeline is configured to handle requests."""
        return bool(self._pipeline and self._pipeline.list_services())

    def _process_session_input(
        self, input: Any, parent: ProcessContext | None = None
    ) -> Any:
        """Runs the nested pipeline as a run descended from ``parent``.

        Both entry points land here, and they differ only in what they descend
        from: a request brings the run its caller minted for the whole exchange,
        while data from the outer chain arrives mid-call and descends from
        whatever that call is running as.
        """
        if not self._pipeline or not self._pipeline.list_services():
            return input
        # The callback is a no-op: the nested runtime fans these out to the
        # target registered in _rebuild. Forwarding them here as well would
        # deliver every one twice.
        return self._pipeline.process(
            input,
            lambda _n: None,
            child_run(parent or (self._host.current_context() if self._host else None)),
        )

    def _do_notify(self, payload: Any, instance_id: str | None = None) -> None:
        if self._host:
            self._host.notify(payload, instance_id or self.uuid)

    def _rebuild(self) -> None:
        from ..types import RuntimeConfiguration

        self._release_notifications()
        # The pipeline being replaced is about to become unreachable; its
        # services keep running until told otherwise. State worth carrying over
        # has already been read into _pipeline_config by _sync_states.
        if self._pipeline:
            self._pipeline.destroy()

        self._pipeline = HostedRuntime(
            RuntimeConfiguration(
                id=f"{self.uuid}:http-sub-runtime",
                name=f"{self.service_name}-{self.uuid}",
                board_name="",
                services=self._pipeline_config,
            ),
            self._create_service,
        )

        # A nested runtime has no notification targets of its own, so what its
        # services report — a Timer's tick, a Hold's counts — reaches nobody
        # unless the service hosting the pipeline carries it out to the board.
        # Services report through their host precisely because it is not always
        # a call they are answering: an autonomous emitter has no caller.
        self._release_pipeline_notifications = self._pipeline.register_notification_target(
            lambda n: self._do_notify(n.payload, n.instance_id)
        )

        # A nested pipeline's entries belong to the same board log as everything
        # else; only the runtime hosting this service can carry them there, since
        # a nested runtime has no route out of its own.
        self._release_pipeline_logs = self._pipeline.register_log_target(
            lambda entry: self._host.forward_log(entry) if self._host else None
        )

        self._apply_log_settings()

    def _release_notifications(self) -> None:
        if self._release_pipeline_notifications:
            self._release_pipeline_notifications()
            self._release_pipeline_notifications = None
        if self._release_pipeline_logs:
            self._release_pipeline_logs()
            self._release_pipeline_logs = None

    def _sync_states(self) -> None:
        if not self._pipeline:
            return
        by_id = {svc.uuid: svc.state for svc in self._pipeline.list_services()}
        self._pipeline_config = [
            ServiceConfiguration(
                service_id=entry.service_id,
                uuid=entry.uuid,
                name=entry.name,
                service_name=entry.service_name,
                state=by_id.get(entry.uuid, entry.state),
            )
            for entry in self._pipeline_config
        ]

    def _get_pipeline_state(self) -> list[dict[str, Any]]:
        if not self._pipeline:
            return [
                {
                    "serviceId": e.service_id,
                    "instanceId": e.uuid,
                    "state": e.state or {},
                }
                for e in self._pipeline_config
            ]
        return [
            {
                "serviceId": svc.service_id,
                "instanceId": svc.uuid,
                "state": svc.state,
            }
            for svc in self._pipeline.list_services()
        ]
