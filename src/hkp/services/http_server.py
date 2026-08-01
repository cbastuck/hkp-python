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

import asyncio
import json
import re
import uuid as _uuid_mod
from typing import Any
from urllib.parse import parse_qsl, urlparse

from aiohttp import web

from ..mounts import MountContext, MountHandle, decode_body
from ..runtime import HostedRuntime
from ..types import (
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
        self._pipeline_config: list[ServiceConfiguration] = []
        self._pipeline: HostedRuntime | None = None
        self._create_service = create_service
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        previous_bypass = self._bypass

        # `port` is accepted and ignored: the endpoint is served by the shared
        # runtime server under an assigned path, so a service no longer picks a
        # port. Older boards still carry the field, and rejecting it would fail
        # them on load for a setting that no longer means anything.

        # Mode change
        if config.get("mode") in ("process_on_session", "process_on_data"):
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

        # Claim if we transitioned from bypassed to active without a mount yet
        if previous_bypass and not self._bypass and not self._mount:
            self._claim_mount()

        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "bypass": self._bypass,
            "mode": self._mode,
            # Public endpoint assigned by the runtime; empty while bypassed.
            # Reserved name: generic board machinery reads and rewrites it (see
            # the frontend's runtime/board/mount).
            "__hkpMount": self._mount.url if self._mount else "",
            "pipeline": self._get_pipeline_state(),
        }

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host
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

    def destroy(self) -> None:
        self._release_mount()
        self._pipeline = None
        self._pipeline_config = []

    # ── Mount ──────────────────────────────────────────────────────────────────

    def _claim_mount(self) -> None:
        if self._mount or not self._host:
            return
        mount = self._host.mount(self.uuid, self._handle_request)
        if not mount:
            return
        self._mount = mount
        # A board reads the assigned endpoint from here (or from state), since
        # it is not knowable at design time.
        self._do_notify({"__hkpMount": mount.url}, self.uuid)

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

        answered_by_subservices = False
        if self._mode == "process_on_data":
            process_input = self._latest_data
            output: Any = process_input
        else:
            process_input = await self._read_request(request, context)
            answered_by_subservices = self._has_subservices()
            output = self._process_session_input(process_input)

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
            output = self._host.process_from(self.uuid, output, lambda _n: None)
            self._host.emit_result(output)

        # With a nested pipeline configured, that pipeline is the handler and
        # what it returned is the answer; the outer runtime ran for its side
        # effects. Without one, the rest of the board is the handler.
        response_value = answer if answered_by_subservices else output
        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps(
                response_value if response_value is not None else None, default=str
            ),
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

    def _process_session_input(self, input: Any) -> Any:
        if not self._pipeline or not self._pipeline.list_services():
            return input
        return self._pipeline.process(
            input,
            lambda n: self._do_notify(n.payload, n.instance_id),
        )

    def _do_notify(self, payload: Any, instance_id: str | None = None) -> None:
        if self._host:
            self._host.notify(payload, instance_id or self.uuid)

    def _rebuild(self) -> None:
        from ..types import RuntimeConfiguration
        self._pipeline = HostedRuntime(
            RuntimeConfiguration(
                id=f"{self.uuid}:http-sub-runtime",
                name=f"{self.service_name}-{self.uuid}",
                board_name="",
                services=self._pipeline_config,
            ),
            self._create_service,
        )

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
