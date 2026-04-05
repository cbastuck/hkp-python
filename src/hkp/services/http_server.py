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
import uuid as _uuid_mod
from typing import Any

from aiohttp import web

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


class HttpServerSubservicesService:
    service_id = HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id
    service_name = HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_name
    version: str | None = None
    capabilities = HTTP_SERVER_SUBSERVICES_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        self.uuid = config.uuid
        self._bypass = True
        self._mode: str = "process_on_session"
        self._port = 0
        self._latest_data: Any = None
        self._pipeline_config: list[ServiceConfiguration] = []
        self._pipeline: HostedRuntime | None = None
        self._create_service = create_service
        self._host: RuntimeHost | None = None
        self._runner: web.AppRunner | None = None

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        previous_bypass = self._bypass

        # Port change
        if isinstance(config.get("port"), int) and not isinstance(config.get("port"), bool):
            new_port = config["port"]
            if 0 <= new_port <= 65535 and self._port != new_port:
                self._port = new_port
                if self._runner:
                    self._restart_server()

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
                self._stop_server()
            else:
                self._start_server()

        # Start server if we transitioned from bypassed to active without a server yet
        if previous_bypass and not self._bypass and not self._runner:
            self._start_server()

        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "bypass": self._bypass,
            "mode": self._mode,
            "port": self._port,
            "pipeline": self._get_pipeline_state(),
        }

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        if self._mode == "process_on_data":
            self._latest_data = input
        return input

    def destroy(self) -> None:
        self._stop_server()
        self._pipeline = None
        self._pipeline_config = []

    # ── Inner HTTP server ──────────────────────────────────────────────────────

    def _start_server(self) -> None:
        if self._runner:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._do_start_server())
        except RuntimeError:
            pass  # no running event loop (e.g. tests without asyncio)

    def _stop_server(self) -> None:
        runner = self._runner
        self._runner = None
        if not runner:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(runner.cleanup())
        except RuntimeError:
            pass

    def _restart_server(self) -> None:
        self._stop_server()
        if not self._bypass:
            self._start_server()

    async def _do_start_server(self) -> None:
        if self._runner:
            return

        inner_app = web.Application()
        inner_app.router.add_route("*", "/", self._handle_request)
        inner_app.router.add_route("*", "/{path_info:.*}", self._handle_request)

        runner = web.AppRunner(inner_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        await site.start()
        self._runner = runner

        # Resolve actual port (important when port=0 lets OS pick)
        if site._server and site._server.sockets:
            self._port = site._server.sockets[0].getsockname()[1]

        self._do_notify({"port": self._port}, self.uuid)

    async def _handle_request(self, request: web.Request) -> web.Response:
        if self._bypass:
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"error": "http-server-subservices is bypassed"}),
            )

        if self._mode == "process_on_data":
            process_input = self._latest_data
            output: Any = process_input
        else:
            process_input = {"path": request.path, "method": request.method}
            output = self._process_session_input(process_input)

        self._do_notify(
            {"__internal": {"state": "call-process", "data": process_input}},
            self.uuid,
        )

        if self._host:
            output = self._host.process_from(
                self.uuid,
                output,
                lambda n: self._do_notify(n.payload, n.instance_id),
            )
            self._host.emit_result(output)

        self._do_notify(
            {"__internal": {"state": "call-process-finished", "data": output}},
            self.uuid,
        )

        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps(output if output is not None else None),
        )

    # ── Pipeline helpers ───────────────────────────────────────────────────────

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
