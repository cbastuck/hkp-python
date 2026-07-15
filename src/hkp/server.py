from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Coroutine

import aiohttp_cors
from aiohttp import WSMsgType, web

from .auth import (
    AllowedOrigins,
    AuthConfig,
    AuthenticatedUser,
    Authenticator,
    is_origin_allowed,
)
from .data import BinaryData, FloatRingBuffer, NullData, TextData, UndefinedData
from .runtime import HostedRuntime, HostedServiceFactory, RuntimeApp
from .yas import (
    MessagePurpose,
    YasError,
    deserialize_message,
    is_yas_message,
    serialize_message,
)
from .services.http_server import (
    HTTP_SERVER_SUBSERVICES_DESCRIPTOR,
    HttpServerSubservicesService,
)
from .services.map_service import MAP_DESCRIPTOR, MapService
from .services.monitor import MONITOR_DESCRIPTOR, MonitorService
from .services.speech_to_text import SPEECH_TO_TEXT_DESCRIPTOR, SpeechToTextService
from .services.text_generation import TEXT_GENERATION_DESCRIPTOR, TextGenerationService
from .services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
from .services.timer import TIMER_DESCRIPTOR, TimerService
from .types import (
    JsonRecord,
    RuntimeConfiguration,
    RuntimeNotification,
    ServiceConfiguration,
)


# Typed request-storage key on aiohttp >= 3.13; plain string on older versions.
_AUTHENTICATED_USER_KEY: Any = (
    web.RequestKey("authenticated_user", AuthenticatedUser)
    if hasattr(web, "RequestKey")
    else "authenticated_user"
)


@dataclass
class SessionToken:
    """A coordinator session token, bound to the user it was minted for and the
    runtime it grants access to."""

    sub: str
    runtime_id: str


class RuntimeServer:
    def __init__(self, options: dict[str, Any]) -> None:
        self._external_host: str = options.get("external_host", "127.0.0.1")
        self._allowed_origins: AllowedOrigins = options.get("allowed_origins", "*")
        # Tests and local dev default to no auth; __main__.py always resolves an
        # explicit config and fails closed for non-loopback binds (see
        # resolve_server_auth_config).
        self._auth_config: AuthConfig = options.get("auth") or AuthConfig(mode="none")
        # Coordinator session tokens this runtime has issued (see POST
        # .../session-token). Opaque, in-memory, and bound to the minting user —
        # so they resolve back to a real `sub`, not an unscoped superuser. They
        # live only as long as this process: if the runtime dies, the
        # coordinator must re-provision (which needs a live user JWT).
        self._session_tokens: dict[str, SessionToken] = {}
        self._authenticator = Authenticator(
            self._auth_config,
            resolve_opaque_token=self._resolve_session_token,
        )

        factories = {
            MONITOR_DESCRIPTOR.service_id: HostedServiceFactory(
                MONITOR_DESCRIPTOR,
                lambda cfg, _cs: MonitorService(cfg),
            ),
            MAP_DESCRIPTOR.service_id: HostedServiceFactory(
                MAP_DESCRIPTOR,
                lambda cfg, _cs: MapService(cfg),
            ),
            SUB_SERVICE_DESCRIPTOR.service_id: HostedServiceFactory(
                SUB_SERVICE_DESCRIPTOR,
                lambda cfg, cs: SubService(cfg, cs),
            ),
            HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id: HostedServiceFactory(
                HTTP_SERVER_SUBSERVICES_DESCRIPTOR,
                lambda cfg, cs: HttpServerSubservicesService(cfg, cs),
            ),
            TIMER_DESCRIPTOR.service_id: HostedServiceFactory(
                TIMER_DESCRIPTOR,
                lambda cfg, _cs: TimerService(cfg),
            ),
            SPEECH_TO_TEXT_DESCRIPTOR.service_id: HostedServiceFactory(
                SPEECH_TO_TEXT_DESCRIPTOR,
                lambda cfg, _cs: SpeechToTextService(cfg),
            ),
            TEXT_GENERATION_DESCRIPTOR.service_id: HostedServiceFactory(
                TEXT_GENERATION_DESCRIPTOR,
                lambda cfg, _cs: TextGenerationService(cfg),
            ),
        }

        self.runtime_app = RuntimeApp(factories)
        self._runtime_sockets: dict[str, set[web.WebSocketResponse]] = {}
        self._app = self._build_app()
        self._runner: web.AppRunner | None = None
        self._port = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        # Single worker so pipeline processing stays serialized (services are
        # not thread-safe) while long-running work (ML inference) does not
        # block the event loop.
        self._process_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hkp-process"
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self, port: int = 0, host: str = "127.0.0.1") -> dict[str, Any]:
        self._loop = asyncio.get_running_loop()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()

        if site._server and site._server.sockets:
            self._port = site._server.sockets[0].getsockname()[1]

        base_url = f"http://{self._external_host}:{self._port}"
        return {"host": host, "port": self._port, "base_url": base_url}

    async def stop(self) -> None:
        for sockets in self._runtime_sockets.values():
            for ws in list(sockets):
                await ws.close()
        self._runtime_sockets.clear()

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._process_executor.shutdown(wait=False, cancel_futures=True)

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _resolve_session_token(self, token: str) -> AuthenticatedUser | None:
        session = self._session_tokens.get(token)
        return AuthenticatedUser(sub=session.sub) if session else None

    def _purge_session_tokens(self, runtime_id: str) -> None:
        """Drop any session tokens a runtime issued, so a dead runtime's tokens
        can't linger as valid credentials."""
        for token, info in list(self._session_tokens.items()):
            if info.runtime_id == runtime_id:
                del self._session_tokens[token]

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler: Any) -> web.Response:
        """Authenticate every request with the same rules as hkp-node: HTTP
        routes take the token from the Authorization header; WebSocket
        handshakes may also carry it as ?access_token= (browsers can't set
        headers on a WS upgrade) and are additionally Origin-checked (CSWSH
        protection)."""
        if request.method == "OPTIONS":
            # CORS preflights are unauthenticated by design.
            return await handler(request)

        is_upgrade = request.headers.get("Upgrade", "").strip().lower() == "websocket"

        header = request.headers.get("Authorization")
        token = header[7:] if header and header.startswith("Bearer ") else None
        if is_upgrade:
            if not is_origin_allowed(
                request.headers.get("Origin"), self._allowed_origins
            ):
                raise web.HTTPForbidden()
            # Non-browser clients (the coordinator) use the standard
            # Authorization header, keeping the token out of URLs/logs.
            token = token or request.query.get("access_token")

        user = await self._authenticator.verify_token(token)
        if user is None:
            raise web.HTTPUnauthorized()
        request[_AUTHENTICATED_USER_KEY] = user
        return await handler(request)

    # ── App construction ───────────────────────────────────────────────────────

    def _build_app(self) -> web.Application:
        # Error middleware is outermost so it also renders auth failures.
        app = web.Application(middlewares=[_error_middleware, self._auth_middleware])

        app.router.add_get("/runtimes", self._get_runtimes)
        app.router.add_post("/runtimes", self._post_runtimes)
        app.router.add_delete("/runtimes", self._delete_runtimes)

        app.router.add_get("/runtimes/{runtime_id}", self._get_runtime)
        app.router.add_delete("/runtimes/{runtime_id}", self._delete_runtime)
        app.router.add_post(
            "/runtimes/{runtime_id}/session-token", self._mint_session_token
        )
        app.router.add_post("/runtimes/{runtime_id}/rearrange", self._rearrange_runtime)
        app.router.add_post("/runtimes/{runtime_id}", self._process_runtime)

        app.router.add_get("/runtimes/{runtime_id}/services", self._get_services)
        app.router.add_post("/runtimes/{runtime_id}/services", self._post_service)
        app.router.add_delete(
            "/runtimes/{runtime_id}/services/{instance_id}", self._delete_service
        )
        app.router.add_post(
            "/runtimes/{runtime_id}/services/{instance_id}", self._configure_service
        )
        app.router.add_get(
            "/runtimes/{runtime_id}/services/{instance_id}", self._get_service
        )
        app.router.add_get(
            "/runtimes/{runtime_id}/services/{instance_id}/property/{property_id}",
            self._get_service_property,
        )

        # WebSocket endpoint — matches /{runtimeId}
        app.router.add_get("/{runtime_id}", self._websocket_handler)

        # CORS — honour the allowed-origins list and let the browser send the
        # Authorization header on authenticated requests.
        cors_origins = (
            ["*"] if self._allowed_origins == "*" else list(self._allowed_origins)
        )
        cors = aiohttp_cors.setup(
            app,
            defaults={
                origin: aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers=["Content-Type", "Authorization"],
                    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                )
                for origin in cors_origins
            },
        )
        for route in list(app.router.routes()):
            try:
                cors.add(route)
            except ValueError:
                pass  # some routes (e.g. OPTIONS added by cors itself) may already be registered

        return app

    # ── URL helper ─────────────────────────────────────────────────────────────

    def _runtime_output_url(self, runtime_id: str) -> str:
        return f"ws://{self._external_host}:{self._port}/{runtime_id}"

    def _serialize_runtime(self, runtime: HostedRuntime) -> dict[str, Any]:
        descriptor = runtime.serialize(self._runtime_output_url(runtime.id))
        return _descriptor_to_dict(descriptor)

    # ── Processing ─────────────────────────────────────────────────────────────

    async def _process_off_loop(self, runtime: HostedRuntime, body: Any) -> Any:
        """Run the pipeline in the worker thread so slow services (e.g. ML
        inference) don't stall the event loop."""
        return await asyncio.get_running_loop().run_in_executor(
            self._process_executor, runtime.process, body, lambda _n: None
        )

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule a coroutine on the server loop; safe from worker threads."""
        loop = self._loop
        if loop is None:
            coro.close()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            asyncio.ensure_future(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

    # ── Notification / result helpers ──────────────────────────────────────────

    def _send_notification(
        self, runtime_id: str, notification: RuntimeNotification
    ) -> None:
        sockets = self._runtime_sockets.get(runtime_id)
        if not sockets:
            return
        message = json.dumps(
            {
                "type": "notification",
                "instanceId": notification.instance_id,
                "value": json.dumps(notification.payload, default=_json_placeholder),
            }
        )
        for ws in list(sockets):
            if not ws.closed:
                self._spawn(ws.send_str(message))

    def _send_result(self, runtime_id: str, result: Any) -> None:
        sockets = self._runtime_sockets.get(runtime_id)
        if not sockets:
            return
        if _is_binary_result(result):
            frame = serialize_message(result, purpose=MessagePurpose.RESULT)
            for ws in list(sockets):
                if not ws.closed:
                    self._spawn(ws.send_bytes(frame))
            return
        message = json.dumps({"type": "result", "data": _jsonable_result(result)})
        for ws in list(sockets):
            if not ws.closed:
                self._spawn(ws.send_str(message))

    def _register_runtime_targets(self, runtime: HostedRuntime) -> None:
        runtime_id = runtime.id
        runtime.register_notification_target(
            lambda n: self._send_notification(runtime_id, n)
        )
        runtime.register_result_target(
            lambda result: self._send_result(runtime_id, result)
        )

    # ── /runtimes handlers ─────────────────────────────────────────────────────

    async def _get_runtimes(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "runtimes": [
                    self._serialize_runtime(rt) for rt in self.runtime_app.get_runtimes()
                ],
                "registry": self.runtime_app.get_registry(),
            }
        )

    async def _post_runtimes(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()

        payloads = body if isinstance(body, list) else [body]
        runtimes = []

        for payload in payloads:
            config = _validate_runtime_configuration(payload)
            if config is None:
                raise web.HTTPBadRequest()
            runtime = self.runtime_app.create_runtime(config)
            self._register_runtime_targets(runtime)
            runtimes.append(self._serialize_runtime(runtime))

        return web.json_response(
            {"runtimes": runtimes, "registry": self.runtime_app.get_registry()}
        )

    async def _delete_runtimes(self, request: web.Request) -> web.Response:
        self.runtime_app.remove_all_runtimes()
        self._session_tokens.clear()
        return web.Response(status=200)

    # ── /runtimes/{id} handlers ────────────────────────────────────────────────

    async def _get_runtime(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        return web.json_response(self._serialize_runtime(runtime))

    async def _delete_runtime(self, request: web.Request) -> web.Response:
        runtime_id = request.match_info["runtime_id"]
        if not self.runtime_app.remove_runtime(runtime_id):
            raise web.HTTPNotFound()
        self._purge_session_tokens(runtime_id)
        return web.json_response({"id": runtime_id})

    async def _mint_session_token(self, request: web.Request) -> web.Response:
        """Mint a coordinator session token for a runtime. Gated by the normal
        auth middleware, so the caller must present a valid user JWT (the
        "bootstrap"). The returned opaque token is bound to that user and this
        runtime, and the coordinator then uses it for its long-lived machine
        calls without needing a user JWT that would expire.

        Limitation (v1, same as hkp-node): tokens live only in this process. If
        the runtime restarts the token is gone and the coordinator must
        re-provision — which requires a live user JWT.
        """
        runtime = self._get_runtime_or_404(request)
        user = request.get(_AUTHENTICATED_USER_KEY)
        sub = user.sub if user else "anonymous"
        token = secrets.token_hex(32)
        self._session_tokens[token] = SessionToken(sub=sub, runtime_id=runtime.id)
        return web.json_response({"token": token})

    async def _rearrange_runtime(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        if not isinstance(body, list) or not all(isinstance(e, str) for e in body):
            raise web.HTTPBadRequest()
        if not runtime.rearrange_services(body):
            raise web.HTTPBadRequest()
        return web.json_response(self._serialize_runtime(runtime))

    async def _process_runtime(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        raw = await request.read()

        if request.content_type == "application/octet-stream" or is_yas_message(raw):
            try:
                body = deserialize_message(raw).data
            except YasError:
                raise web.HTTPBadRequest()
        else:
            try:
                body = json.loads(raw)
            except Exception:
                raise web.HTTPBadRequest()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest()

        result = await self._process_off_loop(runtime, body)
        if _is_binary_result(result):
            return web.Response(
                body=serialize_message(result, purpose=MessagePurpose.RESULT),
                content_type="application/octet-stream",
            )
        return web.json_response(_jsonable_result(result))

    # ── /runtimes/{id}/services handlers ──────────────────────────────────────

    async def _get_services(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        return web.json_response(
            [_service_descriptor_to_dict(s) for s in runtime.list_services()]
        )

    async def _post_service(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        config = _validate_service_configuration(body)
        if config is None:
            raise web.HTTPBadRequest()
        try:
            state = runtime.add_service(config)
        except Exception:
            raise web.HTTPBadRequest()
        return web.json_response(state)

    async def _delete_service(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        instance_id = request.match_info["instance_id"]
        if not runtime.remove_service(instance_id):
            raise web.HTTPNotFound()
        return web.json_response(self._serialize_runtime(runtime))

    async def _configure_service(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        instance_id = request.match_info["instance_id"]
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        if not isinstance(body, dict):
            raise web.HTTPBadRequest()
        state = runtime.configure_service(instance_id, body)
        if state is None:
            raise web.HTTPNotFound()
        # Poll until port is assigned for http-server-subservices with bypass=False, port=0
        state = await _wait_for_service_activation_state(runtime, instance_id)
        return web.json_response(state)

    async def _get_service(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        instance_id = request.match_info["instance_id"]
        svc = runtime.get_service(instance_id)
        if not svc:
            raise web.HTTPNotFound()
        return web.json_response(svc.get_state())

    async def _get_service_property(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        instance_id = request.match_info["instance_id"]
        property_id = request.match_info["property_id"]
        svc = runtime.get_service(instance_id)
        if not svc:
            raise web.HTTPNotFound()
        state = svc.get_state()
        if property_id not in state:
            raise web.HTTPNotFound()
        return web.json_response(state[property_id])

    # ── WebSocket handler ──────────────────────────────────────────────────────

    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        runtime_id = request.match_info["runtime_id"]
        if not self.runtime_app.get_runtime(runtime_id):
            raise web.HTTPNotFound()

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        sockets = self._runtime_sockets.setdefault(runtime_id, set())
        sockets.add(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "readwrite":
                        continue  # protocol handshake, nothing to do

                    # Mirror hkp-node: any JSON params value is processable —
                    # a plain-text Injector ships a bare string, not an object.
                    if data.get("type") == "processRuntime" and "params" in data:
                        runtime = self.runtime_app.get_runtime(runtime_id)
                        if runtime:
                            result = await self._process_off_loop(
                                runtime, data["params"]
                            )
                            if not ws.closed:
                                if _is_binary_result(result):
                                    await ws.send_bytes(
                                        serialize_message(
                                            result, purpose=MessagePurpose.RESULT
                                        )
                                    )
                                else:
                                    await ws.send_str(
                                        json.dumps(
                                            {
                                                "type": "result",
                                                "data": _jsonable_result(result),
                                            }
                                        )
                                    )
                elif msg.type == WSMsgType.BINARY:
                    # Binary frames are YAS-encoded data to process (the
                    # frontend ships FloatRingBuffer etc. this way).
                    try:
                        message = deserialize_message(msg.data)
                    except YasError:
                        continue
                    runtime = self.runtime_app.get_runtime(runtime_id)
                    if runtime:
                        result = await self._process_off_loop(runtime, message.data)
                        if not ws.closed:
                            if _is_binary_result(result):
                                await ws.send_bytes(
                                    serialize_message(
                                        result, purpose=MessagePurpose.RESULT
                                    )
                                )
                            else:
                                await ws.send_str(
                                    json.dumps(
                                        {
                                            "type": "result",
                                            "data": _jsonable_result(result),
                                        }
                                    )
                                )
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            sockets.discard(ws)
            if not sockets:
                self._runtime_sockets.pop(runtime_id, None)

        return ws

    # ── Utility ────────────────────────────────────────────────────────────────

    def _get_runtime_or_404(self, request: web.Request) -> HostedRuntime:
        runtime_id = request.match_info["runtime_id"]
        runtime = self.runtime_app.get_runtime(runtime_id)
        if not runtime:
            raise web.HTTPNotFound()
        return runtime


# ── Module-level factory ───────────────────────────────────────────────────────


def create_runtime_server(options: dict[str, Any] | None = None) -> RuntimeServer:
    return RuntimeServer(options or {})


# ── Result / notification encoding helpers ─────────────────────────────────────


def _is_binary_result(result: Any) -> bool:
    return isinstance(result, (FloatRingBuffer, BinaryData, TextData))


def _jsonable_result(result: Any) -> Any:
    if isinstance(result, (NullData, UndefinedData)):
        return None
    return result


def _json_placeholder(value: Any) -> Any:
    """json.dumps fallback so binary pipeline data can appear in notifications."""
    if isinstance(value, FloatRingBuffer):
        return {
            "type": "FloatRingBuffer",
            "numSamples": value.num_samples,
            "id": value.id,
            "ts": value.ts,
        }
    if isinstance(value, BinaryData):
        return {"type": "BinaryData", "size": len(value.data)}
    if isinstance(value, TextData):
        return value.text
    if isinstance(value, (NullData, UndefinedData)):
        return None
    return repr(value)


# ── Middleware ─────────────────────────────────────────────────────────────────


@web.middleware
async def _error_middleware(request: web.Request, handler: Any) -> web.Response:
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as exc:
        # Don't leak internal error details (paths, stack hints) to clients.
        print(f"[server] Unhandled request error: {exc!r}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": "Internal Server Error"}),
        )


# ── Validation helpers ─────────────────────────────────────────────────────────


def _validate_runtime_configuration(value: Any) -> RuntimeConfiguration | None:
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("id"), str) or not isinstance(value.get("name"), str):
        return None
    if not isinstance(value.get("services"), list):
        return None

    services = []
    for entry in value["services"]:
        svc = _validate_service_configuration(entry)
        if svc is None:
            return None
        services.append(svc)

    return RuntimeConfiguration(
        id=value["id"],
        name=value["name"],
        board_name=value.get("boardName", ""),
        services=services,
    )


def _validate_service_configuration(value: Any) -> ServiceConfiguration | None:
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("serviceId"), str) or not isinstance(value.get("uuid"), str):
        return None
    state = value.get("state")
    if state is not None and not isinstance(state, dict):
        return None
    return ServiceConfiguration(
        service_id=value["serviceId"],
        uuid=value["uuid"],
        name=value.get("name") if isinstance(value.get("name"), str) else None,
        service_name=value.get("serviceName") if isinstance(value.get("serviceName"), str) else None,
        state=state,
    )


# ── Async polling helper ───────────────────────────────────────────────────────


async def _wait_for_service_activation_state(
    runtime: HostedRuntime, instance_id: str
) -> JsonRecord:
    max_attempts = 20
    delay = 0.01

    for _ in range(max_attempts):
        svc = runtime.get_service(instance_id)
        if not svc:
            return {}
        state = svc.get_state()
        if state.get("bypass") is False and state.get("port") == 0:
            await asyncio.sleep(delay)
            continue
        return state

    svc = runtime.get_service(instance_id)
    return svc.get_state() if svc else {}


# ── Serialisation helpers ──────────────────────────────────────────────────────


def _descriptor_to_dict(descriptor: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": descriptor.id,
        "name": descriptor.name,
        "boardName": descriptor.board_name,
        "services": [_service_descriptor_to_dict(s) for s in descriptor.services],
        "inputs": descriptor.inputs,
    }
    if descriptor.output_url is not None:
        result["outputUrl"] = descriptor.output_url
    return result


def _service_descriptor_to_dict(descriptor: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "serviceId": descriptor.service_id,
        "serviceName": descriptor.service_name,
        "uuid": descriptor.uuid,
        "state": descriptor.state,
    }
    if descriptor.version is not None:
        result["version"] = descriptor.version
    if descriptor.capabilities is not None:
        result["capabilities"] = descriptor.capabilities
    return result
