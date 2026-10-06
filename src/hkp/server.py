from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import aiohttp_cors
from aiohttp import WSMsgType, web

from .auth import (
    AllowedOrigins,
    AuthConfig,
    AuthenticatedUser,
    Authenticator,
    is_origin_allowed,
    owner_key_of,
)
from .data import BinaryData, FloatRingBuffer, NullData, TextData, UndefinedData
from .mounts import MOUNT_PREFIX, MountRegistry, RuntimeMounts
from .runtime import set_server_loop
from .secrets import SecretEntry, read_secrets_payload, referenced_secrets
from .assets import read_assets_payload
from .services.asset import ASSET_DESCRIPTOR, AssetService
from .coordinator_links import (
    CoordinatorLinks,
    FileLinkStore,
    LinkRecord,
    MemoryLinkStore,
)
from .runtime import (
    board_space,
    context_for_client,
    context_from_link,
    context_to_wire,
    new_run,
    HostedRuntime,
    HostedServiceFactory,
    LOG_LEVELS,
    RuntimeApp,
    TenantRuntimes,
)
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
from .services.http_client import HTTP_CLIENT_DESCRIPTOR, HttpClientService
from .services.stopper import STOPPER_DESCRIPTOR, StopperService
from .services.hold import HOLD_DESCRIPTOR, HoldService
from .services.monitor import MONITOR_DESCRIPTOR, MonitorService
from .services.speech_to_text import SPEECH_TO_TEXT_DESCRIPTOR, SpeechToTextService
from .services.text_generation import TEXT_GENERATION_DESCRIPTOR, TextGenerationService
from .services.skill_router import SKILL_ROUTER_DESCRIPTOR, SkillRouterService
from .services.text_to_speech import TEXT_TO_SPEECH_DESCRIPTOR, TextToSpeechService
from .services.audio_encode import AUDIO_ENCODE_DESCRIPTOR, AudioEncodeService
from .services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
from .services.join import JOIN_DESCRIPTOR, JoinService
from .services.tracks import TRACKS_DESCRIPTOR, TracksService
from .services.timer import (
    TIMER_DESCRIPTOR,
    TIMER_LEGACY_SERVICE_ID,
    TimerService,
)
from .types import (
    ProcessContext,
    JsonRecord,
    LogEntry,
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


#: Largest accepted request body on a public service endpoint (25 MB).
DEFAULT_MAX_REQUEST_BODY_BYTES = 25 * 1024 * 1024

#: Which runtime server this is, reported beside the runtimes so a client can
#: tell remote runtimes apart without reading their address.
RUNTIME_SERVER_KIND = "python"


def _tenant_key(owner: str, runtime_id: str) -> str:
    """Runtime ids are unique per tenant, not globally, so anything keyed by
    runtime outside a tenant view (socket sets) must be keyed by both. NUL
    cannot occur in either part, so the join is unambiguous."""
    return f"{owner}\x00{runtime_id}"


@dataclass
class SessionToken:
    """A coordinator session token, bound to the user it was minted for and the
    runtime it grants access to."""

    sub: str
    runtime_id: str


class _QuotaError(Exception):
    """Refused for being over a per-tenant limit; says what to tell the caller."""


class RuntimeServer:
    #: Which runtime server this is, as a coordinator link reports it.
    kind = RUNTIME_SERVER_KIND

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
        # Per-tenant limits. Runtimes, services and timers all consume resources
        # on a shared host, so without a cap one tenant can degrade the server
        # for everyone. 0 / unset means unlimited, which is the right default
        # for a single-user instance and for local development.
        quotas: dict[str, Any] = options.get("quotas") or {}
        self._max_runtimes_per_user: int = quotas.get("max_runtimes_per_user") or 0
        self._max_services_per_runtime: int = (
            quotas.get("max_services_per_runtime") or 0
        )
        self._min_timer_interval_ms: int = quotas.get("min_timer_interval_ms") or 0
        # Unlike the other quotas this defaults to a real value rather than
        # unlimited: service endpoints take no token, so "off" would make the
        # dangerous choice the automatic one.
        self._max_request_body_bytes: int = (
            quotas["max_request_body_bytes"]
            if "max_request_body_bytes" in quotas
            else DEFAULT_MAX_REQUEST_BODY_BYTES
        )
        # Builds the authenticator, defaulting to a JWKS-backed one for `auth`.
        # Overriding it replaces only how a raw token is verified — the server
        # still resolves its own session tokens first, by passing the resolver it
        # owns into whatever this returns.
        build_authenticator = options.get("build_authenticator")
        self._authenticator = (
            build_authenticator(self._resolve_session_token)
            if build_authenticator
            else Authenticator(
                self._auth_config,
                resolve_opaque_token=self._resolve_session_token,
            )
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
            JOIN_DESCRIPTOR.service_id: HostedServiceFactory(
                JOIN_DESCRIPTOR,
                lambda cfg, cs: JoinService(cfg, cs),
            ),
            TRACKS_DESCRIPTOR.service_id: HostedServiceFactory(
                TRACKS_DESCRIPTOR,
                lambda cfg, cs: TracksService(cfg, cs),
            ),
            HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id: HostedServiceFactory(
                HTTP_SERVER_SUBSERVICES_DESCRIPTOR,
                lambda cfg, cs: HttpServerSubservicesService(
                    cfg, cs, self._max_request_body_bytes
                ),
            ),
            HTTP_CLIENT_DESCRIPTOR.service_id: HostedServiceFactory(
                HTTP_CLIENT_DESCRIPTOR,
                lambda cfg, _cs: HttpClientService(cfg),
            ),
            STOPPER_DESCRIPTOR.service_id: HostedServiceFactory(
                STOPPER_DESCRIPTOR,
                lambda cfg, _cs: StopperService(cfg),
            ),
            HOLD_DESCRIPTOR.service_id: HostedServiceFactory(
                HOLD_DESCRIPTOR,
                lambda cfg, _cs: HoldService(cfg),
            ),
            ASSET_DESCRIPTOR.service_id: HostedServiceFactory(
                ASSET_DESCRIPTOR,
                lambda cfg, _cs: AssetService(cfg),
            ),
            TIMER_DESCRIPTOR.service_id: HostedServiceFactory(
                TIMER_DESCRIPTOR,
                lambda cfg, _cs: TimerService(cfg, self._min_timer_interval_ms),
            ),
            # Same service, under the id it answered to before it matched
            # hkp-node and hkp-rt. Boards saved against the old id still load;
            # the registry advertises only the canonical one.
            TIMER_LEGACY_SERVICE_ID: HostedServiceFactory(
                TIMER_DESCRIPTOR,
                lambda cfg, _cs: TimerService(cfg, self._min_timer_interval_ms),
            ),
            SPEECH_TO_TEXT_DESCRIPTOR.service_id: HostedServiceFactory(
                SPEECH_TO_TEXT_DESCRIPTOR,
                lambda cfg, _cs: SpeechToTextService(cfg),
            ),
            TEXT_GENERATION_DESCRIPTOR.service_id: HostedServiceFactory(
                TEXT_GENERATION_DESCRIPTOR,
                lambda cfg, _cs: TextGenerationService(cfg),
            ),
            TEXT_TO_SPEECH_DESCRIPTOR.service_id: HostedServiceFactory(
                TEXT_TO_SPEECH_DESCRIPTOR,
                lambda cfg, _cs: TextToSpeechService(cfg),
            ),
            AUDIO_ENCODE_DESCRIPTOR.service_id: HostedServiceFactory(
                AUDIO_ENCODE_DESCRIPTOR,
                lambda cfg, _cs: AudioEncodeService(cfg),
            ),
            SKILL_ROUTER_DESCRIPTOR.service_id: HostedServiceFactory(
                SKILL_ROUTER_DESCRIPTOR,
                lambda cfg, _cs: SkillRouterService(cfg),
            ),
        }

        # Public service endpoints. Declared before the runtime app because
        # runtimes hand mounts to their services as they are created.
        # Keys the derivation of public endpoint addresses. Given none, the
        # registry draws one for this process, so addresses work but change on
        # restart; `__main__` persists one so they do not.
        self._mounts = MountRegistry(
            self._public_mount_url, options.get("mount_secret")
        )
        self.runtime_app = RuntimeApp(
            factories,
            mounts_for=lambda owner, runtime_id, space: RuntimeMounts(
                mount=lambda service_uuid, handler, board_name="", mount_name=None: (
                    self._mounts.register(
                        owner,
                        runtime_id,
                        service_uuid,
                        handler,
                        board_name=board_name,
                        mount_name=mount_name,
                        space=space,
                    )
                )
            ),
        )
        # Keyed by _tenant_key(owner, runtime_id), not runtime id alone: ids are
        # unique per tenant, not globally.
        self._runtime_sockets: dict[str, set[web.WebSocketResponse]] = {}
        # This server's connections to coordinators, and the tickets it keeps
        # to reconnect with. A path keeps them on disk; absent or empty means
        # memory, so the links work and are not re-established after a restart.
        link_store = options.get("coordinator_links")
        self.coordinator_links = CoordinatorLinks(
            self,
            (
                (FileLinkStore(link_store) if link_store else MemoryLinkStore())
                if isinstance(link_store, str) or link_store is None
                else link_store
            ),
            options.get("coordinator_link_options"),
            spawn=self._spawn,
            dumps=lambda value: json.dumps(value, default=_json_placeholder),
        )
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
        # Services schedule their own work — a request, a timer — from a pass
        # that runs on a worker thread, where there is no loop to schedule on.
        set_server_loop(self._loop)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()

        if site._server and site._server.sockets:
            self._port = site._server.sockets[0].getsockname()[1]

        base_url = f"http://{self._external_host}:{self._port}"
        return {"host": host, "port": self._port, "base_url": base_url}

    async def stop(self) -> None:
        await self.coordinator_links.stop()
        # A copy: a socket's handler drops its entry while its close is awaited.
        for sockets in list(self._runtime_sockets.values()):
            for ws in list(sockets):
                await ws.close()
        self._runtime_sockets.clear()

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._process_executor.shutdown(wait=False, cancel_futures=True)

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _at_quota(self, count: int, limit: int) -> bool:
        """True when adding one more to ``count`` would pass ``limit``."""
        return limit > 0 and count >= limit

    def _exceeds_quota(self, count: int, limit: int) -> bool:
        """True when ``count`` items is already more than ``limit`` allows."""
        return limit > 0 and count > limit

    def _tenant_of(self, request: web.Request) -> TenantRuntimes:
        """The caller's runtime namespace. Every route resolves through it."""
        return self.runtime_app.for_owner(self._owner_of(request))

    def _owner_of(self, request: web.Request) -> str:
        return owner_key_of(request.get(_AUTHENTICATED_USER_KEY))

    def _resolve_session_token(self, token: str) -> AuthenticatedUser | None:
        session = self._session_tokens.get(token)
        return AuthenticatedUser(sub=session.sub) if session else None

    def _purge_session_tokens(self, owner: str, runtime_id: str) -> None:
        """Drop any session tokens a runtime issued, so a dead runtime's tokens
        can't linger as valid credentials."""
        for token, info in list(self._session_tokens.items()):
            if info.sub == owner and info.runtime_id == runtime_id:
                del self._session_tokens[token]

    def _purge_owner_session_tokens(self, owner: str) -> None:
        for token, info in list(self._session_tokens.items()):
            if info.sub == owner:
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

        # Service endpoints exist to be called by outside parties (webhooks,
        # uploads) that hold no token; their unguessable mount id is what gates
        # access. Checked before anything else so no auth failure can shadow them.
        if request.path.startswith(f"{MOUNT_PREFIX}/"):
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

        user = await self._authenticator.authorize_owner(token)
        if user is None:
            raise web.HTTPUnauthorized()
        request[_AUTHENTICATED_USER_KEY] = user
        return await handler(request)

    # ── App construction ───────────────────────────────────────────────────────

    def _build_app(self) -> web.Application:
        # Error middleware is outermost so it also renders auth failures.
        app = web.Application(middlewares=[_error_middleware, self._auth_middleware])

        app.router.add_route("*", f"{MOUNT_PREFIX}/{{mount_id}}", self._mounts.handle)
        app.router.add_route(
            "*", f"{MOUNT_PREFIX}/{{mount_id}}/{{sub_path:.*}}", self._mounts.handle
        )

        app.router.add_post("/coordinator-links", self._post_coordinator_link)
        app.router.add_get("/coordinator-links", self._get_coordinator_links)
        app.router.add_delete(
            "/coordinator-links/{board_name}/{runtime_id}",
            self._delete_coordinator_link,
        )

        app.router.add_get("/runtimes", self._get_runtimes)
        app.router.add_post("/runtimes", self._post_runtimes)
        app.router.add_delete("/runtimes", self._delete_runtimes)

        app.router.add_get("/runtimes/{runtime_id}", self._get_runtime)
        app.router.add_delete("/runtimes/{runtime_id}", self._delete_runtime)
        app.router.add_post(
            "/runtimes/{runtime_id}/session-token", self._mint_session_token
        )
        app.router.add_post("/runtimes/{runtime_id}/secrets", self._post_secrets)
        app.router.add_post("/runtimes/{runtime_id}/assets", self._post_assets)
        app.router.add_get(
            "/runtimes/{runtime_id}/assets/{asset_id}", self._check_asset
        )
        app.router.add_post("/runtimes/{runtime_id}/rearrange", self._rearrange_runtime)
        app.router.add_patch("/runtimes/{runtime_id}/state", self._patch_runtime_state)
        app.router.add_post("/runtimes/{runtime_id}", self._process_runtime)

        app.router.add_get("/runtimes/{runtime_id}/services", self._get_services)
        app.router.add_post("/runtimes/{runtime_id}/services", self._post_service)
        app.router.add_delete(
            "/runtimes/{runtime_id}/services/{instance_id}", self._delete_service
        )
        app.router.add_post(
            "/runtimes/{runtime_id}/services/{instance_id}", self._configure_service
        )
        app.router.add_post(
            "/runtimes/{runtime_id}/services/{instance_id}/process",
            self._process_service,
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

    def _public_mount_url(self, mount_path: str) -> str | None:
        if not self._port:
            return None
        return f"http://{self._external_host}:{self._port}{mount_path}"

    def _runtime_output_url(self, runtime_id: str) -> str:
        return f"ws://{self._external_host}:{self._port}/{runtime_id}"

    def _serialize_runtime(self, runtime: HostedRuntime) -> dict[str, Any]:
        descriptor = runtime.serialize(self._runtime_output_url(runtime.id))
        return _descriptor_to_dict(descriptor)

    # ── Processing ─────────────────────────────────────────────────────────────

    async def _process_off_loop(
        self,
        runtime: HostedRuntime,
        body: Any,
        context: ProcessContext | None = None,
    ) -> Any:
        """Run the pipeline in the worker thread so slow services (e.g. ML
        inference) don't stall the event loop."""
        return await asyncio.get_running_loop().run_in_executor(
            self._process_executor, runtime.process, body, lambda _n: None, context
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
        self, socket_key: str, notification: RuntimeNotification
    ) -> None:
        sockets = self._runtime_sockets.get(socket_key)
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

    def _send_log(self, socket_key: str, entry: LogEntry) -> None:
        """Carry a log entry to whoever is collecting this runtime's output.

        The same socket a notification takes, and for the same reason: it is the
        connection the board's coordinator already holds, authenticated with a
        credential minted to outlive the user's session. An entry differs in
        what it is for — a notification is for whoever is watching, an entry has
        to survive with nobody attached — but not in how it travels.

        Nobody listening is not a reason to buffer: an entry exists to be
        written down by the coordinator, and there is no coordinator here to
        write it. A runtime nothing is collecting from is a runtime whose board
        has no log.
        """
        sockets = self._runtime_sockets.get(socket_key)
        if not sockets:
            return
        # `data` is whatever a service passed, which on this runtime may be a
        # ring buffer or raw bytes — the same placeholder a notification uses,
        # rather than failing to send the entry at all.
        message = json.dumps(
            {"type": "log", "entry": entry.to_wire()}, default=_json_placeholder
        )
        for ws in list(sockets):
            if not ws.closed:
                self._spawn(ws.send_str(message))

    def _send_result(self, socket_key: str, result: Any) -> None:
        sockets = self._runtime_sockets.get(socket_key)
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

    def _register_runtime_targets(
        self, owner: str, runtime: HostedRuntime, board_name: str | None
    ) -> None:
        """Wires what a runtime says to whoever is listening: the sockets
        watching it, or the coordinator it was built for."""
        runtime_id = runtime.id

        if board_name is not None:
            links = self.coordinator_links
            linked = (owner, board_name, runtime_id)

            def to_coordinator_notification(notification: RuntimeNotification) -> None:
                message: dict[str, Any] = {
                    "type": "notification",
                    "serviceUuid": notification.instance_id,
                    "payload": notification.payload,
                }
                # Named with whoever began the run it was raised in, which is
                # how the coordinator knows whose it is to hear. Raised outside
                # a run — a timer, a callback — it names nobody.
                context = runtime.current_context()
                if context is not None and context.caller is not None:
                    message["caller"] = context.caller.to_wire()
                links.emit(*linked, message)

            def to_coordinator_result(
                result: Any, context: ProcessContext | None
            ) -> None:
                # The link sends a value holding bytes as a binary frame. With
                # the run it was emitted in, so the coordinator can tell the
                # next runtime which run this continues and who began it.
                links.emit(*linked, _result_message(result, context))

            def to_coordinator_log(entry: LogEntry) -> None:
                links.emit(*linked, {"type": "log", "entry": entry.to_wire()})

            runtime.register_notification_target(to_coordinator_notification)
            runtime.register_result_target(to_coordinator_result)
            runtime.register_log_target(to_coordinator_log)
            return

        socket_key = _tenant_key(owner, runtime_id)
        runtime.register_notification_target(
            lambda notification: self._send_notification(socket_key, notification)
        )
        runtime.register_result_target(
            lambda result, _context: self._send_result(socket_key, result)
        )
        runtime.register_log_target(lambda entry: self._send_log(socket_key, entry))

    def _provision_runtime(
        self,
        owner: str,
        config: RuntimeConfiguration,
        board_name: str | None = None,
    ) -> HostedRuntime:
        """Builds a runtime for a tenant, replacing anything under that id.

        A runtime a client asked for lives in the tenant's own space and speaks
        to the sockets watching it. One built for a coordinator (``board_name``
        given) lives in its board's space and speaks to that coordinator; see
        ``board_space``.

        Quotas apply only to genuinely new runtimes — re-creating one that
        already exists must not be refused for being over the limit.
        """
        space = owner if board_name is None else board_space(owner, board_name)
        is_new = self.runtime_app.get_runtime(space, config.id) is None
        if is_new and self._at_quota(
            self.runtime_app.count_runtimes(owner), self._max_runtimes_per_user
        ):
            raise _QuotaError(f"Runtime limit reached ({self._max_runtimes_per_user})")
        if self._exceeds_quota(len(config.services), self._max_services_per_runtime):
            raise _QuotaError(
                f"Service limit reached ({self._max_services_per_runtime})"
            )
        runtime = self.runtime_app.create_runtime(owner, config, space)
        self._register_runtime_targets(owner, runtime, board_name)
        return runtime

    def _linked_runtime(
        self, owner: str, board_name: str, runtime_id: str
    ) -> HostedRuntime | None:
        return self.runtime_app.get_runtime(board_space(owner, board_name), runtime_id)

    def _remove_runtime(self, owner: str, runtime_id: str) -> None:
        self.runtime_app.remove_runtime(owner, runtime_id)
        self._mounts.release_runtime(owner, runtime_id)
        self._purge_session_tokens(owner, runtime_id)

    @staticmethod
    def _apply_runtime_state(
        runtime: HostedRuntime, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Applies the parts of a runtime's state that can change while it runs."""
        if isinstance(state.get("logging"), bool):
            runtime.set_logging(state["logging"])
        if state.get("logLevel") in LOG_LEVELS:
            runtime.set_log_level(state["logLevel"])
        if isinstance(state.get("logData"), bool):
            runtime.set_log_data(state["logData"])
        settings = runtime.log_settings()
        return {
            "logging": settings["logging"],
            "logData": settings["log_data"],
            "logLevel": settings["log_level"],
        }

    # ── What a coordinator may do here, over a link this server opened ─────────
    #
    # The operations on one runtime, as the tenant who introduced the link — the
    # same things the REST routes do for a caller holding a token. See
    # coordinator_links.LinkHost.

    def link_registry(self) -> list[Any]:
        return self.runtime_app.get_registry()

    def link_runtime_exists(self, owner: str, board_name: str, runtime_id: str) -> bool:
        return self._linked_runtime(owner, board_name, runtime_id) is not None

    def link_provision(
        self,
        owner: str,
        board_name: str,
        runtime_id: str,
        payload: dict[str, Any],
        secrets: dict[str, SecretEntry],
    ) -> dict[str, Any]:
        config = _validate_runtime_configuration(
            {
                "id": runtime_id,
                "name": payload.get("name"),
                "boardName": board_name,
                # The coordinator's until it says otherwise: a deployed board
                # keeps running with nobody watching.
                "garbageCollected": False,
                "state": payload.get("state"),
                "services": payload.get("services"),
                # The board's assets for this runtime, as the coordinator sends
                # them: descriptors, by id.
                "assets": payload.get("assets"),
            }
        )
        if config is None:
            raise ValueError("The board's description of this runtime is malformed")
        # The values this server was handed for the runtime, by the person's own
        # client. They do not come from the coordinator and never go to it.
        config.secrets = dict(secrets)
        runtime = self._provision_runtime(owner, config, board_name)
        held = set(runtime.secrets().aliases())
        missing = [
            alias
            for alias in referenced_secrets(
                [service.state for service in config.services]
            )
            if alias not in held
        ]
        return {
            "registry": self.runtime_app.get_registry(),
            "services": [
                _service_descriptor_to_dict(s) for s in runtime.list_services()
            ],
            "missingSecrets": missing,
        }

    def link_describe(self, owner: str, board_name: str, runtime_id: str) -> dict[str, Any] | None:
        runtime = self._linked_runtime(owner, board_name, runtime_id)
        if runtime is None:
            return None
        return {
            "services": [
                _service_descriptor_to_dict(s) for s in runtime.list_services()
            ]
        }

    async def link_configure_service(
        self, owner: str, board_name: str, runtime_id: str, service_uuid: str, config: Any
    ) -> Any:
        runtime = self._linked_runtime(owner, board_name, runtime_id)
        if runtime is None:
            raise RuntimeError("the runtime is not running")
        if not isinstance(config, dict):
            raise ValueError("a service is configured with an object")
        if runtime.configure_service(service_uuid, config) is None:
            raise LookupError(f'no service "{service_uuid}"')
        return await _wait_for_service_activation_state(runtime, service_uuid)

    def link_set_state(
        self, owner: str, board_name: str, runtime_id: str, state: dict[str, Any]
    ) -> Any:
        runtime = self._linked_runtime(owner, board_name, runtime_id)
        if runtime is None:
            raise RuntimeError("the runtime is not running")
        return self._apply_runtime_state(runtime, state)

    def link_remove(self, owner: str, board_name: str, runtime_id: str) -> None:
        space = board_space(owner, board_name)
        self.runtime_app.remove_runtime(space, runtime_id)
        self._mounts.release_runtime(space, runtime_id)

    async def link_process(
        self, owner: str, board_name: str, runtime_id: str, params: Any, context: Any
    ) -> dict[str, Any]:
        runtime = self._linked_runtime(owner, board_name, runtime_id)
        if runtime is None:
            raise RuntimeError("the runtime is not running")
        # The coordinator names the run its call belongs to, so that a board
        # spanning several runtimes reads as one trace — and who began it,
        # which is taken as stated on this path and on no other.
        run = context_from_link(context) or new_run()
        result = await self._process_off_loop(runtime, params, run)
        return _result_message(result, run)

    def link_process_service(
        self,
        owner: str,
        board_name: str,
        runtime_id: str,
        service_uuid: str,
        params: Any,
        context: Any,
        done: Callable[[dict[str, Any]], None],
    ) -> None:
        runtime = self._linked_runtime(owner, board_name, runtime_id)
        if runtime is None:
            raise RuntimeError("the runtime is not running")
        if not runtime.get_service(service_uuid):
            raise RuntimeError(f'no service "{service_uuid}"')
        # As on `link_process`: the run and its caller are the coordinator's
        # to state, over this link and nowhere else.
        run = context_from_link(context) or new_run()

        async def work() -> None:
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self._process_executor,
                    runtime.process_at,
                    service_uuid,
                    params,
                    lambda _n: None,
                    run,
                )
            except Exception as err:  # noqa: BLE001 - reported, never raised
                print(
                    f'[coordinator-link] Runtime "{runtime_id}" failed to '
                    f'process at "{service_uuid}": {err}'
                )
                return
            done(_result_message(result, run))

        self._spawn(work())

    # ── /coordinator-links handlers ────────────────────────────────────────────

    async def _post_coordinator_link(self, request: web.Request) -> web.Response:
        """Introduces this server to a coordinator, for one runtime of one board.

        Called by the person's own client while it deploys a board: it has asked
        the coordinator for a ticket and passes it on, over the same session it
        creates runtimes here with. This server then connects to the coordinator
        — the coordinator connects to nothing — and keeps the ticket to
        reconnect with.

        The address dialled is one the caller chose, and the caller is someone
        this server already runs services for; nothing here can be made to reach
        further than they could with a service of their own.

        ``secrets`` are the values for the references that runtime's services
        carry. They are handed to the runtime when the coordinator builds it,
        and are not sent to the coordinator.
        """
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        fields = ("coordinatorUrl", "ticket", "boardName", "runtimeId")
        if not isinstance(body, dict) or not all(
            isinstance(body.get(name), str) and body[name] for name in fields
        ):
            raise web.HTTPBadRequest()
        try:
            await self.coordinator_links.introduce(
                LinkRecord(
                    owner=self._owner_of(request),
                    board_name=body["boardName"],
                    runtime_id=body["runtimeId"],
                    coordinator_url=body["coordinatorUrl"],
                    ticket=body["ticket"],
                ),
                read_secrets_payload(body.get("secrets")),
            )
        except (ConnectionError, ValueError) as err:
            return web.json_response(
                {"error": str(err) or "Could not connect"}, status=502
            )
        return web.json_response({"connected": True}, status=201)

    async def _get_coordinator_links(self, request: web.Request) -> web.Response:
        """The caller's links: which runtimes belong to which board, never a
        ticket."""
        return web.json_response(
            {"links": self.coordinator_links.list(self._owner_of(request))}
        )

    async def _delete_coordinator_link(self, request: web.Request) -> web.Response:
        """Leaves a board: drops the link and the runtime it was for."""
        removed = await self.coordinator_links.remove(
            self._owner_of(request),
            request.match_info["board_name"],
            request.match_info["runtime_id"],
        )
        return web.Response(status=200 if removed else 404)

    # ── /runtimes handlers ─────────────────────────────────────────────────────

    async def _get_runtimes(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "runtimes": [
                    self._serialize_runtime(rt)
                    for rt in self._tenant_of(request).get_runtimes()
                ],
                # The service registry is a property of the build, not a tenant.
                "registry": self.runtime_app.get_registry(),
                "server": RUNTIME_SERVER_KIND,
                # This server can connect to a coordinator when introduced to
                # one; see POST /coordinator-links. Said here so a client can
                # tell before it deploys a board that needs it.
                "coordinatorLinks": True,
                # A runtime built for a coordinator is kept apart from the ones
                # a client creates, so a client deleting its own does not
                # delete a deployed board's. A client checks for this before it
                # deploys: against a server that keeps them together, leaving
                # the board it deployed would stop it.
                "boardRuntimes": True,
            }
        )

    async def _post_runtimes(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()

        owner = self._owner_of(request)
        payloads = body if isinstance(body, list) else [body]
        runtimes = []

        for payload in payloads:
            config = _validate_runtime_configuration(payload)
            if config is None:
                raise web.HTTPBadRequest()

            try:
                runtime = self._provision_runtime(owner, config)
            except _QuotaError as err:
                raise web.HTTPTooManyRequests(
                    text=json.dumps({"error": str(err)}),
                    content_type="application/json",
                )
            runtimes.append(self._serialize_runtime(runtime))

        return web.json_response(
            {
                "runtimes": runtimes,
                "registry": self.runtime_app.get_registry(),
                "server": RUNTIME_SERVER_KIND,
            }
        )

    async def _delete_runtimes(self, request: web.Request) -> web.Response:
        owner = self._owner_of(request)
        self.runtime_app.remove_all_runtimes(owner)
        self._mounts.release_owner(owner)
        self._purge_owner_session_tokens(owner)
        return web.Response(status=200)

    # ── /runtimes/{id} handlers ────────────────────────────────────────────────

    async def _get_runtime(self, request: web.Request) -> web.Response:
        runtime = self._get_runtime_or_404(request)
        return web.json_response(self._serialize_runtime(runtime))

    async def _delete_runtime(self, request: web.Request) -> web.Response:
        runtime_id = request.match_info["runtime_id"]
        owner = self._owner_of(request)
        # Idempotent: a runtime already gone is the desired end state, not an
        # error. A client removing a runtime closes its notification socket
        # first, and that close reaps a garbage-collected runtime on its own —
        # so by the time this explicit DELETE arrives the runtime is frequently
        # already removed. Reporting NOT_FOUND there surfaces a spurious
        # "Failed to remove runtime" to the user.
        #
        # Scoped to the caller, so this can only ever remove their own runtime;
        # an id owned by another tenant is indistinguishable from a missing one.
        self.runtime_app.remove_runtime(owner, runtime_id)
        self._mounts.release_runtime(owner, runtime_id)
        self._purge_session_tokens(owner, runtime_id)
        return web.json_response({"id": runtime_id})

    def _reap_if_abandoned(self, owner: str, runtime_id: str) -> None:
        """Tear down a runtime whose creator asked for cleanup, now that its
        last client has disconnected.

        Whoever created it said whether it should outlive its clients. A browser
        running a board is that board's controller and asks for cleanup: closing
        the tab should not leave runtimes behind. A coordinator, a config file
        or a script says nothing, and their runtimes stay until deleted — a
        headless runtime is not an abandoned one, and a runtime is never reaped
        because of who happened to connect to it.
        """
        runtime = self.runtime_app.get_runtime(owner, runtime_id)
        if not runtime or not runtime.garbage_collected:
            return
        self.runtime_app.remove_runtime(owner, runtime_id)
        self._mounts.release_runtime(owner, runtime_id)
        self._purge_session_tokens(owner, runtime_id)

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
        sub = self._owner_of(request)
        token = secrets.token_hex(32)
        self._session_tokens[token] = SessionToken(sub=sub, runtime_id=runtime.id)
        return web.json_response({"token": token})

    async def _post_secrets(self, request: web.Request) -> web.Response:
        """Values for the references this runtime's services hold.

        Provisioning carries them already; this is for the moments it cannot
        cover — a board being built a service at a time, an entry edited while a
        board is running, and a re-push after a restart where the services
        survived but the vault did not. It merges, so a client sending one entry
        does not strip the rest.

        POST rather than PUT: it merges rather than replaces, and every other
        mutation this server takes is a POST — the CORS allowlist says so, and a
        lone PUT is a method each runtime implementation would have to remember
        to allow separately.

        There is deliberately no GET. The values go one way: in, and then only
        to a service resolving a reference for a call it is making. What is held
        can be *named* — the response says which aliases the runtime now has —
        because a client needs to show whether a credential is configured.
        """
        runtime = self._get_runtime_or_404(request)
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        if not isinstance(body, dict):
            raise web.HTTPBadRequest()
        runtime.set_secrets(read_secrets_payload(body))
        return web.json_response({"aliases": runtime.secrets().aliases()})

    async def _post_assets(self, request: web.Request) -> web.Response:
        """Descriptors for the assets this runtime's services reference.

        Provisioning carries them already; this is for a configuration naming
        one the runtime was not given, for an asset edited while the board runs
        — which is how an edit reaches a service without reconfiguring it — and
        for a re-push after a restart. It merges, and ``null`` removes an asset.
        Answers with the ids held, never content.
        """
        runtime = self._get_runtime_or_404(request)
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        if not isinstance(body, (dict, list)):
            raise web.HTTPBadRequest()
        runtime.set_assets(read_assets_payload(body))
        return web.json_response({"ids": runtime.assets().ids()})

    async def _check_asset(self, request: web.Request) -> web.Response:
        """Whether an asset resolves here, and to what — a check, not a download."""
        runtime = self._get_runtime_or_404(request)
        resolution = await asyncio.to_thread(
            runtime.assets().resolve, f"hkp-asset://{request.match_info['asset_id']}"
        )
        if resolution.asset is None:
            return web.json_response({"ok": False, "problem": resolution.problem})
        return web.json_response(
            {
                "ok": True,
                "mediaType": resolution.asset.media_type,
                "size": len(resolution.asset.content),
            }
        )

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

    async def _patch_runtime_state(self, request: web.Request) -> web.Response:
        """Change what a running runtime records, without rebuilding it.

        Logging is a decision a board revisits — switched on to look into
        something, off again afterwards — and re-provisioning to carry it would
        restart every service in the runtime to change one boolean. That is what
        the coordinator's per-board log switch calls, and a runtime that does
        not answer it is reported as one the switch did not reach.

        Separate from POST /runtimes/{id}, which processes data rather than
        configuring anything.
        """
        runtime = self._get_runtime_or_404(request)
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest()
        if not isinstance(body, dict):
            raise web.HTTPBadRequest()
        return web.json_response(self._apply_runtime_state(runtime, body))

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
            # `null` is a payload: it says run with nothing on the input, which
            # is how a caller writes "no input" in a format that has no
            # undefined. The websocket path already accepts it, and a service
            # that answers an empty input with its own configuration needs it.
            if body is not None and not isinstance(body, dict):
                raise web.HTTPBadRequest()

        # An external HTTP caller is not continuing a run, it is starting one —
        # as whoever its token says it is.
        result = await self._process_off_loop(
            runtime,
            body,
            context_for_client(None, request.get(_AUTHENTICATED_USER_KEY)),
        )
        if _is_binary_result(result):
            return web.Response(
                body=serialize_message(result, purpose=MessagePurpose.RESULT),
                content_type="application/octet-stream",
            )
        return web.json_response(_jsonable_result(result))

    async def _process_service(self, request: web.Request) -> web.Response:
        """Run the pipeline starting at one service, with a given payload.

        Distinct from configuring it: configure says what a service *is*, this
        says do your job with this. A facade button had only the former, so
        anything it needed to cause had to be smuggled in as a config field that
        a service read as a command.
        """
        runtime = self._get_runtime_or_404(request)
        instance_id = request.match_info["instance_id"]
        raw = await request.read()

        if request.content_type == "application/octet-stream" or is_yas_message(raw):
            try:
                body = deserialize_message(raw).data
            except YasError:
                raise web.HTTPBadRequest()
        else:
            try:
                body = json.loads(raw) if raw else {}
            except Exception:
                raise web.HTTPBadRequest()

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._process_executor,
                runtime.process_at,
                instance_id,
                body,
                lambda _n: None,
                # The caller is never the body's to name: it is whoever the
                # token was verified as.
                context_for_client(
                    body.get("__context") if isinstance(body, dict) else None,
                    request.get(_AUTHENTICATED_USER_KEY),
                ),
            )
        except KeyError:
            raise web.HTTPNotFound()

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
        if self._at_quota(
            len(runtime.list_services()), self._max_services_per_runtime
        ):
            raise web.HTTPTooManyRequests(
                text=json.dumps(
                    {
                        "error": f"Service limit reached ({self._max_services_per_runtime})"
                    }
                ),
                content_type="application/json",
            )
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
        # The runtime is resolved in the authenticated user's namespace, so a
        # token cannot open a socket onto another tenant's runtime; an id owned
        # by someone else is indistinguishable from one that does not exist.
        owner = self._owner_of(request)
        if not self.runtime_app.get_runtime(owner, runtime_id):
            raise web.HTTPNotFound()

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        socket_key = _tenant_key(owner, runtime_id)
        sockets = self._runtime_sockets.setdefault(socket_key, set())
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
                        runtime = self.runtime_app.get_runtime(owner, runtime_id)
                        if runtime:
                            # A peer driving this runtime names the run its
                            # call belongs to, so that a board spanning several
                            # runtimes reads as one trace rather than one each.
                            # Who is calling is not the frame's to say: it is
                            # whoever opened this socket.
                            result = await self._process_off_loop(
                                runtime,
                                data["params"],
                                context_for_client(
                                    data.get("context"),
                                    request.get(_AUTHENTICATED_USER_KEY),
                                ),
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
                    runtime = self.runtime_app.get_runtime(owner, runtime_id)
                    if runtime:
                        # Bytes have nowhere to name a run, so this begins one;
                        # it is still whoever opened this socket that began it.
                        result = await self._process_off_loop(
                            runtime,
                            message.data,
                            context_for_client(
                                None, request.get(_AUTHENTICATED_USER_KEY)
                            ),
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
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            sockets.discard(ws)
            if not sockets:
                self._runtime_sockets.pop(socket_key, None)
                self._reap_if_abandoned(owner, runtime_id)

        return ws

    # ── Utility ────────────────────────────────────────────────────────────────

    def _get_runtime_or_404(self, request: web.Request) -> HostedRuntime:
        """Resolve a runtime inside the caller's namespace.

        A runtime owned by another tenant is reported as 404 rather than 403, so
        runtime ids belonging to other users cannot be probed for existence.
        """
        runtime_id = request.match_info["runtime_id"]
        runtime = self._tenant_of(request).get_runtime(runtime_id)
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


def _result_message(result: Any, context: ProcessContext | None) -> dict[str, Any]:
    """What a runtime produced, as its coordinator is told: the value, and the
    run it was produced in — which the coordinator hands to the next runtime."""
    message: dict[str, Any] = {"type": "result", "data": _jsonable_result(result)}
    wire = context_to_wire(context)
    if wire is not None:
        message["context"] = wire
    return message


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
        # Absent means persist; see RuntimeConfiguration.garbage_collected.
        garbage_collected=value.get("garbageCollected") is True,
        # Both absent mean off; see RuntimeConfiguration.logging / log_data.
        logging=isinstance(value.get("state"), dict)
        and value["state"].get("logging") is True,
        log_level=(
            value["state"]["logLevel"]
            if isinstance(value.get("state"), dict)
            and value["state"].get("logLevel")
            in ("debug", "info", "warn", "error")
            else "info"
        ),
        # Absent means allowed; see RuntimeConfiguration.log_data.
        log_data=not (
            isinstance(value.get("state"), dict)
            and value["state"].get("logData") is False
        ),
        # Values for the references the services carry. Read out of the payload
        # here and handed to the runtime's vault; they are never put back into
        # any service's state, and never appear in a serialized runtime.
        secrets=read_secrets_payload(value.get("secrets")),
        # Descriptors for the assets the services reference; a removal means
        # nothing to a runtime being created, so only descriptors are kept.
        assets={
            asset_id: entry
            for asset_id, entry in read_assets_payload(value.get("assets")).items()
            if entry is not None
        },
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
