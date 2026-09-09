from __future__ import annotations

import asyncio

import time
import uuid as _uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Coroutine, Callable, Iterator

from .data import ControlFlowData
from .mounts import MountHandle, MountHandler, RuntimeMounts
from .secrets import SecretEntry, SecretVault
from .types import (
    LogEntry,
    LogLevel,
    ProcessContext,
    HostedService,
    JsonRecord,
    NotificationCallback,
    RuntimeConfiguration,
    RuntimeDescriptor,
    RuntimeNotification,
    ServiceConfiguration,
    ServiceCreator,
    ServiceDescriptor,
    ServiceRegistryEntry,
)


#: Severity order, so a runtime can drop anything below what it records.
LOG_LEVELS = {"debug": 0, "info": 1, "warn": 2, "error": 3}


def iso_timestamp() -> str:
    """UTC, ISO 8601, to milliseconds — what every runtime stamps an entry with.

    Not ``datetime.isoformat()``, which writes microseconds and ``+00:00``. A
    board's log holds entries from every runtime it spans in one file, and it is
    ordered and filtered by comparing this field as text — so a stamp that spells
    the same instant differently to hkp-node's and hkp-rt's sorts against them
    rather than among them.
    """
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%dT%H:%M:%S')}.{now.microsecond // 1000:03d}Z"


def new_run() -> ProcessContext:
    """A run with no parent: something outside the board asked for this."""
    return ProcessContext(run_id=str(_uuid.uuid4()))


def context_from_wire(value: Any) -> ProcessContext | None:
    """Read a context a peer sent, filling in what it left out.

    A caller that names no run is not continuing one, so a run is begun rather
    than left unidentified — work that cannot be attributed to anything is worse
    than work attributed to a run of its own. Returns None only when there was
    no context at all, which lets the caller decide.
    """
    if not isinstance(value, dict):
        return None

    def text(key: str) -> str | None:
        found = value.get(key)
        return found if isinstance(found, str) and found else None

    return ProcessContext(
        run_id=text("runId") or str(_uuid.uuid4()),
        parent_run_id=text("parentRunId"),
        request_id=text("requestId"),
    )


def child_run(parent: ProcessContext | None) -> ProcessContext:
    """A run invoked from inside another one, as a nested pipeline is.

    The child gets an identity of its own rather than borrowing its parent's, so
    that work done inside a sub-pipeline stays distinguishable from work done
    around it — which is the whole difference between a trace that shows nesting
    and one that shows a flat list in timestamp order.
    """
    if parent is None:
        return new_run()
    return ProcessContext(run_id=str(_uuid.uuid4()), parent_run_id=parent.run_id)


#: The loop the server runs on, for work a service starts from a worker thread.
#:
#: A runtime's ``process`` is called on a thread pool, so a service scheduling
#: something during a pass has no running loop to schedule it on. The server
#: records its loop here once, and every runtime — nested ones included, which
#: are built during a pass and would otherwise capture nothing — schedules
#: through it.
_server_loop: asyncio.AbstractEventLoop | None = None


def set_server_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Told once by the server, at startup."""
    global _server_loop
    _server_loop = loop


class HostedRuntime:
    def __init__(
        self,
        config: RuntimeConfiguration,
        create_service: ServiceCreator,
        mounts: RuntimeMounts | None = None,
    ) -> None:
        self._mounts = mounts
        self.id = config.id
        self.name = config.name
        self.board_name = config.board_name
        #: See RuntimeConfiguration.garbage_collected. False means persist.
        self.garbage_collected = config.garbage_collected
        self._services: dict[str, HostedService] = {}
        self._service_order: list[str] = []
        self._notification_targets: set[NotificationCallback] = set()
        self._result_targets: set[Callable[[Any], None]] = set()
        #: The call being processed right now; see _with_context.
        self._context: ProcessContext | None = None
        #: Which service the pass is inside, so a log entry can name it.
        self._current_service: str | None = None
        self._log_targets: set[Callable[[LogEntry], None]] = set()
        #: Whether log entries may carry their ``data`` payload.
        #:
        #: Off unless a board turns it on, because ``data`` is the one free-form
        #: field and therefore the only place a service can record something it
        #: did not mean to. Redaction at the source is a discipline and
        #: disciplines fail; leaving the channel closed by default means a lapse
        #: can only escape somewhere somebody deliberately opened it.
        # Absent means allowed; the per-service choice is the gate.
        self._log_data = config.log_data
        #: Whether anything is recorded at all; see RuntimeConfiguration.logging
        self._logging = config.logging
        self._log_level = config.log_level if config.log_level in LOG_LEVELS else "info"
        self._create_service = create_service
        #: Values for the references this runtime's services carry. Held apart
        #: from every service's state, and reachable only through secrets().
        #: Filled before any service is built, because a service that opens a
        #: connection while being configured asks for its credential then.
        self._vault = SecretVault()
        self._vault.replace(config.secrets)
        #: Where this runtime's secrets come from when they are not its own;
        #: see delegate_secrets.
        self._secrets_from: Callable[[], SecretVault | None] | None = None

        for svc_config in config.services:
            self.add_service(svc_config)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def serialize(self, output_url: str | None = None) -> RuntimeDescriptor:
        return RuntimeDescriptor(
            id=self.id,
            name=self.name,
            board_name=self.board_name,
            services=self.list_services(),
            inputs=[],
            output_url=output_url,
        )

    def list_services(self) -> list[ServiceDescriptor]:
        result: list[ServiceDescriptor] = []
        for uuid in self._service_order:
            svc = self._services.get(uuid)
            if svc:
                result.append(
                    ServiceDescriptor(
                        service_id=svc.service_id,
                        service_name=svc.service_name,
                        version=getattr(svc, "version", None),
                        capabilities=getattr(svc, "capabilities", None),
                        uuid=svc.uuid,
                        state=svc.get_state(),
                    )
                )
        return result

    # ── Service management ─────────────────────────────────────────────────────

    def get_service(self, uuid: str) -> HostedService | None:
        return self._services.get(uuid)

    def add_service(self, config: ServiceConfiguration) -> JsonRecord:
        if config.uuid in self._services:
            raise ValueError(f"Service already exists: {config.uuid}")
        svc = self._create_service(config)
        if hasattr(svc, "set_host"):
            svc.set_host(self)
        self._services[svc.uuid] = svc
        self._service_order.append(svc.uuid)
        return svc.get_state()

    def configure_service(self, uuid: str, config: JsonRecord) -> JsonRecord | None:
        svc = self._services.get(uuid)
        if not svc:
            return None
        return svc.configure(config)

    def remove_service(self, uuid: str) -> bool:
        svc = self._services.get(uuid)
        if svc and hasattr(svc, "destroy"):
            svc.destroy()
        if uuid not in self._services:
            return False
        del self._services[uuid]
        self._service_order = [u for u in self._service_order if u != uuid]
        return True

    def rearrange_services(self, new_order: list[str]) -> bool:
        if len(new_order) != len(self._service_order):
            return False
        known = set(self._service_order)
        for uuid in new_order:
            if uuid not in known:
                return False
        self._service_order = list(new_order)
        return True

    def destroy(self) -> None:
        for svc in self._services.values():
            if hasattr(svc, "destroy"):
                svc.destroy()
        self._services.clear()
        self._service_order = []
        self._notification_targets.clear()
        self._result_targets.clear()
        self._log_targets.clear()

    # ── Notification / result targets ──────────────────────────────────────────

    def register_notification_target(self, target: NotificationCallback) -> Callable[[], None]:
        self._notification_targets.add(target)
        return lambda: self._notification_targets.discard(target)

    def register_result_target(self, target: Callable[[Any], None]) -> Callable[[], None]:
        self._result_targets.add(target)
        return lambda: self._result_targets.discard(target)

    # ── Pipeline processing ────────────────────────────────────────────────────

    def process(
        self,
        input: Any,
        on_notification: NotificationCallback,
        context: ProcessContext | None = None,
    ) -> Any:
        with self._with_context(context or new_run()):
            return self._process_from_index(0, input, on_notification)

    def process_at(
        self,
        start_at_uuid: str,
        data: Any,
        on_notification: NotificationCallback,
        context: ProcessContext | None = None,
    ) -> Any:
        """Run the pipeline starting **at** a service rather than after it.

        `process_from` exists for a service handing work onward — it means
        "carry on behind me", so it advances past the caller. This is the other
        question: something outside the pipeline wants a particular service to
        do its job with a given payload, and that service must actually run.

        hkp-rt spells the same distinction as
        ``processFrom(service, data, advanceBefore)``; kept as a separate entry
        point here so the advancing call, which every service uses, cannot
        change shape by accident.
        """
        try:
            start_index = self._service_order.index(start_at_uuid)
        except ValueError:
            raise KeyError(start_at_uuid)

        # Nothing to continue: whoever asked for this is outside the board, so
        # it begins a run rather than joining one.
        with self._with_context(context or new_run()):
            return self._process_from_index(start_index, data, on_notification)

    # ── RuntimeHost interface ──────────────────────────────────────────────────

    def current_context(self) -> ProcessContext | None:
        return self._context

    def process_from(
        self,
        start_after_uuid: str,
        data: Any,
        on_notification: NotificationCallback,
        context: ProcessContext | None = None,
    ) -> Any:
        # Three ways to arrive here, and each wants a different run:
        #
        # - Named explicitly: a service that left its call and came back — an
        #   HTTP response, an awaited write — handing back what it captured.
        # - Called from inside a call: a service pulling the services after it
        #   rather than returning to them. Still the same run, and the current
        #   context already says which, so nothing has to be threaded by hand.
        # - Neither: a timer tick, an arriving message. Nothing to continue, so
        #   this begins a run.
        #
        # A service that leaves its call and forgets to capture lands in the
        # third case, which splits its trace in two rather than attributing its
        # work to whichever run happened to be in flight. Fragmentation is
        # visible in a trace; misattribution reads as fact.
        run_context = context or self._context or new_run()

        try:
            start_index = self._service_order.index(start_after_uuid) + 1
        except ValueError:
            start_index = len(self._service_order)

        # A service pushing from itself (a Timer tick, an inbound request) was
        # never called by the loop below, so the loop never reported it. Report
        # it here, or the UI shows a service producing nothing while the service
        # after it plainly receives data.
        self._emit_notification(
            RuntimeNotification(
                instance_id=start_after_uuid,
                payload={"__internal": {"state": "call-process", "data": None}},
            ),
            on_notification,
        )
        self._emit_notification(
            RuntimeNotification(
                instance_id=start_after_uuid,
                payload={"__internal": {"state": "call-process-finished", "data": data}},
            ),
            on_notification,
        )

        with self._with_context(run_context):
            return self._process_from_index(start_index, data, on_notification)

    def mount(
        self,
        service_uuid: str,
        handler: MountHandler,
        mount_name: str | None = None,
    ) -> MountHandle | None:
        """Claim a publicly reachable endpoint served by the shared server.

        Returns None when the host cannot serve mounts — an inner sub-service
        pipeline, or a server that is not listening yet — in which case the
        service has no public endpoint and should say so in its state rather
        than falling back to a port of its own.

        The board is named here rather than by the caller: a service knows what
        its endpoint is called, and the runtime knows which board it is in, and
        the address is derived from both.
        """
        if not self._mounts:
            return None
        return self._mounts.mount(
            service_uuid,
            handler,
            board_name=self.board_name,
            mount_name=mount_name,
        )

    def notify(self, payload: Any, instance_id: str) -> None:
        self._emit_notification(
            RuntimeNotification(instance_id=instance_id, payload=payload),
            lambda _: None,
        )

    def log(self, level: LogLevel, event: str, data: Any = None) -> None:
        # Nothing to attribute an entry to means nothing worth recording: a
        # service logging outside a call has no run, and an entry that names no
        # run cannot be found again.
        # Off means off: no entry is built, so nothing is spent deciding what
        # it would have said.
        if (
            not self._logging
            or LOG_LEVELS[level] < LOG_LEVELS[self._log_level]
            or self._context is None
            or not self._log_targets
        ):
            return

        entry = LogEntry(
            run_id=self._context.run_id,
            parent_run_id=self._context.parent_run_id,
            ts=iso_timestamp(),
            runtime_id=self.id,
            service_uuid=self._current_service or "",
            level=level,
            event=event,
            data=data if (self._log_data and data is not None) else None,
        )
        for target in list(self._log_targets):
            target(entry)

    def _log_processed(self, result: Any, duration_ms: float) -> None:
        """service.processed, carrying how long the call took."""
        if (
            not self._logging
            or LOG_LEVELS["debug"] < LOG_LEVELS[self._log_level]
            or self._context is None
            or not self._log_targets
        ):
            return
        entry = LogEntry(
            run_id=self._context.run_id,
            parent_run_id=self._context.parent_run_id,
            ts=iso_timestamp(),
            runtime_id=self.id,
            service_uuid=self._current_service or "",
            level="debug",
            event="service.processed",
            duration_ms=duration_ms,
        )
        for target in list(self._log_targets):
            target(entry)

    def forward_log(self, entry: LogEntry) -> None:
        """Pass an entry a nested pipeline produced outward, unchanged."""
        for target in list(self._log_targets):
            target(entry)

    def register_log_target(
        self, target: Callable[[LogEntry], None]
    ) -> Callable[[], None]:
        """Where this runtime's entries go.

        The server registers one to carry them to the board's coordinator; a
        nested pipeline's host registers one to carry them out to the runtime
        around it.
        """
        self._log_targets.add(target)
        return lambda: self._log_targets.discard(target)

    def set_log_data(self, enabled: bool) -> None:
        self._log_data = enabled

    def set_logging(self, enabled: bool) -> None:
        self._logging = enabled

    def set_log_level(self, level: str) -> None:
        if level in LOG_LEVELS:
            self._log_level = level

    def log_settings(self) -> dict[str, bool]:
        return {
            "logging": self._logging,
            "log_data": self._log_data,
            "log_level": self._log_level,
        }

    def secrets(self) -> SecretVault:
        """The runtime's secrets, for a service that has a credential to send.

        A service holds the reference it was configured with and asks here for
        the value, naming where it is about to send it. What comes back is used
        and dropped: assigning it to state would put it back on the path a board
        is saved from, which is the whole thing this arrangement exists to
        prevent.
        """
        if self._secrets_from is not None:
            delegated = self._secrets_from()
            if delegated is not None:
                return delegated
        return self._vault

    def set_secrets(self, entries: dict[str, SecretEntry]) -> None:
        """Takes in values for references this runtime's services already hold.

        Merges rather than replaces, because this is what a client editing one
        entry sends, and what a client re-pushes after a restart. Replacing on a
        partial push would strip credentials from services nobody touched.
        """
        self._vault.merge(entries)

    def delegate_secrets(
        self, source: Callable[[], SecretVault | None]
    ) -> None:
        """Take secrets from somewhere else rather than from this runtime's own.

        A nested pipeline is a runtime nobody provisions: no create payload
        reaches it, so its own vault stays empty and a service inside it could
        never resolve a reference. What it does have is the runtime around it,
        which was provisioned — so it asks that one instead.

        Asked for each time rather than copied, so that a value pushed after a
        board is running reaches a nested service as immediately as a top-level
        one, and so that nesting composes: each level delegates outward until it
        reaches the runtime that was actually given something.
        """
        self._secrets_from = source

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> bool:
        """Run a coroutine on the server's loop, from wherever this is called.

        A service that starts work during ``process`` — a request, a timer — is
        running on a worker thread, where there is no loop to schedule on.
        Answering False rather than raising lets a caller say so in its own
        words; the coroutine is closed either way, so nothing is left
        un-awaited.
        """
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            running.create_task(coro)
            return True
        if _server_loop is not None:
            asyncio.run_coroutine_threadsafe(coro, _server_loop)
            return True

        coro.close()
        return False

    def emit_result(self, output: Any) -> None:
        for target in list(self._result_targets):
            target(output)

    # ── Internals ──────────────────────────────────────────────────────────────

    @contextmanager
    def _with_context(self, context: ProcessContext) -> Iterator[None]:
        """Runs the block with ``context`` current, restoring what was there.

        Restoring rather than clearing is what makes this survive a service that
        calls back into this runtime from inside its own ``process`` — the pull
        that a cache miss or a router performs. That inner call is still part of
        the outer run, and when it returns the outer loop has more services to
        visit, so the context it was running under has to come back.

        Safe as ambient state only because a pass is synchronous: it never
        awaits, so no second call can interleave with this one and observe a
        context that is not its own. A pass that awaited would need the context
        threaded through the call instead.
        """
        previous = self._context
        self._context = context
        try:
            yield
        finally:
            self._context = previous

    def _process_from_index(
        self,
        start_index: int,
        input: Any,
        on_notification: NotificationCallback,
    ) -> Any:
        result = input

        for uuid in self._service_order[start_index:]:
            svc = self._services.get(uuid)
            if not svc:
                continue

            self._emit_notification(
                RuntimeNotification(
                    instance_id=uuid,
                    payload={"__internal": {"state": "call-process", "data": result}},
                ),
                on_notification,
            )

            def _make_notify(u: str) -> NotificationCallback:
                def _notify_cb(payload: Any, inst_id: str | None = None) -> None:
                    self._emit_notification(
                        RuntimeNotification(instance_id=inst_id or u, payload=payload),
                        on_notification,
                    )
                return _notify_cb  # type: ignore[return-value]

            # Restored rather than cleared, for the same reason the context is:
            # a service that pulls the ones after it re-enters this loop, and
            # when it returns the entries that follow still belong to the
            # service that pulled.
            outer_service = self._current_service
            self._current_service = uuid
            started_at = time.monotonic()
            try:
                # The flow itself, at debug: which service the runtime called,
                # and below, what it returned and how long it took.
                #
                # Deliberately without the value flowing through. The level says
                # how much of the shape of a run to keep, and turning it up must
                # not also start recording the data — what flows through is
                # recorded only where a service was configured to record it.
                self.log("debug", "service.process")
                result = svc.process(result, _make_notify(uuid))
                self._log_processed(result, (time.monotonic() - started_at) * 1000)
            finally:
                self._current_service = outer_service

            # Early return: skip the remaining services, the carried result
            # becomes the runtime's output.
            early_return = isinstance(result, ControlFlowData)
            if early_return:
                result = result.result

            self._emit_notification(
                RuntimeNotification(
                    instance_id=uuid,
                    payload={"__internal": {"state": "call-process-finished", "data": result}},
                ),
                on_notification,
            )

            if early_return or result is None:
                if result is None:
                    # Where the run ended, named. Above debug because it is the
                    # outcome of the run rather than a step in it.
                    self._current_service = uuid
                    self.log("info", "pipeline.stopped")
                    self._current_service = None
                break

        return result

    def _emit_notification(
        self,
        notification: RuntimeNotification,
        on_notification: NotificationCallback,
    ) -> None:
        on_notification(notification)
        for target in list(self._notification_targets):
            target(notification)


# ── RuntimeApp ─────────────────────────────────────────────────────────────────


class HostedServiceFactory:
    def __init__(
        self,
        descriptor: ServiceRegistryEntry,
        create_fn: Callable[[ServiceConfiguration, ServiceCreator], HostedService],
    ) -> None:
        self.descriptor = descriptor
        self._create_fn = create_fn

    def create(self, config: ServiceConfiguration, create_service: ServiceCreator) -> HostedService:
        return self._create_fn(config, create_service)


class TenantRuntimes:
    """A single tenant's view of the runtime app.

    Runtime ids are only unique within an owner — boards ship stable,
    human-readable ids (``node``, ``chat-node``), so two users loading the same
    board must each get their own runtime rather than sharing one. Every route
    resolves runtimes through one of these views, so a handler cannot reach
    another tenant's runtime even by id.
    """

    def __init__(self, owner: str, app: "RuntimeApp") -> None:
        self.owner = owner
        self._app = app

    def create_runtime(self, config: RuntimeConfiguration) -> HostedRuntime:
        return self._app.create_runtime(self.owner, config)

    def get_runtime(self, runtime_id: str) -> HostedRuntime | None:
        return self._app.get_runtime(self.owner, runtime_id)

    def get_runtimes(self) -> list[HostedRuntime]:
        return self._app.get_runtimes(self.owner)

    def remove_runtime(self, runtime_id: str) -> bool:
        return self._app.remove_runtime(self.owner, runtime_id)

    def remove_all_runtimes(self) -> None:
        self._app.remove_all_runtimes(self.owner)


class RuntimeApp:
    def __init__(
        self,
        registry: dict[str, HostedServiceFactory],
        # Supplied by the server, which owns the listening socket. Absent in
        # tests and anywhere runtimes need no public endpoints.
        mounts_for: Callable[[str, str], RuntimeMounts] | None = None,
    ) -> None:
        self._registry = registry
        self._mounts_for = mounts_for
        # owner key -> runtime id -> runtime. The owner key is the authenticated
        # ``sub`` (or "anonymous" when auth is off, collapsing to one bucket).
        self._runtimes: dict[str, dict[str, HostedRuntime]] = {}

    def for_owner(self, owner: str) -> TenantRuntimes:
        """A tenant-scoped view; the only way route handlers reach runtimes."""
        return TenantRuntimes(owner, self)

    def create_runtime(self, owner: str, config: RuntimeConfiguration) -> HostedRuntime:
        owned = self._runtimes.setdefault(owner, {})
        existing = owned.get(config.id)
        if existing:
            existing.destroy()
        runtime = HostedRuntime(
            config,
            self.create_service,
            self._mounts_for(owner, config.id) if self._mounts_for else None,
        )
        owned[runtime.id] = runtime
        return runtime

    def get_runtime(self, owner: str, runtime_id: str) -> HostedRuntime | None:
        return self._runtimes.get(owner, {}).get(runtime_id)

    def get_runtimes(self, owner: str) -> list[HostedRuntime]:
        return list(self._runtimes.get(owner, {}).values())

    def remove_runtime(self, owner: str, runtime_id: str) -> bool:
        owned = self._runtimes.get(owner)
        if not owned:
            return False
        runtime = owned.pop(runtime_id, None)
        if not owned:
            self._runtimes.pop(owner, None)
        if runtime:
            runtime.destroy()
            return True
        return False

    def remove_all_runtimes(self, owner: str) -> None:
        owned = self._runtimes.pop(owner, None)
        if not owned:
            return
        for runtime in owned.values():
            runtime.destroy()

    def get_registry(self) -> list[dict[str, Any]]:
        result = []
        # A service may be registered under more than one id (an alias kept so
        # older boards still load). The registry advertises each one once, under
        # the id its descriptor calls canonical.
        seen: set[str] = set()
        for factory in self._registry.values():
            if factory.descriptor.service_id in seen:
                continue
            seen.add(factory.descriptor.service_id)
            entry: dict[str, Any] = {
                "serviceId": factory.descriptor.service_id,
                "serviceName": factory.descriptor.service_name,
            }
            if factory.descriptor.version is not None:
                entry["version"] = factory.descriptor.version
            if factory.descriptor.capabilities is not None:
                entry["capabilities"] = factory.descriptor.capabilities
            result.append(entry)
        return result

    def create_service(self, config: ServiceConfiguration) -> HostedService:
        factory = self._registry.get(config.service_id)
        if not factory:
            raise ValueError(f"Unknown serviceId: {config.service_id}")
        return factory.create(config, self.create_service)
