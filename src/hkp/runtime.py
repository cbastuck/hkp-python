from __future__ import annotations

from typing import Any, Callable

from .types import (
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


class HostedRuntime:
    def __init__(self, config: RuntimeConfiguration, create_service: ServiceCreator) -> None:
        self.id = config.id
        self.name = config.name
        self.board_name = config.board_name
        self._services: dict[str, HostedService] = {}
        self._service_order: list[str] = []
        self._notification_targets: set[NotificationCallback] = set()
        self._result_targets: set[Callable[[Any], None]] = set()
        self._create_service = create_service

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

    # ── Notification / result targets ──────────────────────────────────────────

    def register_notification_target(self, target: NotificationCallback) -> Callable[[], None]:
        self._notification_targets.add(target)
        return lambda: self._notification_targets.discard(target)

    def register_result_target(self, target: Callable[[Any], None]) -> Callable[[], None]:
        self._result_targets.add(target)
        return lambda: self._result_targets.discard(target)

    # ── Pipeline processing ────────────────────────────────────────────────────

    def process(self, input: Any, on_notification: NotificationCallback) -> Any:
        return self._process_from_index(0, input, on_notification)

    # ── RuntimeHost interface ──────────────────────────────────────────────────

    def process_from(
        self,
        start_after_uuid: str,
        data: Any,
        on_notification: NotificationCallback,
    ) -> Any:
        try:
            start_index = self._service_order.index(start_after_uuid) + 1
        except ValueError:
            start_index = len(self._service_order)
        return self._process_from_index(start_index, data, on_notification)

    def notify(self, payload: Any, instance_id: str) -> None:
        self._emit_notification(
            RuntimeNotification(instance_id=instance_id, payload=payload),
            lambda _: None,
        )

    def emit_result(self, output: Any) -> None:
        for target in list(self._result_targets):
            target(output)

    # ── Internals ──────────────────────────────────────────────────────────────

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

            result = svc.process(result, _make_notify(uuid))

            self._emit_notification(
                RuntimeNotification(
                    instance_id=uuid,
                    payload={"__internal": {"state": "call-process-finished", "data": result}},
                ),
                on_notification,
            )

            if result is None:
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


class RuntimeApp:
    def __init__(self, registry: dict[str, HostedServiceFactory]) -> None:
        self._registry = registry
        self._runtimes: dict[str, HostedRuntime] = {}

    def create_runtime(self, config: RuntimeConfiguration) -> HostedRuntime:
        existing = self._runtimes.get(config.id)
        if existing:
            existing.destroy()
        runtime = HostedRuntime(config, self.create_service)
        self._runtimes[runtime.id] = runtime
        return runtime

    def get_runtime(self, runtime_id: str) -> HostedRuntime | None:
        return self._runtimes.get(runtime_id)

    def get_runtimes(self) -> list[HostedRuntime]:
        return list(self._runtimes.values())

    def remove_runtime(self, runtime_id: str) -> bool:
        runtime = self._runtimes.pop(runtime_id, None)
        if runtime:
            runtime.destroy()
            return True
        return False

    def remove_all_runtimes(self) -> None:
        for runtime in self._runtimes.values():
            runtime.destroy()
        self._runtimes.clear()

    def get_registry(self) -> list[dict[str, Any]]:
        result = []
        for factory in self._registry.values():
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
