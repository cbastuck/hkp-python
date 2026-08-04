from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

# Generic JSON object type
JsonRecord = dict[str, Any]

# Callback signatures
NotifyCallback = Callable[[Any, str | None], None]
NotificationCallback = Callable[["RuntimeNotification"], None]
ServiceCreator = Callable[["ServiceConfiguration"], "HostedService"]


@dataclass
class ServiceRegistryEntry:
    service_id: str
    service_name: str
    version: str | None = None
    capabilities: list[str] | None = None


@dataclass
class ServiceConfiguration:
    service_id: str
    uuid: str
    name: str | None = None
    service_name: str | None = None
    state: JsonRecord | None = None


@dataclass
class RuntimeConfiguration:
    id: str
    name: str
    board_name: str = ""
    #: Whether this runtime should be torn down once the last client that was
    #: connected to it disconnects.
    #:
    #: Declared by whoever creates it, because only they know: a browser
    #: running a board says ``True`` — it is the controller, and its runtimes
    #: should not outlive it — while a coordinator, a config file or a script
    #: says nothing and gets a runtime that lives until it is deleted.
    #:
    #: Absent means persist. Cleanup is opted into, so nothing that exists
    #: today starts disappearing, and a runtime is never reaped because of who
    #: happened to connect to it.
    garbage_collected: bool = False
    services: list[ServiceConfiguration] = field(default_factory=list)


@dataclass
class ServiceDescriptor:
    service_id: str
    service_name: str
    uuid: str
    state: JsonRecord
    version: str | None = None
    capabilities: list[str] | None = None


@dataclass
class RuntimeDescriptor:
    id: str
    name: str
    board_name: str
    services: list[ServiceDescriptor]
    inputs: list[dict[str, Any]] = field(default_factory=list)
    output_url: str | None = None


@dataclass
class RuntimeNotification:
    instance_id: str
    payload: Any


class RuntimeHost(Protocol):
    def process_from(
        self,
        start_after_uuid: str,
        data: Any,
        on_notification: NotificationCallback,
    ) -> Any: ...

    def notify(self, payload: Any, instance_id: str) -> None: ...

    def emit_result(self, output: Any) -> None: ...


@runtime_checkable
class HostedService(Protocol):
    service_id: str
    service_name: str
    uuid: str
    version: str | None
    capabilities: list[str] | None

    def configure(self, config: JsonRecord) -> JsonRecord: ...
    def get_state(self) -> JsonRecord: ...
    def process(self, input: Any, notify: NotifyCallback) -> Any: ...
    def set_host(self, host: RuntimeHost) -> None: ...
    def destroy(self) -> None: ...
