from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from .secrets import SecretEntry, SecretVault

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
    #: Whether this runtime records anything at all.
    #:
    #: Off unless the board turns it on. A board that is not being looked into
    #: has no reason to be writing a line per call to somebody's disk, and a log
    #: kept by default is one nobody decided to keep — including for the data it
    #: holds.
    logging: bool = False
    #: The least severe level this runtime records.
    #:
    #: The flow itself — every service call and return — is recorded at
    #: ``debug``, so this is what decides whether a board keeps a trace of what
    #: ran or only what its services chose to say.
    log_level: str = "info"

    #: Whether an entry may carry the ``data`` a service passed with it.
    #:
    #: Only service-authored entries have one: the flow this runtime records for
    #: itself never carries the values passing through, so nothing reaches a log
    #: unless a service was configured to put it there. That per-service opt-in
    #: is the real gate, which is why this defaults to allowed. Setting it False
    #: is a board-wide override for a deployment that must never write payloads.
    log_data: bool = True
    #: Values for the ``{{secret.<alias>}}`` references this runtime's services
    #: carry, by alias.
    #:
    #: They ride with the create payload because provisioning is one call: the
    #: services in it are constructed *and* configured before it returns, and a
    #: service that opens a connection while being configured needs its
    #: credential by then. Sending them later would be too late for exactly the
    #: services that have one.
    #:
    #: They are unpacked into the runtime's vault and go no further — never into
    #: a service's state, never into a serialized runtime, never back out.
    secrets: dict[str, SecretEntry] = field(default_factory=dict)
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


@dataclass
class ProcessContext:
    """What travels with a process call rather than with the data it carries.

    The ordered service list says what runs; this says which invocation it is
    running as. The distinction matters as soon as anything has to attribute
    work after the fact — which run produced this, and what invoked that run —
    because the payload cannot answer it: the same data can flow through the
    same services for entirely unrelated reasons.

    Kept deliberately separate from ``request_id``, which the browser and hkp-rt
    runtimes carry for a different purpose: that is a *reply address*, exists
    only while someone awaits a response, and is consumed on resolution. A run
    outlives any number of those, so the two are not interchangeable — see
    TODO-CONSOLIDATION.md section 4.
    """

    #: Identifies one invocation of a board — one webhook, one timer tick, one
    #: user action — across every service and runtime it reaches.
    run_id: str
    #: The run this one was invoked from, for a nested pipeline. Absent on a run
    #: triggered from outside rather than from inside another run, which is what
    #: makes a trace reconstructable as a tree rather than a list.
    parent_run_id: str | None = None
    #: Where to send a result somebody is waiting for. Absent for the fire-and-
    #: forget calls that make up most traffic. Unused by this runtime today;
    #: named here so the shape matches the runtimes that do carry one.
    request_id: str | None = None


LogLevel = Literal["debug", "info", "warn", "error"]


@dataclass
class LogEntry:
    """One thing worth recording about a run.

    A board's log is assembled by its coordinator from every runtime it spans,
    because only the coordinator can see the whole board — a log held per
    runtime would have to be stitched back together by timestamp to answer the
    first question anyone asks it, which is what one run did.

    ``data`` is the only free-form field and therefore the only one that can
    carry something a service did not mean to record. It is dropped unless a
    board asks for it, so a service that forgets to redact can only leak
    through a channel somebody deliberately opened.
    """

    run_id: str
    #: ISO 8601, set by the runtime that produced the entry.
    ts: str
    runtime_id: str
    service_uuid: str
    level: LogLevel
    #: What happened, as a short stable name a reader can group by.
    event: str
    parent_run_id: str | None = None
    data: Any = None
    duration_ms: float | None = None


class RuntimeHost(Protocol):
    def process_from(
        self,
        start_after_uuid: str,
        data: Any,
        on_notification: NotificationCallback,
        context: "ProcessContext | None" = None,
    ) -> Any: ...

    def notify(self, payload: Any, instance_id: str) -> None: ...

    def emit_result(self, output: Any) -> None: ...

    def spawn(self, coro: Any) -> bool:
        """Run a coroutine on the server's loop, from wherever this is called.

        A service that starts work during ``process`` is running on a worker
        thread, where there is no loop to schedule on. False means there was
        none to schedule on at all, so the caller can say so in its own words.
        """
        ...

    def log(self, level: LogLevel, event: str, data: Any = None) -> None:
        """Record something about the run in progress.

        Unlike ``notify``, which exists for whoever is watching and may be
        dropped when nobody is, an entry has to survive with nobody attached —
        a board running unwatched is exactly the case a log is for. The run and
        the service are taken from the call in progress, so a service says only
        what happened.
        """
        ...

    def log_settings(self) -> dict[str, bool]:
        """What the runtime around a nested pipeline records.

        A nested runtime is built from a service's own configuration, which says
        nothing about logging — so without asking, a sub-pipeline would sit
        silent inside a board that is being looked into.
        """
        ...

    def forward_log(self, entry: "LogEntry") -> None:
        """Pass an entry a nested pipeline produced outward, unchanged.

        Distinct from ``log`` because the entry already names its own run and
        service: re-deriving those from the call in progress would relabel work
        done inside a sub-pipeline as the work of the service hosting it, which
        is the nesting the entry exists to record.
        """
        ...

    def current_context(self) -> "ProcessContext | None":
        """The context of the call being processed now, or None outside one.

        A service that finishes its work after ``process`` returns — an HTTP
        response arriving, a socket pushing — has left the call it belongs to by
        the time it has something to pass on. Capturing this while still inside
        ``process`` and handing it back to ``process_from`` is what keeps the
        two halves recognisable as one run.
        """
        ...


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
