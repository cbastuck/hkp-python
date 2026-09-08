"""The board log: what a runtime records about a run, and who it reaches."""

from __future__ import annotations

from typing import Any

from hkp.runtime import HostedRuntime
from hkp.services.monitor import MONITOR_DESCRIPTOR, MonitorService
from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
from hkp.types import (
    JsonRecord,
    LogEntry,
    RuntimeConfiguration,
    RuntimeHost,
    ServiceConfiguration,
)


class Talker:
    """Records one entry each time it is called, through the host."""

    service_id = "talker"
    service_name = "Talker"
    version = None
    capabilities = None

    def __init__(self, config: ServiceConfiguration) -> None:
        self.uuid = config.uuid
        self._host: RuntimeHost | None = None

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def configure(self, _config: JsonRecord) -> JsonRecord:
        return {}

    def get_state(self) -> JsonRecord:
        return {}

    def process(self, input: Any, _notify: Any = None) -> Any:
        if self._host:
            self._host.log("info", "handled", {"secret": "shhh"})
        return input

    def destroy(self) -> None:
        pass


def create_service(config: ServiceConfiguration) -> Any:
    if config.service_id == MONITOR_DESCRIPTOR.service_id:
        return MonitorService(config)
    if config.service_id == SUB_SERVICE_DESCRIPTOR.service_id:
        return SubService(config, create_service)
    return Talker(config)


def runtime_with(services: list[ServiceConfiguration]):
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1", name="python", logging=True, services=services
        ),
        create_service,
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)
    return runtime, entries


def test_records_nothing_at_all_until_the_board_turns_logging_on() -> None:
    # Off by default: a board nobody is looking into has no reason to be writing
    # a line per call to somebody's disk.
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            services=[ServiceConfiguration(service_id="talker", uuid="a")],
        ),
        create_service,
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)

    runtime.process({}, lambda _n: None)
    assert entries == []

    runtime.set_logging(True)
    runtime.process({}, lambda _n: None)
    assert len(entries) == 1


class Passthrough:
    """Passes its input on, and records nothing itself."""

    service_id = "passthrough"
    service_name = "Passthrough"
    version = None
    capabilities = None

    def __init__(self, config: ServiceConfiguration) -> None:
        self.uuid = config.uuid

    def set_host(self, host: RuntimeHost) -> None:
        pass

    def configure(self, _config: JsonRecord) -> JsonRecord:
        return {}

    def get_state(self) -> JsonRecord:
        return {}

    def process(self, input: Any, _notify: Any = None) -> Any:
        return input

    def destroy(self) -> None:
        pass


class Stopper(Passthrough):
    """Stops the chain, the way a Filter with a failed predicate does."""

    def process(self, input: Any, _notify: Any = None) -> Any:
        return None


def test_records_the_flow_itself() -> None:
    # What ran, in what order, with what. No service has to cooperate.
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            logging=True,
            log_level="debug",
            services=[
                ServiceConfiguration(service_id="passthrough", uuid="a"),
                ServiceConfiguration(service_id="passthrough", uuid="b"),
            ],
        ),
        lambda config: Passthrough(config),
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)

    runtime.process({"hello": True}, lambda _n: None)

    assert [f"{e.service_uuid}:{e.event}" for e in entries] == [
        "a:service.process",
        "a:service.processed",
        "b:service.process",
        "b:service.processed",
    ]
    assert isinstance(entries[1].duration_ms, float)


def test_says_where_a_run_stopped() -> None:
    # The service that stopped the chain is the one that logged nothing, so only
    # the runtime can answer.
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            logging=True,
            services=[
                ServiceConfiguration(service_id="passthrough", uuid="a"),
                ServiceConfiguration(service_id="stopper", uuid="b"),
                ServiceConfiguration(service_id="passthrough", uuid="c"),
            ],
        ),
        lambda config: (
            Stopper(config) if config.service_id == "stopper" else Passthrough(config)
        ),
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)

    runtime.process({}, lambda _n: None)

    stopped = next(e for e in entries if e.event == "pipeline.stopped")
    assert stopped.service_uuid == "b"
    assert stopped.level == "info"


def test_keeps_the_flow_out_unless_the_board_asks_for_debug() -> None:
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            logging=True,
            services=[ServiceConfiguration(service_id="passthrough", uuid="a")],
        ),
        lambda config: Passthrough(config),
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)

    runtime.process({}, lambda _n: None)
    assert entries == []

    runtime.set_log_level("debug")
    runtime.process({}, lambda _n: None)
    assert [e.event for e in entries] == ["service.process", "service.processed"]


def test_names_the_run_and_the_service_that_produced_an_entry() -> None:
    runtime, entries = runtime_with(
        [
            ServiceConfiguration(service_id="talker", uuid="svc-a"),
            ServiceConfiguration(service_id="talker", uuid="svc-b"),
        ]
    )

    runtime.process({}, lambda _n: None)

    assert len(entries) == 2
    assert entries[0].service_uuid == "svc-a"
    assert entries[1].service_uuid == "svc-b"
    assert entries[0].runtime_id == "py-1"
    assert entries[0].event == "handled"
    assert entries[0].level == "info"
    # One pass is one run, so both entries answer to the same id.
    assert entries[1].run_id == entries[0].run_id
    assert entries[0].ts


def test_keeps_what_a_service_chose_to_record_with_its_entry() -> None:
    # A service that passes data has decided to record it; that per-service
    # choice is the gate.
    runtime, entries = runtime_with(
        [ServiceConfiguration(service_id="talker", uuid="svc-a")]
    )

    runtime.process({}, lambda _n: None)
    assert entries[0].data == {"secret": "shhh"}


def test_lets_a_board_refuse_payloads_whatever_its_services_do() -> None:
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            logging=True,
            log_data=False,
            services=[ServiceConfiguration(service_id="talker", uuid="svc-a")],
        ),
        create_service,
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)

    runtime.process({}, lambda _n: None)

    assert entries[0].event == "handled"
    assert entries[0].data is None


def test_never_puts_the_values_passing_through_into_the_recorded_flow() -> None:
    # Turning the level up asks for more of the shape of a run, not for the data
    # moving through it.
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            logging=True,
            log_level="debug",
            log_data=True,
            services=[ServiceConfiguration(service_id="passthrough", uuid="a")],
        ),
        lambda config: Passthrough(config),
    )
    entries: list[LogEntry] = []
    runtime.register_log_target(entries.append)

    runtime.process({"secret": "shhh"}, lambda _n: None)

    assert [e.event for e in entries] == ["service.process", "service.processed"]
    assert all(e.data is None for e in entries)


def test_keeps_a_nested_pipelines_entries_and_says_which_run_they_belong_to() -> None:
    runtime, entries = runtime_with(
        [
            ServiceConfiguration(service_id="talker", uuid="outer"),
            ServiceConfiguration(
                service_id="sub-service",
                uuid="nest",
                state={"pipeline": [{"serviceId": "talker", "instanceId": "inner"}]},
            ),
        ]
    )

    runtime.process({}, lambda _n: None)

    outer = next(e for e in entries if e.service_uuid == "outer")
    inner = next(e for e in entries if e.service_uuid == "inner")

    # The nested pipeline is its own run, descended from the one that called it,
    # so a reader can rebuild the nesting rather than seeing a flat list.
    assert inner.run_id != outer.run_id
    assert inner.parent_run_id == outer.run_id


def test_records_nothing_when_nobody_is_collecting() -> None:
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="py-1",
            name="python",
            logging=True,
            services=[ServiceConfiguration(service_id="talker", uuid="a")],
        ),
        create_service,
    )

    # No target registered: the call still runs, it simply produces no entries.
    runtime.process({}, lambda _n: None)


def test_lets_a_monitor_feed_the_log_without_a_second_service() -> None:
    runtime, entries = runtime_with(
        [
            ServiceConfiguration(
                service_id="monitor", uuid="probe", state={"logToBoard": True}
            )
        ]
    )

    runtime.process({"value": 1}, lambda _n: None)

    assert len(entries) == 1
    assert entries[0].event == "monitor"
    assert entries[0].service_uuid == "probe"


def test_stays_quiet_when_the_monitor_was_not_asked_to_log() -> None:
    runtime, entries = runtime_with(
        [ServiceConfiguration(service_id="monitor", uuid="probe")]
    )

    runtime.process({"value": 1}, lambda _n: None)

    assert entries == []
