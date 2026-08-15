"""Run attribution: what travels with a process call rather than with the data."""

from __future__ import annotations

from typing import Any

from hkp.runtime import HostedRuntime, child_run, new_run
from hkp.types import (
    JsonRecord,
    ProcessContext,
    RuntimeConfiguration,
    RuntimeHost,
    ServiceConfiguration,
)


class ContextSpy:
    """Records the context it ran under each time it is called.

    The only way to observe attribution from outside: the context travels with
    the call rather than with the data, so nothing about the input reveals it.
    """

    service_id = "context-spy"
    service_name = "ContextSpy"
    version = None
    capabilities = None

    def __init__(self, config: ServiceConfiguration) -> None:
        self.uuid = config.uuid
        self.seen: list[ProcessContext | None] = []
        self._host: RuntimeHost | None = None

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def configure(self, _config: JsonRecord) -> JsonRecord:
        return {}

    def get_state(self) -> JsonRecord:
        return {}

    def process(self, input: Any, _notify: Any = None) -> Any:
        self.seen.append(self._host.current_context() if self._host else None)
        return input

    def destroy(self) -> None:
        pass


class Puller(ContextSpy):
    """Calls the services after it from inside its own call — the pull pattern."""

    service_id = "puller"
    service_name = "Puller"

    def process(self, input: Any, _notify: Any = None) -> Any:
        if self._host:
            self._host.process_from(self.uuid, input, lambda _n: None)
        return None


def build(*services: tuple[str, type[ContextSpy]]):
    spies: dict[str, ContextSpy] = {}
    kinds = dict(services)

    def create(config: ServiceConfiguration) -> Any:
        service = kinds[config.uuid](config)
        spies[config.uuid] = service
        return service

    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="test",
            name="test",
            services=[
                ServiceConfiguration(service_id="spy", uuid=uuid)
                for uuid, _ in services
            ],
        ),
        create,
    )
    return runtime, spies


def test_gives_every_service_in_one_pass_the_same_run() -> None:
    runtime, spies = build(("a", ContextSpy), ("b", ContextSpy))

    runtime.process({}, lambda _n: None)

    a = spies["a"].seen[0]
    b = spies["b"].seen[0]
    assert a is not None and a.run_id
    assert b is not None and b.run_id == a.run_id
    assert a.parent_run_id is None


def test_gives_separate_passes_separate_runs() -> None:
    runtime, spies = build(("a", ContextSpy))

    runtime.process({}, lambda _n: None)
    runtime.process({}, lambda _n: None)

    first, second = spies["a"].seen
    assert first is not None and second is not None
    assert first.run_id != second.run_id


def test_honours_a_context_supplied_by_the_caller() -> None:
    runtime, spies = build(("a", ContextSpy))
    context = new_run()

    runtime.process({}, lambda _n: None, context)

    seen = spies["a"].seen[0]
    assert seen is not None and seen.run_id == context.run_id


def test_keeps_a_pulled_call_inside_the_run_that_pulled_it() -> None:
    # The pull is the inversion-of-control path: a service running the services
    # after it rather than returning to them. It is one run, not two.
    runtime, spies = build(("before", ContextSpy), ("puller", Puller), ("after", ContextSpy))

    runtime.process({}, lambda _n: None)

    before = spies["before"].seen[0]
    after = spies["after"].seen[0]
    assert before is not None and after is not None
    assert after.run_id == before.run_id


def test_restores_the_outer_run_once_a_nested_call_returns() -> None:
    # A pull re-enters the runtime mid-pass. The services after the puller must
    # still see the run the outer pass was running under, not a leftover.
    runtime, spies = build(("before", ContextSpy), ("puller", Puller), ("after", ContextSpy))

    runtime.process({}, lambda _n: None, ProcessContext(run_id="outer"))

    assert spies["before"].seen[0].run_id == "outer"  # type: ignore[union-attr]
    assert spies["after"].seen[0].run_id == "outer"  # type: ignore[union-attr]


def test_starts_a_run_when_process_from_names_none() -> None:
    # A timer tick or an arriving message: nothing was in flight, so there is
    # nothing to continue.
    runtime, spies = build(("a", ContextSpy), ("b", ContextSpy))

    runtime.process_from("a", {}, lambda _n: None)

    b = spies["b"].seen[0]
    assert b is not None and b.run_id
    assert b.parent_run_id is None
    # "a" pushed from itself, so it was never called.
    assert spies["a"].seen == []


def test_continues_the_named_run_when_process_from_is_given_one() -> None:
    # What a service captured before leaving its call and handed back on
    # returning — an HTTP response, a delayed emit.
    runtime, spies = build(("a", ContextSpy), ("b", ContextSpy))

    runtime.process_from("a", {}, lambda _n: None, ProcessContext(run_id="captured"))

    assert spies["b"].seen[0].run_id == "captured"  # type: ignore[union-attr]


def test_reports_no_context_outside_a_call() -> None:
    runtime, _ = build(("a", ContextSpy))

    assert runtime.current_context() is None

    runtime.process({}, lambda _n: None)

    # The pass has returned; nothing is running.
    assert runtime.current_context() is None


def test_child_run_descends_from_its_parent() -> None:
    parent = new_run()
    child = child_run(parent)

    assert child.parent_run_id == parent.run_id
    assert child.run_id != parent.run_id


def test_child_run_starts_its_own_when_there_is_no_parent() -> None:
    child = child_run(None)

    assert child.run_id
    assert child.parent_run_id is None
