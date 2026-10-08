"""Run attribution: what travels with a process call rather than with the data."""

from __future__ import annotations

import threading
import time
from typing import Any

from hkp.runtime import HostedRuntime, child_run, new_run
from hkp.services.sub_service import SubService
from hkp.types import (
    JsonRecord,
    PersonRunActor,
    ProcessContext,
    SourceRunActor,
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
        self.configured: list[ProcessContext | None] = []
        self._host: RuntimeHost | None = None

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def configure(self, _config: JsonRecord) -> JsonRecord:
        self.configured.append(self._host.current_context() if self._host else None)
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

    runtime.process(
        {},
        lambda _n: None,
        ProcessContext(run_id="outer", actor=SourceRunActor(kind="board")),
    )

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

    runtime.process_from(
        "a",
        {},
        lambda _n: None,
        ProcessContext(run_id="captured", actor=SourceRunActor(kind="board")),
    )

    assert spies["b"].seen[0].run_id == "captured"  # type: ignore[union-attr]


def test_reports_no_context_outside_a_call() -> None:
    runtime, _ = build(("a", ContextSpy))

    assert runtime.current_context() is None

    runtime.process({}, lambda _n: None)

    # The pass has returned; nothing is running.
    assert runtime.current_context() is None


def test_gives_configure_the_context_supplied_by_the_framework() -> None:
    runtime, spies = build(("a", ContextSpy))
    run = ProcessContext(
        run_id="configured-by-member",
        actor=SourceRunActor(kind="board"),
    )

    runtime.configure_service("a", {}, run)
    runtime.configure_service("a", {})

    assert spies["a"].configured == [run, None]
    assert runtime.current_context() is None


def person(sub: str = "auth0|member") -> ProcessContext:
    return ProcessContext(
        run_id=f"run-of-{sub}",
        actor=PersonRunActor(
            kind="person", sub=sub, expires_at=int(time.time() * 1000) + 60_000
        ),
    )


class Waiter(ContextSpy):
    """Holds its call open until told to go on, so another can overlap it."""

    def __init__(self, config: ServiceConfiguration) -> None:
        super().__init__(config)
        self.entered = threading.Event()
        self.release = threading.Event()

    def process(self, input: Any, _notify: Any = None) -> Any:
        self.entered.set()
        assert self.release.wait(5)
        return super().process(input, _notify)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if config.get("wait"):
            self.entered.set()
            assert self.release.wait(5)
        return super().configure(config)


def test_keeps_a_call_on_one_thread_out_of_a_call_on_another() -> None:
    # A pass runs on a worker thread while a service is configured on the
    # server's loop. Shared, the context each saved on the way in would be the
    # other's, and whichever returned last would leave a finished run in place
    # for every tick that followed.
    runtime, spies = build(("slow", Waiter), ("after", ContextSpy))
    slow = spies["slow"]
    assert isinstance(slow, Waiter)

    passing = threading.Thread(
        target=lambda: runtime.process({}, lambda _n: None, person("auth0|anna"))
    )
    passing.start()
    assert slow.entered.wait(5)

    # Overlapping it from here: begun while the pass is inside its service,
    # and seeing none of it.
    assert runtime.current_context() is None
    runtime.configure_service("after", {}, person("auth0|owner"))
    assert runtime.current_context() is None

    slow.release.set()
    passing.join(5)

    assert spies["after"].configured[-1].actor.sub == "auth0|owner"
    assert spies["after"].seen[-1].actor.sub == "auth0|anna"
    assert runtime.current_context() is None


def test_leaves_no_run_behind_when_a_configure_outlasts_a_pass() -> None:
    runtime, spies = build(("slow", Waiter), ("after", ContextSpy))
    slow = spies["slow"]
    assert isinstance(slow, Waiter)

    configuring = threading.Thread(
        target=lambda: runtime.configure_service(
            "slow", {"wait": True}, person("auth0|owner")
        )
    )
    configuring.start()
    assert slow.entered.wait(5)

    # A whole pass, begun and finished while the configure is still inside.
    slow.process = lambda input, _notify=None: input  # type: ignore[method-assign]
    runtime.process({}, lambda _n: None, person("auth0|anna"))

    slow.release.set()
    configuring.join(5)

    # Neither is left for what runs next: a tick begins a run of its own.
    assert runtime.current_context() is None
    runtime.process_from("slow", {}, lambda _n: None)
    assert spies["after"].seen[-1].actor.kind == "board"


def test_reports_a_scoped_entry_as_the_run_that_entered_it() -> None:
    # The runtime around the scope is never in a call here, so only the nested
    # runtime knows whose run a report belongs to. Reported without it, a
    # member's answer reads as the board's and is sent to everybody.
    def create(config: ServiceConfiguration) -> Any:
        if config.service_id == "sub-service":
            return SubService(config, create)
        return ContextSpy(config)

    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="test",
            name="test",
            services=[
                ServiceConfiguration(
                    service_id="sub-service",
                    uuid="scope",
                    state={"pipeline": [{"serviceId": "spy", "uuid": "inner"}]},
                )
            ],
        ),
        create,
    )
    actors: list[Any] = []
    runtime.register_notification_target(
        lambda n: actors.append(n.context.actor if n.context else None)
        if n.instance_id == "scope.inner"
        else None
    )

    runtime.process_at("scope.inner", {"n": 1}, lambda _n: None, person())

    assert actors
    assert all(
        actor is not None and actor.kind == "person" and actor.sub == "auth0|member"
        for actor in actors
    )


def test_child_run_descends_from_its_parent() -> None:
    parent = new_run()
    child = child_run(parent)

    assert child.parent_run_id == parent.run_id
    assert child.run_id != parent.run_id


def test_child_run_starts_its_own_when_there_is_no_parent() -> None:
    child = child_run(None)

    assert child.run_id
    assert child.parent_run_id is None
