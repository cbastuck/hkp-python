"""Iteration as a service.

Ported from hkp-node/tests/iterator.test.ts: the same cases, because the point
of the service is that one board reads the same whichever runtime hosts it.
What is worth pinning is the shape of the loop — how many times the nested
pipeline runs, what comes back, and what happens when one item goes wrong.
"""
from __future__ import annotations

from typing import Any

from hkp.services.iterator import IteratorService
from hkp.types import ProcessContext, ServiceConfiguration

#: What the nested pipeline saw, in the order it saw it.
SEEN: list[Any] = []


class _Fake:
    """Stands in for whatever a board nests; its state says how to behave."""

    service_id = "fake"
    service_name = "Fake"

    def __init__(self, config: ServiceConfiguration) -> None:
        self.uuid = config.uuid
        self._state = dict(config.state or {})

    def configure(self, config: dict) -> dict:
        self._state.update(config)
        return self.get_state()

    def get_state(self) -> dict:
        return dict(self._state)

    def process(self, input: Any, _notify: Any) -> Any:
        SEEN.append(input)
        if self._state.get("failOn") == input:
            raise RuntimeError("malformed")
        if "stopOn" in self._state and self._state["stopOn"] == input:
            return None
        return input * 2 if isinstance(input, int) else {"saw": input}


class _Host:
    def __init__(self, context: ProcessContext | None = None) -> None:
        self.context = context
        self.logged: list[tuple[str, str, Any]] = []

    def current_context(self) -> ProcessContext | None:
        return self.context

    def log(self, level: str, event: str, data: Any = None) -> None:
        self.logged.append((level, event, data))

    def log_settings(self) -> dict:
        return {"logging": False, "log_data": False, "log_level": "info"}

    def notify(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def forward_log(self, _entry: Any) -> None:
        return None


def _iterator(state: dict | None = None, nested: dict | None = None, host: _Host | None = None):
    SEEN.clear()
    service = IteratorService(
        ServiceConfiguration(
            service_id="iterator",
            uuid="iter-1",
            state={
                "pipeline": [{"serviceId": "fake", "instanceId": "inner-1", "state": nested or {}}],
                **(state or {}),
            },
        ),
        lambda cfg, _cs=None: _Fake(cfg),
    )
    host = host or _Host()
    service.set_host(host)
    notifications: list[Any] = []

    def run(input: Any) -> Any:
        return service.process(input, lambda payload, _id=None: notifications.append(payload))

    return service, run, notifications, host


def test_runs_the_nested_pipeline_once_per_element():
    _, run, _, _ = _iterator()

    assert run([1, 2, 3]) == [2, 4, 6]
    assert SEEN == [1, 2, 3]


def test_treats_a_single_item_as_a_list_of_one():
    _, run, _, _ = _iterator()

    assert run(7) == [14]
    assert SEEN == [7]


def test_treats_an_object_as_a_single_item_not_as_its_values():
    _, run, _, _ = _iterator()

    assert run({"a": 1}) == [{"saw": {"a": 1}}]
    assert len(SEEN) == 1


def test_reaches_into_a_result_that_says_more_than_the_list():
    _, run, _, _ = _iterator({"itemsFrom": "conversations"})

    run({"conversations": ["a@x", "b@x"], "count": 2})

    assert SEEN == ["a@x", "b@x"]


def test_stops_at_the_limit():
    _, run, _, _ = _iterator({"limit": 2})

    run([1, 2, 3, 4, 5])

    assert SEEN == [1, 2]


def test_does_nothing_at_all_with_nothing_to_do():
    _, run, _, _ = _iterator()

    assert run([]) is None
    assert run(None) is None
    assert SEEN == []


def test_drops_the_items_whose_pipeline_stopped():
    _, run, _, _ = _iterator({}, {"stopOn": 2})

    assert run([1, 2, 3]) == [2, 6]


def test_passes_nothing_on_when_every_item_stopped_and_still_says_so():
    _, run, notifications, _ = _iterator({}, {"stopOn": 1})

    assert run([1]) is None
    assert notifications == [{"items": 1, "results": 0, "failed": 0}]


def test_says_how_many_it_ran_kept_and_lost():
    service, run, notifications, _ = _iterator()

    run([1, 2])

    assert notifications == [{"items": 2, "results": 2, "failed": 0}]
    assert service.get_state()["lastItems"] == 2
    assert service.get_state()["lastFailed"] == 0


def test_one_item_failing_does_not_end_the_loop():
    # Nine conversations should not go unprocessed because the tenth had a
    # malformed address.
    _, run, notifications, host = _iterator({}, {"failOn": 2})

    assert run([1, 2, 3]) == [2, 6]
    assert SEEN == [1, 2, 3]
    assert notifications == [{"items": 3, "results": 2, "failed": 1}]
    assert host.logged[0][1] == "service.failed"
    assert "malformed" in host.logged[0][2]["message"]


def test_an_iterator_with_nothing_in_it_passes_its_input_through():
    service = IteratorService(
        ServiceConfiguration(service_id="iterator", uuid="iter-1", state={"pipeline": []}),
        lambda cfg, _cs=None: _Fake(cfg),
    )

    assert service.process([1, 2], lambda *_: None) == [1, 2]


def test_passes_through_when_bypassed():
    _, run, _, _ = _iterator({"bypass": True})

    assert run([1, 2]) == [1, 2]
    assert SEEN == []


def test_configuration_survives_being_set_in_the_constructor():
    service, _, _, _ = _iterator({"itemsFrom": "rows", "limit": 5})

    state = service.get_state()
    assert state["itemsFrom"] == "rows"
    assert state["limit"] == 5
    # And the pipeline the base owns is still there beside it.
    assert [entry["instanceId"] for entry in state["pipeline"]] == ["inner-1"]
