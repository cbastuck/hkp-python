"""One value in, as many out as it has items.

Ported from hkp-node/tests/spread.test.ts. What is worth pinning is that each
item goes down the rest of the board and leaves the runtime by itself, in the
run that reached the Spread, and what each one carries for a Gather.
"""
from __future__ import annotations

from typing import Any, Callable

from hkp.services.spread import SpreadService
from hkp.types import ServiceConfiguration


class _Host:
    def __init__(self, context: Any = None, downstream: Callable[[Any], Any] = lambda d: d) -> None:
        self.context = context
        self.downstream = downstream
        self.pulled: list[tuple[str, Any, Any]] = []
        self.emitted: list[Any] = []

    def current_context(self) -> Any:
        return self.context

    def process_from(self, after: str, data: Any, _on_notification: Any, context: Any = None) -> Any:
        self.pulled.append((after, data, context))
        return self.downstream(data)

    def emit_result(self, output: Any) -> None:
        self.emitted.append(output)


def _spread(state: dict | None = None, host: _Host | None = None):
    service = SpreadService(ServiceConfiguration(service_id="spread", uuid="spread-1", state=state or {}))
    host = host or _Host()
    service.set_host(host)
    notifications: list[Any] = []

    def run(input: Any) -> Any:
        return service.process(input, lambda payload, _id=None: notifications.append(payload))

    return service, run, notifications, host


def test_sends_each_item_on_by_itself_saying_where_it_sat():
    _, run, _, host = _spread()

    assert run(["a", "b", "c"]) is None

    assert all(after == "spread-1" for after, _, _ in host.pulled)
    assert [(p["item"], p["index"], p["total"]) for p in host.emitted] == [
        ("a", 0, 3),
        ("b", 1, 3),
        ("c", 2, 3),
    ]
    # One group per list, the same for all of its items.
    assert len({p["group"] for p in host.emitted}) == 1


def test_gives_each_list_a_group_of_its_own():
    _, run, _, host = _spread()

    run(["a"])
    run(["b"])

    assert host.emitted[0]["group"] != host.emitted[1]["group"]


def test_treats_a_single_item_as_a_list_of_one():
    _, run, _, host = _spread()

    run({"x": 1})

    assert len(host.emitted) == 1
    assert host.emitted[0]["item"] == {"x": 1}
    assert (host.emitted[0]["index"], host.emitted[0]["total"]) == (0, 1)


def test_reaches_into_a_result_that_says_more_than_the_list():
    _, run, _, host = _spread({"itemsFrom": "rows"})

    run({"rows": [1, 2], "count": 2})

    assert [p["item"] for p in host.emitted] == [1, 2]


def test_sends_nothing_on_for_nothing():
    _, run, notifications, host = _spread()

    assert run(None) is None
    assert run([]) is None
    assert host.pulled == []
    assert notifications == []


def test_an_item_the_rest_of_the_pipeline_stopped_goes_no_further():
    host = _Host(downstream=lambda d: None if d["item"] == "skip" else d)
    _, run, _, _ = _spread({}, host)

    run(["keep", "skip", "keep too"])

    assert len(host.pulled) == 3
    assert [p["item"] for p in host.emitted] == ["keep", "keep too"]


def test_sends_every_item_on_in_the_run_that_reached_it():
    run_context = object()
    host = _Host(context=run_context)
    _, run, _, _ = _spread({}, host)

    run([1, 2])

    assert [context for _, _, context in host.pulled] == [run_context, run_context]


def test_says_how_many_it_sent_on_and_keeps_where_they_come_from():
    service, run, notifications, _ = _spread({"itemsFrom": "rows"})

    run({"rows": [1, 2, 3]})

    assert notifications == [{"lastTotal": 3}]
    assert service.get_state() == {"bypass": False, "itemsFrom": "rows", "lastTotal": 3}


def test_passes_its_input_through_untouched_when_bypassed():
    _, run, _, host = _spread({"bypass": True})

    assert run([1, 2]) == [1, 2]
    assert host.pulled == []
