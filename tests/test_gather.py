"""The other half of Spread.

Ported from hkp-node/tests/gather.test.ts. What is worth pinning is when a
group counts as complete, what it becomes, and what happens to one that never
completes — the rest is the pipeline's, and pinned where the pipeline is.
"""
from __future__ import annotations

import time
from typing import Any

from hkp.services.gather import GatherService, is_part
from hkp.services.map_service import MapService
from hkp.types import ServiceConfiguration


class _Stops:
    """Stands in for a reducer that stops: the one behaviour `map` cannot show."""

    service_id = "stops"
    service_name = "Stops"

    def __init__(self, config: ServiceConfiguration) -> None:
        self.uuid = config.uuid

    def configure(self, config: dict) -> dict:
        return {}

    def get_state(self) -> dict:
        return {}

    def process(self, _input: Any, _notify: Any) -> Any:
        return None


def _create(config: ServiceConfiguration, _cs: Any = None):
    return MapService(config) if config.service_id == "map" else _Stops(config)


class _Host:
    def __init__(self) -> None:
        self.logged: list[tuple[str, str, Any]] = []

    def current_context(self):
        return None

    def log(self, level: str, event: str, data: Any = None) -> None:
        self.logged.append((level, event, data))

    def log_settings(self) -> dict:
        return {"logging": False, "log_data": False, "log_level": "info"}

    def notify(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def forward_log(self, _entry: Any) -> None:
        return None


def _gather(state: dict | None = None):
    service = GatherService(
        ServiceConfiguration(service_id="gather", uuid="gather-1", state=state or {}),
        _create,
    )
    host = _Host()
    service.set_host(host)
    notifications: list[Any] = []

    def run(input: Any) -> Any:
        return service.process(input, lambda payload, _id=None: notifications.append(payload))

    return service, run, notifications, host


def _part(group: str, index: int, total: int, item: Any) -> dict:
    return {"item": item, "group": group, "index": index, "total": total}


JOIN = [
    {
        "serviceId": "map",
        "instanceId": "join",
        "state": {"template": {"=": "join(params.items, '')"}},
    }
]


def test_passes_nothing_on_until_every_part_is_in_then_the_items_in_order():
    _, run, _, _ = _gather()

    assert run(_part("g", 0, 3, "ab")) is None
    assert run(_part("g", 1, 3, "cd")) is None
    assert run(_part("g", 2, 3, "ef")) == ["ab", "cd", "ef"]


def test_puts_parts_back_in_order_whatever_order_they_arrived_in():
    _, run, _, _ = _gather()

    run(_part("g", 2, 3, "c"))
    run(_part("g", 0, 3, "a"))

    assert run(_part("g", 1, 3, "b")) == ["a", "b", "c"]


def test_keeps_two_groups_apart():
    _, run, _, _ = _gather()

    run(_part("one", 0, 2, "1a"))
    run(_part("two", 0, 2, "2a"))

    assert run(_part("two", 1, 2, "2b")) == ["2a", "2b"]
    assert run(_part("one", 1, 2, "1b")) == ["1a", "1b"]


def test_counts_a_part_sent_twice_once():
    _, run, _, _ = _gather()

    run(_part("g", 0, 2, "a"))
    assert run(_part("g", 0, 2, "a again")) is None

    assert run(_part("g", 1, 2, "b")) == ["a", "b"]


def test_says_how_far_a_group_has_got():
    service, run, notifications, _ = _gather()

    run(_part("g", 0, 2, "a"))

    assert notifications == [{"group": "g", "received": 1, "total": 2, "pending": 1}]
    assert service.get_state()["pending"] == 1


def test_passes_on_what_is_not_a_part():
    _, run, _, _ = _gather({"pipeline": JOIN})

    assert run("a whole file") == "a whole file"
    assert run({"item": "x", "group": "g"}) == {"item": "x", "group": "g"}


def test_a_group_becomes_what_the_pipeline_makes_of_it():
    _, run, _, _ = _gather({"pipeline": JOIN})

    run(_part("g", 1, 2, "world"))

    assert run(_part("g", 0, 2, "hello ")) == "hello world"


def test_stops_here_when_the_pipeline_stops():
    _, run, _, _ = _gather({"pipeline": [{"serviceId": "stops", "instanceId": "s"}]})

    assert run(_part("g", 0, 1, "a")) is None


def test_is_nothing_passed_on_when_the_board_says_so():
    _, run, _, _ = _gather({"stopPropagation": True})

    assert run(_part("g", 0, 1, "a")) is None


def test_is_refused_once_it_grows_past_max_bytes():
    service, run, notifications, host = _gather({"maxBytes": 5})

    assert run(_part("g", 0, 3, "abc")) is None
    assert run(_part("g", 1, 3, "def")) is None
    # The rest of a refused group does not start it again.
    assert run(_part("g", 2, 3, "g")) is None

    assert service.get_state()["pending"] == 0
    assert "larger than 5 bytes" in service.get_state()["error"]
    assert [event for _, event, _ in host.logged] == ["gather.refused"]
    assert {"error": "group g is larger than 5 bytes", "pending": 0} in notifications


def test_is_let_go_after_timeout_ms_and_said_so():
    service, run, _, host = _gather({"timeoutMs": 10})

    run(_part("old", 0, 2, "a"))
    time.sleep(0.02)
    run(_part("new", 0, 2, "b"))

    assert service.get_state()["pending"] == 1
    assert host.logged == [
        ("warn", "gather.timed-out", {"group": "old", "received": 1, "total": 2})
    ]


def test_is_refused_when_its_parts_disagree_on_how_many_there_are():
    service, run, _, _ = _gather()

    run(_part("g", 0, 2, "a"))

    assert run(_part("g", 1, 3, "b")) is None
    assert "said 2 parts, then 3" in service.get_state()["error"]


def test_what_counts_as_a_part():
    assert is_part(_part("g", 0, 1, None))
    assert not is_part({"group": "g", "index": 0, "total": 1})
    assert not is_part(_part("", 0, 1, "x"))
    assert not is_part(_part("g", 1, 1, "x"))
    assert not is_part(_part("g", 0.5, 2, "x"))
    assert not is_part(_part("g", True, 2, "x"))
    assert not is_part([_part("g", 0, 1, "x")])


def test_configuration_is_read_from_the_board_as_well_as_set_later():
    service, _, _, _ = _gather({"maxBytes": 1000, "timeoutMs": 5000})

    assert service.get_state()["maxBytes"] == 1000
    assert service.get_state()["timeoutMs"] == 5000
    service.configure({"maxBytes": 2000})
    assert service.get_state()["maxBytes"] == 2000
    assert service.get_state()["timeoutMs"] == 5000
