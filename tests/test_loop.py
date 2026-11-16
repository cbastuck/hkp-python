"""Repetition as a service.

What is worth pinning is the shape of the loop: what each round is given, when
it ends and why, what a call with nothing does, and that Stop gets through
while a call is running.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from hkp.data import MixedData
from hkp.services.loop import LoopService
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
        self.before_round: Any = None

    def configure(self, config: dict) -> dict:
        self._state.update(config)
        return self.get_state()

    def get_state(self) -> dict:
        return dict(self._state)

    def process(self, input: Any, _notify: Any) -> Any:
        if self.before_round:
            self.before_round(input)
        SEEN.append(input)
        if self._state.get("failOn") == input:
            raise RuntimeError("malformed")
        if self._state.get("stopOn") == input:
            return None
        if isinstance(input, MixedData):
            return input
        return {"n": input["n"] + 1}


class _Host:
    def __init__(self) -> None:
        self.contexts: list[ProcessContext | None] = []

    def current_context(self) -> ProcessContext | None:
        return None

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_settings(self) -> dict:
        return {"logging": False, "log_data": False, "log_level": "info"}

    def notify(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def forward_log(self, _entry: Any) -> None:
        return None


def _loop(state: dict | None = None, nested: dict | None = None):
    SEEN.clear()
    fakes: list[_Fake] = []

    def create(cfg: ServiceConfiguration, _cs: Any = None) -> _Fake:
        fakes.append(_Fake(cfg))
        return fakes[-1]

    service = LoopService(
        ServiceConfiguration(
            service_id="loop",
            uuid="loop-1",
            state={
                "pipeline": [{"serviceId": "fake", "instanceId": "inner-1", "state": nested or {}}],
                **(state or {}),
            },
        ),
        create,
    )
    service.set_host(_Host())
    notifications: list[Any] = []

    def run(input: Any = None) -> Any:
        return service.process(input, lambda payload, _id=None: notifications.append(payload))

    return service, run, notifications, fakes[-1]


def test_gives_each_round_what_the_round_before_returned():
    _, run, _, _ = _loop({"maxRounds": 3})

    assert run({"n": 0}) == {"n": 3}
    assert SEEN == [{"n": 0}, {"n": 1}, {"n": 2}]


def test_ends_when_the_condition_holds_on_a_rounds_output():
    service, run, _, _ = _loop({"until": "params.n >= 2"})

    assert run({"n": 0}) == {"n": 2}
    assert len(SEEN) == 2
    state = service.get_state()
    assert (state["done"], state["reason"], state["round"]) == (True, "until", 2)


def test_ends_at_the_limit_and_says_that_is_why():
    service, run, _, _ = _loop({"until": "params.n >= 50", "maxRounds": 4})

    assert run({"n": 0}) == {"n": 4}
    assert service.get_state()["reason"] == "maxRounds"


def test_reports_every_round_with_its_output():
    _, run, notifications, _ = _loop({"maxRounds": 2})

    run({"n": 0})

    assert notifications == [
        {"round": 1, "done": False, "reason": "", "output": {"n": 1}},
        {"round": 2, "done": True, "reason": "maxRounds", "output": {"n": 2}},
    ]


def test_step_runs_one_round_per_call_and_a_call_with_nothing_carries_on():
    service, run, _, _ = _loop({"mode": "step", "until": "params.n >= 3"})

    assert run({"n": 0}) == {"n": 1}
    assert run() == {"n": 2}
    assert run() == {"n": 3}
    assert service.get_state()["done"] is True
    # Done: a further step has nothing to do, and passes nothing on.
    assert run() is None
    assert len(SEEN) == 3


def test_a_stepped_loop_can_be_run_to_its_end():
    service, run, _, _ = _loop({"mode": "step", "until": "params.n >= 5"})

    run({"n": 0})
    service.configure({"mode": "run"})

    assert run() == {"n": 5}
    assert SEEN[0] == {"n": 0} and len(SEEN) == 5


def test_a_value_starts_over():
    service, run, _, _ = _loop({"until": "params.n >= 2"})

    run({"n": 0})
    assert run({"n": 10}) == {"n": 11}
    assert service.get_state()["round"] == 1


def test_a_call_with_nothing_and_nothing_held_passes_nothing_on():
    _, run, _, _ = _loop()

    assert run() is None
    assert SEEN == []


def test_stop_arrives_between_rounds_and_the_loop_can_be_carried_on():
    service, run, notifications, fake = _loop({"maxRounds": 10})
    # Stop arrives as a configure while the third round is running.
    fake.before_round = lambda value: value["n"] == 2 and service.configure({"stop": True})

    assert run({"n": 0}) == {"n": 3}
    state = service.get_state()
    assert (state["done"], state["reason"], state["running"]) == (False, "stopped", False)

    fake.before_round = None
    assert run() == {"n": 10}
    assert notifications[-1]["reason"] == "maxRounds"


def test_stop_from_another_thread_ends_a_loop_with_no_limit():
    service, run, _, fake = _loop({"maxRounds": 0})
    reached = threading.Event()
    fake.before_round = lambda value: value["n"] >= 5 and reached.set()
    result: list[Any] = []
    worker = threading.Thread(target=lambda: result.append(run({"n": 0})))

    worker.start()
    assert reached.wait(5)
    service.configure({"stop": True})
    worker.join(5)

    assert not worker.is_alive()
    assert result[0]["n"] >= 5
    assert service.get_state()["reason"] == "stopped"


def test_a_pipeline_that_stops_ends_the_loop_with_the_last_value():
    service, run, _, _ = _loop({}, {"stopOn": {"n": 2}})

    assert run({"n": 0}) == {"n": 2}
    assert service.get_state()["reason"] == "ended"


def test_a_pipeline_that_stops_in_the_first_round_passes_nothing_on():
    _, run, notifications, _ = _loop({}, {"stopOn": {"n": 0}})

    assert run({"n": 0}) is None
    assert notifications == [{"round": 0, "done": True, "reason": "ended"}]


def test_a_failing_round_raises_and_keeps_the_value_before_it():
    service, run, _, fake = _loop({"until": "params.n >= 4"}, {"failOn": {"n": 2}})

    with pytest.raises(RuntimeError, match="malformed"):
        run({"n": 0})
    state = service.get_state()
    assert (state["round"], state["running"], state["done"]) == (2, False, False)

    # Carried on once what failed is put right.
    fake.configure({"failOn": None})
    assert run() == {"n": 4}


def test_reset_drops_what_is_held():
    service, run, _, _ = _loop({"mode": "step"})

    run({"n": 0})
    service.configure({"reset": True})

    assert run() is None
    assert service.get_state()["round"] == 0


def test_a_condition_that_does_not_parse_is_refused_when_configured():
    service, _, _, _ = _loop()

    with pytest.raises(ValueError, match="invalid expression"):
        service.configure({"until": "params.n >="})
    assert service.get_state()["until"] == ""


def test_a_value_that_is_not_json_goes_round_untouched():
    _, run, _, _ = _loop({"maxRounds": 2})
    value = MixedData({"shape": [1, 2]}, b"\x00" * 8)

    assert run(value) is value
    assert SEEN == [value, value]


def test_with_nothing_in_it_or_bypassed_passes_its_input_through():
    service, run, _, _ = _loop({"bypass": True})
    assert run({"n": 0}) == {"n": 0}
    assert SEEN == []

    service.configure({"bypass": False, "pipeline": []})
    assert run({"n": 0}) == {"n": 0}


def test_reports_what_it_repeats_beside_its_pipeline():
    service, _, _, _ = _loop({"until": "params.done", "maxRounds": 7, "mode": "step"})

    state = service.get_state()
    assert state["until"] == "params.done"
    assert state["maxRounds"] == 7
    assert state["mode"] == "step"
    assert state["pipeline"][0]["instanceId"] == "inner-1"
