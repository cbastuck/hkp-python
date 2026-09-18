"""The answer beside what you already had, rather than instead of it.

Ported from hkp-node/tests/join.test.ts: the same cases, because the point of
the service is that one board reads the same whichever runtime hosts it.
"""
from __future__ import annotations

from typing import Any

from hkp.services.join import JoinService
from hkp.types import ServiceConfiguration


class _Echo:
    """A service standing in for the detour: it answers, and answers only."""

    service_id = "echo"
    service_name = "Echo"

    def __init__(self, config: ServiceConfiguration, _create: Any = None) -> None:
        self.uuid = config.uuid
        self._answer = (config.state or {}).get("answer")

    def configure(self, config: dict) -> dict:
        if "answer" in config:
            self._answer = config["answer"]
        return self.get_state()

    def get_state(self) -> dict:
        return {"answer": self._answer}

    def process(self, _input: Any, _notify: Any) -> Any:
        return self._answer


def _join(state: dict, answer: Any) -> JoinService:
    def create(config: ServiceConfiguration):
        return _Echo(config)

    return JoinService(
        ServiceConfiguration(
            service_id="join",
            uuid="join-1",
            state={
                "pipeline": [
                    {
                        "instanceId": "echo",
                        "serviceId": "echo",
                        "serviceName": "Echo",
                        "state": {"answer": answer},
                    }
                ],
                **state,
            },
        ),
        create,
    )


def _noop(_payload: Any, _instance_id: str | None = None) -> None:
    return None


def test_the_result_arrives_under_the_name_it_was_given():
    service = _join({"as": "audio"}, b"\xff\xfb")

    merged = service.process({"path": "episodes/one.mp3", "text": "hello"}, _noop)

    # The carrier never went anywhere; the detour's answer sits beside it.
    assert merged == {"path": "episodes/one.mp3", "text": "hello", "audio": b"\xff\xfb"}


def test_a_named_result_cannot_collide_with_what_the_input_carried():
    service = _join({"as": "result"}, {"path": "elsewhere"})

    merged = service.process({"path": "episodes/one.mp3"}, _noop)

    assert merged == {"path": "episodes/one.mp3", "result": {"path": "elsewhere"}}


def test_a_scalar_input_keeps_its_place_under_input():
    service = _join({"as": "answer"}, 42)

    assert service.process("a question", _noop) == {"input": "a question", "answer": 42}


def test_merging_at_the_top_level_prefers_the_result_by_default():
    service = _join({}, {"b": 2, "a": "from the detour"})

    assert service.process({"a": "from the input", "c": 3}, _noop) == {
        "a": "from the detour",
        "b": 2,
        "c": 3,
    }


def test_add_mode_lets_the_input_win():
    service = _join({"mode": "add"}, {"a": "from the detour", "b": 2})

    assert service.process({"a": "from the input"}, _noop) == {
        "a": "from the input",
        "b": 2,
    }


def test_a_detour_that_stopped_stops_this_one():
    # Continuing with the input alone would be indistinguishable downstream
    # from a merge that worked.
    service = _join({"as": "audio"}, None)

    assert service.process({"path": "episodes/one.mp3"}, _noop) is None


def test_an_empty_join_passes_its_input_through():
    service = JoinService(
        ServiceConfiguration(service_id="join", uuid="join-1", state={"pipeline": []}),
        lambda config: _Echo(config),
    )

    assert service.process({"path": "one"}, _noop) == {"path": "one"}
