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


# ── A detour that fails, told apart from one that chose to stop ──────────────
#
# Ported from hkp-node's "when something in the detour fails": both end with
# nothing, and only the failure has a reason, which `onFailure: "merge"` hands
# on beside the input.


class _Judging:
    """Fails for an input carrying `fail`, stops for one carrying `skip`."""

    service_id = "judging"
    service_name = "Judging"

    def __init__(self, config: ServiceConfiguration, _create: Any = None) -> None:
        self.uuid = config.uuid

    def configure(self, config: dict) -> dict:
        return {}

    def get_state(self) -> dict:
        return {}

    def process(self, input: Any, notify: Any) -> Any:
        if input.get("fail"):
            notify({"error": input["fail"]})
            return None
        if input.get("skip"):
            return None
        return {"text": f"brief of {input['id']}"}


def _judged(state: dict) -> JoinService:
    return JoinService(
        ServiceConfiguration(
            service_id="join",
            uuid="brief",
            state={
                "pipeline": [
                    {"instanceId": "write", "serviceId": "judging", "serviceName": "Judging"}
                ],
                **state,
            },
        ),
        lambda config: _Judging(config),
    )


def test_a_failed_detour_goes_on_with_its_failure_where_the_result_would_be():
    join = _judged({"as": "brief", "onFailure": "merge"})
    assert join.process({"id": "a", "fail": "no value stored for anthropic.key"}, _noop) == {
        "id": "a",
        "fail": "no value stored for anthropic.key",
        "brief": {"error": "no value stored for anthropic.key", "service": "write"},
    }


def test_a_detour_that_chose_to_stop_still_stops_with_merge():
    join = _judged({"as": "brief", "onFailure": "merge"})
    assert join.process({"id": "a", "skip": True}, _noop) is None


def test_a_failure_stops_unless_the_board_asked_to_hear_it():
    join = _judged({"as": "brief"})
    assert join.process({"id": "a", "fail": "boom"}, _noop) is None


def test_a_detour_that_worked_is_merged_as_before_with_merge():
    join = _judged({"as": "brief", "onFailure": "merge"})
    assert join.process({"id": "a"}, _noop) == {"id": "a", "brief": {"text": "brief of a"}}


def test_it_says_how_it_handles_a_failure():
    join = _judged({"onFailure": "merge"})
    assert join.get_state()["onFailure"] == "merge"
    join.configure({"onFailure": "nonsense"})
    assert join.get_state()["onFailure"] == "merge"
