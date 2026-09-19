"""Several pipelines over one input.

Ported from hkp-node/tests/tracks.test.ts: the same cases, because the point of
the service is that one board reads the same whichever runtime hosts it. What
differs here is timing — this runtime processes synchronously, so `run` is
recorded and the tracks go one at a time — and nothing about the answers.
"""
from __future__ import annotations

from typing import Any

from hkp.services.map_service import MapService
from hkp.services.tracks import TracksService
from hkp.types import ServiceConfiguration

#: What each fake service did, in the order it happened.
TRACE: list[str] = []
#: How many services have been built, for the rebuild question.
BUILT = {"count": 0}


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
        TRACE.append(self.uuid)
        if self._state.get("fail"):
            raise RuntimeError("no good")
        if self._state.get("stop"):
            return None
        if "answer" in self._state:
            return self._state["answer"]
        return {"saw": input}


def _create(config: ServiceConfiguration, _cs: Any = None):
    BUILT["count"] += 1
    if config.service_id == "map":
        return MapService(config)
    return _Fake(config)


def _track(name: str, state: dict | None = None) -> dict:
    return {
        "name": name,
        "pipeline": [
            {
                "instanceId": name,
                "serviceId": "fake",
                "serviceName": "Fake",
                "state": state or {},
            }
        ],
    }


def _reducer(template: dict) -> list[dict]:
    return [
        {
            "instanceId": "shape",
            "serviceId": "map",
            "serviceName": "Map",
            "state": {"mode": "replace", "template": template},
        }
    ]


def _tracks(state: dict) -> TracksService:
    return TracksService(
        ServiceConfiguration(service_id="tracks", uuid="tracks-1", state=state),
        _create,
    )


def _noop(_payload: Any, _instance_id: str | None = None) -> None:
    return None


def test_every_track_is_given_the_same_input():
    service = _tracks({"tracks": [_track("keep"), _track("drop")]})

    assert service.process({"intent": "keep"}, _noop) == [
        {"saw": {"intent": "keep"}},
        {"saw": {"intent": "keep"}},
    ]


def test_a_track_that_stopped_leaves_a_hole():
    # Position still names the track that produced it, so a missing answer is
    # visible rather than absent.
    service = _tracks(
        {"tracks": [_track("keep", {"stop": True}), _track("drop", {"answer": {"rows": 1}})]}
    )

    assert service.process({}, _noop) == [None, {"rows": 1}]


def test_tracks_run_in_declaration_order():
    TRACE.clear()
    service = _tracks({"tracks": [_track("first"), _track("second")]})

    service.process({}, _noop)

    assert TRACE == ["first", "second"]


def test_the_reducer_is_given_what_came_in_beside_what_was_answered():
    service = _tracks(
        {
            "tracks": [_track("keep", {"answer": {"rows": 1}})],
            "reduce": _reducer({"wrote=": "params.results[0].rows", "for=": "params.input.link"}),
        }
    )

    assert service.process({"link": "https://example.test"}, _noop) == {
        "wrote": 1,
        "for": "https://example.test",
    }


def test_carrying_the_input_on_is_one_term():
    service = _tracks(
        {
            "tracks": [_track("keep", {"answer": {"changes": 1}}), _track("drop", {"stop": True})],
            "reduce": _reducer({"=": "params.input"}),
        }
    )

    kept = {"intent": "keep", "link": "https://example.test"}
    assert service.process(kept, _noop) == kept


def test_only_the_reducer_stops_the_pipeline():
    # An array of nothing but Nones is still an array; whether that means stop
    # is the board's to say.
    silent = _tracks({"tracks": [_track("a", {"stop": True}), _track("b", {"stop": True})]})
    assert silent.process({}, _noop) == [None, None]

    service = _tracks(
        {
            "tracks": [_track("a", {"stop": True})],
            "reduce": _reducer({"=": "params.results[0]"}),
        }
    )
    assert service.process({}, _noop) is None


def test_one_track_raising_does_not_take_the_others_with_it():
    service = _tracks(
        {"tracks": [_track("broken", {"fail": True}), _track("fine", {"answer": "ok"})]}
    )

    assert service.process({}, _noop) == [None, "ok"]
    assert "track 'broken' failed" in service.get_state()["error"]


def test_bypass_passes_the_input_through_and_a_bypassed_track_leaves_a_hole():
    whole = _tracks({"bypass": True, "tracks": [_track("keep")]})
    assert whole.process({"a": 1}, _noop) == {"a": 1}

    one = _tracks({"tracks": [{**_track("keep"), "bypass": True}, _track("drop", {"answer": "ran"})]})
    assert one.process({}, _noop) == [None, "ran"]


def test_editing_one_track_leaves_the_others_running():
    # Rebuilding a pipeline destroys what is running inside it — a mount, a
    # timer — so an edit to one track must not touch another.
    BUILT["count"] = 0
    service = _tracks({"tracks": [_track("keep"), _track("drop")]})
    assert BUILT["count"] == 2

    service.configure({"tracks": [_track("keep"), _track("drop"), _track("tell")]})

    assert BUILT["count"] == 3
    assert len(service.get_state()["tracks"]) == 3


def test_a_track_may_not_take_the_reducers_name_or_anothers():
    for bad, message in (
        ([_track("reduce")], "reducer"),
        ([_track("keep"), _track("keep")], "both called"),
    ):
        try:
            _tracks({"tracks": bad})
        except ValueError as error:
            assert message in str(error)
        else:  # pragma: no cover - the failure path is the assertion
            raise AssertionError("expected the configuration to be refused")
