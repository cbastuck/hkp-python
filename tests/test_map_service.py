"""Unit tests for the Map service — mirrors hkp-node/tests/service-map.test.ts."""
from __future__ import annotations

from typing import Any

from hkp.services.map_service import MapService
from hkp.types import ServiceConfiguration


def create_map(state: dict[str, Any] | None = None) -> MapService:
    return MapService(ServiceConfiguration(service_id="map", uuid="map-1", state=state))


class RecordingHost:
    """A host that records what a service pushes and notifies."""

    def __init__(self) -> None:
        self.notifications: list[tuple[str, Any]] = []
        self.processed: list[Any] = []
        self.results: list[Any] = []

    def process_from(self, _start_after_uuid: str, data: Any, _on_notification: Any) -> Any:
        self.processed.append(data)
        return data

    def notify(self, payload: Any, instance_id: str) -> None:
        self.notifications.append((instance_id, payload))

    def emit_result(self, output: Any) -> None:
        self.results.append(output)


# ── Flat templates ────────────────────────────────────────────────────────────


def test_static_properties_and_dynamic_terms():
    svc = create_map({"template": {"label": "hello", "answer=": "params.value * 2"}})

    assert svc.process({"value": 21}) == {"label": "hello", "answer": 42}


def test_lone_equals_maps_to_a_scalar():
    svc = create_map({"template": {"=": "params.value + 1"}})

    assert svc.process({"value": 1}) == 2


def test_dotted_keys_nest():
    svc = create_map({"template": {"position.x=": "params.n", "position.y": 0}})

    assert svc.process({"n": 3}) == {"position": {"x": 3, "y": 0}}


def test_merge_modes():
    template = {"value=": "params.value * 2", "extra": True}
    data = {"value": 2, "keep": 1}

    assert create_map({"template": template, "mode": "replace"}).process(data) == {
        "value": 4,
        "extra": True,
    }
    assert create_map({"template": template, "mode": "overwrite"}).process(data) == {
        "value": 4,
        "keep": 1,
        "extra": True,
    }
    # add: input wins over template for keys that already exist
    assert create_map({"template": template, "mode": "add"}).process(data) == {
        "value": 2,
        "keep": 1,
        "extra": True,
    }


def test_maps_each_element_of_an_array_input():
    svc = create_map({"template": {"n=": "params.n + 1"}})

    assert svc.process([{"n": 1}, {"n": 2}]) == [{"n": 2}, {"n": 3}]


def test_array_mode_single_maps_the_array_as_a_whole():
    svc = create_map({"arrayMode": "single", "template": {"count=": "params.length"}})

    assert svc.process([1, 2, 3]) == {"count": 3}


def test_empty_template():
    assert create_map().process({"value": 1}) == {}
    assert create_map({"mode": "overwrite"}).process({"value": 1}) == {"value": 1}


def test_returns_the_input_unchanged_when_an_expression_fails():
    svc = create_map({"template": {"x=": "nope("}})

    assert svc.process({"value": 1}) == {"value": 1}


def test_a_missing_path_yields_null_rather_than_failing_the_mapping():
    svc = create_map({"template": {"x=": "params.missing.deep", "y": 1}})

    assert svc.process({"value": 1}) == {"x": None, "y": 1}


# ── Structured templates ──────────────────────────────────────────────────────


def test_nested_objects_keep_their_shape():
    svc = create_map({"template": {"outer": {"inner=": "params.value * 3", "static": "keep"}}})

    assert svc.process({"value": 2}) == {"outer": {"inner": 6, "static": "keep"}}
    assert svc.get_state()["template"] == {
        "outer": {"inner=": "params.value * 3", "static": "keep"}
    }


def test_array_template_maps_into_an_array_result():
    svc = create_map({"template": [{"role=": "'user'"}, {"text=": "params.text"}]})

    assert svc.process({"text": "hi"}) == [{"role": "user"}, {"text": "hi"}]


def test_structured_result_merges_with_the_input_in_overwrite_mode():
    svc = create_map({"mode": "overwrite", "template": {"nested": {"a=": "params.a"}}})

    assert svc.process({"a": 1, "keep": 2}) == {"a": 1, "keep": 2, "nested": {"a": 1}}


# ── Expression scope ──────────────────────────────────────────────────────────


def test_shared_helper_functions():
    svc = create_map(
        {
            "template": {
                "rounded=": "round(params.value)",
                "joined=": "concat('a', 'b')",
                "total=": "sum(params.list)",
                "picked=": "find(params.list, 'item > 1')",
                "branch=": "params.value > 1 ? 'big' : 'small'",
            }
        }
    )

    assert svc.process({"value": 1.6, "list": [1, 2, 3]}) == {
        "rounded": 2,
        "joined": "ab",
        "total": 6,
        "picked": 2,
        "branch": "big",
    }


def test_moment_style_date_tokens():
    svc = create_map({"template": {"date=": "reformatDate(params.d, 'YYYY-MM-DD', 'DD.MM.YYYY')"}})

    assert svc.process({"d": "2026-07-30"}) == {"date": "30.07.2026"}


def test_python_spellings_still_evaluate():
    svc = create_map(
        {
            "template": {
                "flag=": "params.a and not params.b",
                "size=": "len(params.list)",
                "text=": "str(params.a)",
            }
        }
    )

    assert svc.process({"a": True, "b": False, "list": [1, 2]}) == {
        "flag": True,
        "size": 2,
        "text": "true",
    }


# ── UI-driven behaviour ───────────────────────────────────────────────────────


def test_sensing_mode_learns_a_flat_template():
    svc = create_map({"sensingMode": True})

    assert svc.process({"a": {"b": 1}, "c": "x"}) is None

    state = svc.get_state()
    assert state["sensingMode"] is False
    assert state["template"] == {"a.b": 1, "c": "x"}
    assert svc.process({}) == {"a": {"b": 1}, "c": "x"}


def test_notifies_the_ui_about_template_mode_and_sensing_changes():
    svc = create_map()
    host = RecordingHost()
    svc.set_host(host)

    svc.configure({"mode": "add"})
    svc.configure({"sensingMode": True})
    svc.configure({"template": {"x=": "params.x"}})

    assert host.notifications == [
        ("map-1", {"mode": "add"}),
        ("map-1", {"sensingMode": True}),
        ("map-1", {"template": {"x=": "params.x"}}),
    ]


def test_inject_command_pushes_the_mapped_result_downstream():
    svc = create_map({"template": {"greeting=": "'hi ' + params.name"}})
    host = RecordingHost()
    svc.set_host(host)

    svc.configure({"command": {"action": "inject", "params": {"name": "ada"}}})

    assert host.processed == [{"greeting": "hi ada"}]
    assert host.results == [{"greeting": "hi ada"}]


def test_inject_pushes_nothing_when_the_mapping_stops_the_flow():
    svc = create_map({"sensingMode": True})
    host = RecordingHost()
    svc.set_host(host)

    svc.configure({"command": {"action": "inject", "params": {"a": 1}}})

    assert host.processed == []
    assert host.results == []


# ── An address a board mentions rather than dials ─────────────────────────────


def _map_with(state: dict) -> MapService:
    return MapService(ServiceConfiguration(service_id="map", uuid="map-1", state=state))


def test_a_mount_reference_becomes_the_address_handed_over():
    service = _map_with({"mode": "replace", "template": {"base": "hkp-mount://library/listen"}})

    service.configure({"__hkpMount": "http://host:8080/hosted/abc"})

    assert service.process({}) == {"base": "http://host:8080/hosted/abc"}


def test_a_reference_an_expression_produced_is_covered_too():
    service = _map_with(
        {
            "mode": "replace",
            "template": {"base=": "params.target"},
            "__hkpMount": "http://host:8080/hosted/abc",
        }
    )

    assert service.process({"target": "hkp-mount://library/listen"}) == {
        "base": "http://host:8080/hosted/abc"
    }


def test_an_unresolved_reference_is_left_as_it_stands():
    # Visibly not an address beats an empty field that looks unfilled.
    service = _map_with({"mode": "replace", "template": {"base": "hkp-mount://library/listen"}})

    assert service.process({}) == {"base": "hkp-mount://library/listen"}


def test_everything_that_is_not_a_reference_is_untouched():
    service = _map_with(
        {
            "mode": "replace",
            "template": {"url": "https://example.test/feed.xml", "n": 3},
            "__hkpMount": "http://host:8080/hosted/abc",
        }
    )

    assert service.process({}) == {"url": "https://example.test/feed.xml", "n": 3}


def test_without_urls_strips_addresses_and_the_labels_that_introduced_them():
    """Text on its way to a voice.

    Feed summaries are written for programs as much as for people. Spoken, an
    address is a minute of punctuation, and the label that introduced it says
    nothing once it is gone.
    """
    hacker_news = (
        "Article URL: https://overreacted.io/how-i-vibed-a-proof/ "
        "Comments URL: https://news.ycombinator.com/item?id=49755024 "
        "Points: 48 # Comments: 31"
    )
    service = create_map(
        {"mode": "replace", "template": {"text=": "withoutUrls(params.summary)"}}
    )

    assert service.process({"summary": hacker_news}, lambda _n: None) == {
        "text": "Points: 48 # Comments: 31"
    }
    # Prose keeps every word it had; an absent field is nothing said, not "null".
    assert service.process({"summary": "Police said so."}, lambda _n: None) == {
        "text": "Police said so."
    }
    assert service.process({}, lambda _n: None) == {"text": ""}
