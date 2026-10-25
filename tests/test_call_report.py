"""A call report carries the value only up to a limit; above it, a description.

The same cases as hkp-node's call-report.test.ts, hkp-frontend's
call-report.test.ts and hkp-rt's call_report.test.cpp.
"""
from __future__ import annotations

import time
from typing import Any

from hkp.call_report import CALL_REPORT_LIMIT, PREVIEW_LENGTH, is_too_large, reported_value
from hkp.data import TextData
from hkp.runtime import HostedRuntime
from hkp.types import RuntimeConfiguration, ServiceConfiguration

LIMIT = 100


def test_a_value_small_enough_is_reported_as_it_is():
    value = {"a": [1, "two", None, True]}
    assert reported_value(value, LIMIT) is value
    assert reported_value("short", LIMIT) == "short"
    assert reported_value(42, LIMIT) == 42
    assert reported_value(None, LIMIT) is None
    assert reported_value(bytes(LIMIT), LIMIT) == bytes(LIMIT)


def test_anything_that_is_not_data_is_not_looked_inside():
    class Holder:
        def __init__(self) -> None:
            self.big = "x" * (LIMIT * 10)

    holder = Holder()
    assert reported_value(holder, LIMIT) is holder
    assert not is_too_large(reported_value({"holder": holder}, LIMIT))


def test_a_text_is_described_by_its_size_in_bytes_and_how_it_begins():
    text = "é" * 60  # 120 bytes in 60 characters
    assert reported_value(text, LIMIT) == {
        "__hkpTooLarge": {"type": "string", "bytes": 120, "exact": True, "preview": text}
    }
    # Text data reads as the text it holds.
    assert reported_value(TextData(text), LIMIT)["__hkpTooLarge"]["type"] == "string"


def test_a_text_longer_than_the_limit_in_characters_is_not_counted_further():
    assert reported_value("x" * (PREVIEW_LENGTH * 2), LIMIT) == {
        "__hkpTooLarge": {
            "type": "string",
            "bytes": PREVIEW_LENGTH * 2,
            "exact": False,
            "preview": "x" * PREVIEW_LENGTH,
        }
    }


def test_bytes_are_described_by_their_size_alone():
    assert reported_value(bytes(LIMIT + 1), LIMIT) == {
        "__hkpTooLarge": {"type": "binary", "bytes": LIMIT + 1, "exact": True}
    }


def test_json_is_counted_only_until_it_passes_the_limit_with_how_it_begins():
    described = reported_value({"items": [f"item-{i}" for i in range(50)]}, LIMIT)["__hkpTooLarge"]
    assert described["type"] == "json"
    assert described["exact"] is False
    assert described["bytes"] > LIMIT
    assert described["preview"].startswith('{"items":["item-0","item-1"')


def test_json_is_judged_without_visiting_all_of_it():
    huge = [1] * 10_000_000
    started = time.monotonic()
    described = reported_value(huge)["__hkpTooLarge"]
    assert time.monotonic() - started < 0.2
    assert described["type"] == "json"
    assert described["exact"] is False


def test_a_preview_is_at_most_its_length_ending_in_an_ellipsis():
    described = reported_value({"text": "y" * 5000}, LIMIT)["__hkpTooLarge"]
    assert described["preview"] == '{"text":"' + "y" * (PREVIEW_LENGTH - 9) + "…"


def test_a_description_is_told_apart_from_a_value():
    assert is_too_large({"__hkpTooLarge": {"type": "json", "bytes": 1, "exact": False}})
    assert not is_too_large({"tooLarge": True})
    assert not is_too_large([{"__hkpTooLarge": {}}])
    assert not is_too_large(None)


class _Producer:
    """Answers with a text of ``length`` characters, whatever it is given."""

    service_id = "producer"
    service_name = "Producer"

    def __init__(self, config: ServiceConfiguration, length: int) -> None:
        self.uuid = config.uuid
        self._length = length

    def configure(self, config: dict) -> dict:
        return {}

    def get_state(self) -> dict:
        return {}

    def process(self, _input: Any, _notify: Any) -> Any:
        return "z" * self._length


def _finished_report(length: int) -> tuple[Any, Any]:
    runtime = HostedRuntime(
        RuntimeConfiguration(
            id="rt",
            name="Rt",
            services=[ServiceConfiguration(service_id="producer", uuid="p")],
        ),
        lambda config: _Producer(config, length),
    )
    reports: list[Any] = []
    result = runtime.process_at("p", "in", reports.append)
    finished = next(
        n.payload["__internal"]
        for n in reports
        if isinstance(n.payload, dict)
        and n.payload.get("__internal", {}).get("state") == "call-process-finished"
    )
    return result, finished


def test_a_runtime_describes_a_value_above_the_limit_while_the_run_carries_all_of_it():
    result, finished = _finished_report(CALL_REPORT_LIMIT + 1)

    assert len(result) == CALL_REPORT_LIMIT + 1
    assert finished["data"]["__hkpTooLarge"]["type"] == "string"
    assert finished["data"]["__hkpTooLarge"]["bytes"] == CALL_REPORT_LIMIT + 1


def test_a_runtime_reports_a_value_at_the_limit_as_it_is():
    _, finished = _finished_report(CALL_REPORT_LIMIT)

    assert len(finished["data"]) == CALL_REPORT_LIMIT
