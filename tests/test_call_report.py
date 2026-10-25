"""A call report carries the value only up to a limit; above it, a description.

The same cases as hkp-node's call-report.test.ts, hkp-frontend's
call-report.test.ts and hkp-rt's call_report.test.cpp.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.call_report import (
    CALL_REPORT_LIMIT,
    PREVIEW_LENGTH,
    is_too_large,
    outgoing_payload,
    reported_value,
)
from hkp.data import TextData
from hkp.server import create_runtime_server
from hkp.services.monitor import MONITOR_DESCRIPTOR

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


def _report(state: str, data: Any = ..., call: int = 7) -> dict:
    internal: dict[str, Any] = {"state": state, "call": call}
    if data is not ...:
        internal["data"] = data
    return {"__internal": internal}


def test_without_capture_mode_only_that_a_call_happened_leaves():
    assert outgoing_payload(_report("call-process", {"secret": 1}), False) == _report("call-process")
    assert outgoing_payload(_report("call-process-finished", {"secret": 1}), False) == _report(
        "call-process-finished"
    )


def test_without_capture_mode_a_call_that_passed_nothing_on_says_so():
    assert outgoing_payload({"__internal": {"state": "call-process-finished", "data": None}}, False) == {
        "__internal": {"state": "call-process-finished", "stopped": True}
    }
    assert outgoing_payload({"__internal": {"state": "call-process-finished"}}, False) == {
        "__internal": {"state": "call-process-finished", "stopped": True}
    }


def test_with_capture_mode_the_value_leaves_bounded():
    assert outgoing_payload(_report("call-process", {"a": 1}), True) == _report("call-process", {"a": 1})
    carried = outgoing_payload(_report("call-process-finished", "z" * (CALL_REPORT_LIMIT + 1)), True)
    assert carried["__internal"]["data"]["__hkpTooLarge"]["type"] == "string"


def test_any_other_notification_leaves_as_it_is():
    own = {"level": 0.5}
    assert outgoing_payload(own, False) is own
    other = {"__internal": {"state": "something-else", "data": 1}}
    assert outgoing_payload(other, False) is other


# ── Out of a real server ────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def server_url():
    server = create_runtime_server({"external_host": "127.0.0.1"})
    address = await server.start(0, "127.0.0.1")
    yield address["base_url"]
    await server.stop()


async def _reports_while(base_url: str, ws_url: str, payload: Any) -> list[dict]:
    """Every call report the runtime's socket carried while `payload` ran."""
    seen: list[dict] = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            await ws.send_str(json.dumps({"type": "readwrite", "id": "rt-1"}))

            async def reader() -> None:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    message = json.loads(msg.data)
                    if message.get("type") != "notification":
                        continue
                    value = json.loads(message["value"])
                    if isinstance(value, dict) and "__internal" in value:
                        seen.append(value["__internal"])

            task = asyncio.create_task(reader())
            async with session.post(f"{base_url}/runtimes/rt-1", json=payload) as response:
                assert response.status == 200
            await asyncio.sleep(0.3)
            task.cancel()
    return seen


@pytest.mark.asyncio
async def test_a_server_sends_call_values_only_while_captured(server_url):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{server_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [{"serviceId": MONITOR_DESCRIPTOR.service_id, "uuid": "monitor-1"}],
            },
        ) as response:
            ws_url = (await response.json())["runtimes"][0]["outputUrl"]

        uncaptured = await _reports_while(server_url, ws_url, {"secret": "inside the run"})
        assert [r["state"] for r in uncaptured] == ["call-process", "call-process-finished"]
        assert all("data" not in r for r in uncaptured)

        async with session.patch(f"{server_url}/runtimes/rt-1/state", json={"capture": True}) as response:
            assert (await response.json())["capture"] is True
        captured = await _reports_while(server_url, ws_url, {"secret": "inside the run"})
        assert captured[0]["data"] == {"secret": "inside the run"}

        async with session.patch(f"{server_url}/runtimes/rt-1/state", json={"capture": False}) as response:
            assert (await response.json())["capture"] is False
        released = await _reports_while(server_url, ws_url, {"secret": "inside the run"})
        assert all("data" not in r for r in released)
