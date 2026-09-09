"""Where a runtime's entries go: the socket the board's coordinator collects on.

`test_board_log.py` covers what a runtime records. This covers the leg after
that — recording into nothing is the same as not recording at all, which is
exactly what this runtime did before the server registered a log target.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.monitor import MONITOR_DESCRIPTOR


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start():
        server = create_runtime_server({"external_host": "127.0.0.1"})
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


async def create_runtime(session, base_url: str, state: dict[str, Any]) -> str:
    """A runtime holding one monitor that feeds the log. Returns its socket."""
    async with session.post(
        f"{base_url}/runtimes",
        json={
            "id": "rt-1",
            "name": "Python",
            "state": state,
            "services": [
                {
                    "serviceId": MONITOR_DESCRIPTOR.service_id,
                    "uuid": "monitor-1",
                    "state": {"logToBoard": True, "logToConsole": False},
                }
            ],
        },
    ) as response:
        assert response.status == 200
        body = await response.json()
    return body["runtimes"][0]["outputUrl"]


async def collect_entries(
    ws_url: str,
    run,
    drain_seconds: float = 0.3,
) -> list[dict[str, Any]]:
    """Every log entry the runtime socket carried while `run` ran."""
    seen: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            await ws.send_str(json.dumps({"type": "readwrite", "id": "rt-1"}))

            async def reader() -> None:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    message = json.loads(msg.data)
                    if message.get("type") == "log":
                        seen.append(message["entry"])

            reader_task = asyncio.create_task(reader())
            await run()
            await asyncio.sleep(drain_seconds)
            reader_task.cancel()

    return seen


async def drive(base_url: str) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes/rt-1", json={"note": "hello"}
        ) as response:
            assert response.status == 200


@pytest.mark.asyncio
async def test_entries_reach_whoever_collects_this_runtimes_output(servers):
    # Regression: the runtime built entries and handed them to its log targets,
    # and the server registered none — so a deployed board's python half
    # recorded into nothing while appearing to be logging.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await create_runtime(
            session, base_url, {"logging": True, "logLevel": "info"}
        )

    entries = await collect_entries(ws_url, lambda: drive(base_url))

    assert [entry["event"] for entry in entries] == ["monitor"]


@pytest.mark.asyncio
async def test_an_entry_is_spelled_the_way_the_other_runtimes_spell_one(servers):
    # A board's log is one file assembled from every runtime it spans, so an
    # entry from here has to be indistinguishable from one hkp-node wrote.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await create_runtime(
            session, base_url, {"logging": True, "logLevel": "debug"}
        )

    entries = await collect_entries(ws_url, lambda: drive(base_url))
    monitor = next(entry for entry in entries if entry["event"] == "monitor")

    assert monitor["runtimeId"] == "rt-1"
    assert monitor["serviceUuid"] == "monitor-1"
    assert monitor["level"] == "info"
    assert isinstance(monitor["runId"], str) and monitor["runId"]
    # Spelled as hkp-node and hkp-rt spell it: milliseconds and a trailing Z.
    # The log store orders and filters entries by comparing this as text, so a
    # runtime with its own dialect sorts against the others rather than among
    # them.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", monitor["ts"])
    assert monitor["data"] == {"note": "hello"}
    # Absent rather than null: a reader that asks whether an entry carries a
    # payload must not see one on every entry.
    assert "parentRunId" not in monitor
    assert "durationMs" not in monitor

    # The flow the runtime records for itself travels the same way, which is
    # what makes "where did this stop" answerable from the board's log.
    processed = next(
        entry for entry in entries if entry["event"] == "service.processed"
    )
    assert isinstance(processed["durationMs"], (int, float))
    assert "data" not in processed


@pytest.mark.asyncio
async def test_a_board_that_logs_nothing_sends_nothing(servers):
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await create_runtime(session, base_url, {})

    assert await collect_entries(ws_url, lambda: drive(base_url)) == []


@pytest.mark.asyncio
async def test_logging_can_be_turned_on_without_rebuilding_the_runtime(servers):
    # What the coordinator's per-board log switch calls. Re-provisioning to
    # carry it would restart every service in the runtime to change one boolean
    # — and a runtime that does not answer this is reported to the user as one
    # the switch did not reach.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await create_runtime(session, base_url, {})

        async with session.patch(
            f"{base_url}/runtimes/rt-1/state",
            json={"logging": True, "logLevel": "info"},
        ) as response:
            assert response.status == 200
            assert await response.json() == {
                "logging": True,
                "logData": True,
                "logLevel": "info",
            }

    entries = await collect_entries(ws_url, lambda: drive(base_url))
    assert [entry["event"] for entry in entries] == ["monitor"]


@pytest.mark.asyncio
async def test_payloads_can_be_refused_while_logging_stays_on(servers):
    # Two decisions, not one: what a board records about its own flow and what
    # it is willing to write down of the data passing through it.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await create_runtime(
            session, base_url, {"logging": True, "logLevel": "info"}
        )
        async with session.patch(
            f"{base_url}/runtimes/rt-1/state", json={"logData": False}
        ) as response:
            assert response.status == 200
            assert (await response.json())["logData"] is False

    entries = await collect_entries(ws_url, lambda: drive(base_url))
    monitor = next(entry for entry in entries if entry["event"] == "monitor")
    assert "data" not in monitor
