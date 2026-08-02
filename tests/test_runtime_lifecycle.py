"""How long a runtime lives.

Whoever creates a runtime says whether it should be cleaned up when the last
client connected to it disconnects — they are the only ones who know. A browser
running a board is its controller and asks for cleanup: its runtimes should not
outlive the tab. A coordinator, a config file or a script says nothing and gets
a runtime that lives until it is deleted.

Cleanup is opted into rather than assumed, so a runtime is never reaped because
of who happened to connect to it.

Mirrors hkp-node/tests/runtime-lifecycle.test.ts.
"""
from __future__ import annotations

import asyncio
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


async def create_runtime(base_url: str, extra: dict[str, Any]) -> str:
    """Creates a runtime and returns the URL its clients connect to."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [
                    {"uuid": "mon-1", "serviceId": MONITOR_DESCRIPTOR.service_id}
                ],
                **extra,
            },
        ) as res:
            assert res.status == 200
            body = await res.json()
            return body["runtimes"][0]["outputUrl"]


async def runtime_status(base_url: str) -> int:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/runtimes/rt-1") as res:
            return res.status


async def connect_then_close(output_url: str) -> None:
    """Connects a client and drops it — a tab closing."""
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(output_url) as ws:
            await ws.send_json({"type": "readwrite", "id": "rt-1"})
            await asyncio.sleep(0.05)
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_declared_ephemeral_is_reaped_when_the_last_client_leaves(servers):
    _server, base_url = await servers()
    output_url = await create_runtime(base_url, {"garbageCollected": True})

    await connect_then_close(output_url)

    assert await runtime_status(base_url) == 404


@pytest.mark.asyncio
async def test_declared_ephemeral_survives_while_another_client_watches(servers):
    # "Last client", not "the one that created it": something is still using it.
    _server, base_url = await servers()
    output_url = await create_runtime(base_url, {"garbageCollected": True})

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(output_url) as staying:
            await staying.send_json({"type": "readwrite", "id": "rt-1"})
            await connect_then_close(output_url)
            assert await runtime_status(base_url) == 200
    await asyncio.sleep(0.1)

    assert await runtime_status(base_url) == 404


@pytest.mark.asyncio
async def test_declaring_nothing_outlives_the_clients_watching_it(servers):
    # A deployed board keeps running with no browser attached. So does a runtime
    # someone started from a script or a config file.
    _server, base_url = await servers()
    output_url = await create_runtime(base_url, {})

    await connect_then_close(output_url)

    assert await runtime_status(base_url) == 200


@pytest.mark.asyncio
async def test_explicit_false_is_the_same_as_saying_nothing(servers):
    _server, base_url = await servers()
    output_url = await create_runtime(base_url, {"garbageCollected": False})

    await connect_then_close(output_url)

    assert await runtime_status(base_url) == 200


@pytest.mark.asyncio
async def test_a_runtime_nobody_connected_to_is_never_reaped(servers):
    # Cleanup happens when a client goes away. With no client there is no going
    # away — a headless runtime is not an abandoned one.
    _server, base_url = await servers()
    await create_runtime(base_url, {"garbageCollected": True})

    await asyncio.sleep(0.1)

    assert await runtime_status(base_url) == 200


@pytest.mark.asyncio
async def test_deleting_still_works(servers):
    _server, base_url = await servers()
    await create_runtime(base_url, {})

    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{base_url}/runtimes/rt-1") as res:
            assert res.status == 200

    assert await runtime_status(base_url) == 404
