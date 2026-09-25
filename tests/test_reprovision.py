"""What happens when a runtime that already exists is provisioned again.

hkp-node reuses the running runtime, so a browser attaching to a
coordinator-managed board does not kill services the coordinator started
(hkp-node/tests/cloud-reprovision.test.ts).

**hkp-python rebuilds it instead.** These tests pin that difference rather than
endorse it: it is a known divergence, listed in plans/TODO-CONSOLIDATION.md. Change
them when python adopts node's behaviour — the assertions below say plainly
which way each one should flip.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR
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


ENDPOINT_SERVICE: dict[str, Any] = {
    "uuid": "http-1",
    "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
    "state": {"bypass": False, "mode": "process_on_session", "pipeline": []},
}


async def provision(base_url: str, runtime_id: str, services: list[dict[str, Any]]):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={"id": runtime_id, "name": "Python", "services": services},
        ) as res:
            assert res.status == 200


async def published_mount(base_url: str, runtime_id: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/runtimes/{runtime_id}/services/http-1"
        ) as res:
            assert res.status == 200
            return (await res.json())["__hkpMount"]


@pytest.mark.asyncio
async def test_reprovisioning_keeps_the_published_endpoint(servers):
    # Not a divergence any more: the runtime is still rebuilt, but an address is
    # derived from what identifies the mount rather than drawn when it is
    # claimed, so the new registration lands on the one already handed out.
    _server, base_url = await servers()
    await provision(base_url, "rt-1", [ENDPOINT_SERVICE])
    first = await published_mount(base_url, "rt-1")

    await provision(base_url, "rt-1", [ENDPOINT_SERVICE])
    second = await published_mount(base_url, "rt-1")

    assert second == first

    # And it is the rebuilt service answering there, not a stale record.
    async with aiohttp.ClientSession() as session:
        async with session.get(first) as res:
            assert res.status == 200


@pytest.mark.asyncio
async def test_reprovisioning_discards_accumulated_service_state(servers):
    # Divergence: on hkp-node the configured value survives.
    _server, base_url = await servers()
    await provision(
        base_url, "rt-1", [{"uuid": "mon-1", "serviceId": MONITOR_DESCRIPTOR.service_id}]
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes/rt-1/services/mon-1", json={"logToConsole": True}
        ) as res:
            assert res.status == 200

    await provision(
        base_url, "rt-1", [{"uuid": "mon-1", "serviceId": MONITOR_DESCRIPTOR.service_id}]
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/runtimes/rt-1/services/mon-1") as res:
            assert (await res.json())["logToConsole"] is False
