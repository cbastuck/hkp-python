"""What happens when a runtime that already exists is provisioned again.

hkp-node reuses the running runtime, so a browser attaching to a
coordinator-managed board does not kill services the coordinator started, and
mounts keep the addresses already handed to their consumers
(hkp-node/tests/cloud-reprovision.test.ts).

**hkp-python rebuilds it instead.** These tests pin that difference rather than
endorse it: it is a known divergence, listed in TODO-CONSOLIDATION.md. Change
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
async def test_reprovisioning_replaces_the_runtime_and_its_mounts(servers):
    # Divergence: on hkp-node the address survives. Flip this to == when python
    # adopts reuse.
    _server, base_url = await servers()
    await provision(base_url, "rt-1", [ENDPOINT_SERVICE])
    first = await published_mount(base_url, "rt-1")

    await provision(base_url, "rt-1", [ENDPOINT_SERVICE])
    second = await published_mount(base_url, "rt-1")

    assert second != first

    # The address the first registration published no longer serves, so anything
    # already pointed at it — a coordinator's consumer, say — is stranded.
    async with aiohttp.ClientSession() as session:
        async with session.get(first) as res:
            assert res.status == 404
        async with session.get(second) as res:
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
