"""Asking one service to do its job.

Ported from hkp-node/tests/process-at.test.ts.

Distinct from configuring it, and distinct from running the whole chain: a
facade button needs to say "you, with this payload, now". Before this the only
verb it had was configure, so anything it needed to cause had to be smuggled in
as a config field a service read as a command.

The service named here runs. `process_from` deliberately skips the service it
names — it means "carry on behind me" — and that difference is the whole reason
this is a separate entry point rather than a flag on that one.
"""
from __future__ import annotations

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.map_service import MAP_DESCRIPTOR


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(services):
        server = create_runtime_server({"external_host": "127.0.0.1"})
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        base_url = address["base_url"]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/runtimes",
                json={
                    "id": "rt-1",
                    "name": "Python",
                    "boardName": "Board",
                    "services": services,
                },
            ) as response:
                assert response.status == 200
        return base_url

    yield start
    for server in started:
        await server.stop()


def tag(uuid: str, value: str) -> dict:
    return {
        "serviceId": MAP_DESCRIPTOR.service_id,
        "uuid": uuid,
        "state": {"mode": "add", "template": {uuid: value}},
    }


async def process_at(base_url: str, uuid: str, payload: dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes/rt-1/services/{uuid}/process", json=payload
        ) as response:
            return response.status, (
                await response.json() if response.status == 200 else None
            )


@pytest.mark.asyncio
async def test_runs_the_service_it_names(servers):
    base_url = await servers([tag("first", "ran"), tag("second", "ran")])

    status, body = await process_at(base_url, "first", {"payload": True})

    assert status == 200
    assert body["first"] == "ran"


@pytest.mark.asyncio
async def test_does_not_run_what_comes_before_it(servers):
    base_url = await servers([tag("first", "ran"), tag("second", "ran")])

    status, body = await process_at(base_url, "second", {"payload": True})

    assert status == 200
    assert "first" not in body
    assert body["second"] == "ran"


@pytest.mark.asyncio
async def test_carries_on_through_the_services_after_it(servers):
    base_url = await servers([tag("first", "ran"), tag("second", "ran")])

    _, body = await process_at(base_url, "first", {})

    assert body["second"] == "ran"


@pytest.mark.asyncio
async def test_says_so_when_there_is_no_such_service(servers):
    base_url = await servers([tag("first", "ran")])

    status, _ = await process_at(base_url, "absent", {})

    assert status == 404
