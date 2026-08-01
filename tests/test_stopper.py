"""Stopper, and the service ids a board may use to reach a service.

Ported from hkp-node/tests/stopper.test.ts.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.map_service import MAP_DESCRIPTOR
from hkp.services.stopper import STOPPER_DESCRIPTOR, StopperService
from hkp.services.timer import TIMER_DESCRIPTOR, TIMER_LEGACY_SERVICE_ID
from hkp.types import ServiceConfiguration


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


def make_stopper(state: dict[str, Any] | None = None) -> StopperService:
    return StopperService(
        ServiceConfiguration(service_id="stopper", uuid="stop-1", state=state)
    )


def test_returns_none_so_the_runtime_stops_there():
    assert make_stopper().process({"anything": True}, lambda _n: None) is None


def test_passes_input_through_when_bypassed():
    # Opens the chain back up without moving services around.
    stopper = make_stopper({"bypass": True})
    assert stopper.process({"kept": True}, lambda _n: None) == {"kept": True}


@pytest.mark.asyncio
async def test_stops_the_services_that_follow_it_in_the_runtime(servers):
    _server, base_url = await servers()

    # The map would rewrite anything that reached it, so the result says plainly
    # whether it was called.
    map_service = {
        "serviceId": MAP_DESCRIPTOR.service_id,
        "uuid": "map-1",
        "state": {"mode": "replace", "template": {"reached": True}},
    }

    async def with_services(runtime_id: str, services: list[dict[str, Any]]) -> Any:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/runtimes",
                json={"id": runtime_id, "name": "Python", "services": services},
            ) as res:
                assert res.status == 200
            async with session.post(
                f"{base_url}/runtimes/{runtime_id}", json={"value": 42}
            ) as res:
                assert res.status == 200
                return await res.json()

    # Control: without the stopper the map runs and rewrites the value.
    assert await with_services("rt-control", [map_service]) == {"reached": True}

    assert (
        await with_services(
            "rt-stopped",
            [
                {"serviceId": STOPPER_DESCRIPTOR.service_id, "uuid": "stop-1"},
                map_service,
            ],
        )
        is None
    )


@pytest.mark.asyncio
async def test_timer_is_reachable_under_both_its_ids(servers):
    # The canonical id matches hkp-node and hkp-rt so one board runs on any of
    # them; the older id still resolves so boards saved against it keep loading.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [
                    {"serviceId": TIMER_DESCRIPTOR.service_id, "uuid": "timer-1"},
                    {"serviceId": TIMER_LEGACY_SERVICE_ID, "uuid": "timer-2"},
                ],
            },
        ) as res:
            assert res.status == 200

        async with session.get(f"{base_url}/runtimes/rt-1/services") as res:
            assert res.status == 200
            services = await res.json()

    # Both report the canonical id: the alias is a way in, not a second service.
    assert [svc["serviceId"] for svc in services] == ["timer", "timer"]


@pytest.mark.asyncio
async def test_the_registry_advertises_the_timer_once(servers):
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={"id": "rt-1", "name": "Python", "services": []},
        ) as res:
            registry = (await res.json())["registry"]

    assert [entry["serviceId"] for entry in registry].count("timer") == 1
    assert TIMER_LEGACY_SERVICE_ID not in [entry["serviceId"] for entry in registry]
