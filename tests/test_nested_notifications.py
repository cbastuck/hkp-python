"""What a nested pipeline reports, and what it leaves behind when it goes.

Ported from hkp-node/tests/sub-service.test.ts.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.hold import HOLD_DESCRIPTOR
from hkp.services.http_server import (
    HTTP_SERVER_SUBSERVICES_DESCRIPTOR,
    HttpServerSubservicesService,
)
from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
from hkp.services.timer import TIMER_DESCRIPTOR
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


NESTED_PIPELINE = [
    {
        "serviceId": TIMER_DESCRIPTOR.service_id,
        "uuid": "timer-1",
        "state": {"periodic": True, "periodicValue": 60, "periodicUnit": "s"},
    },
    {
        "serviceId": HOLD_DESCRIPTOR.service_id,
        "uuid": "hold-1",
        "state": {"property": "triggerCount"},
    },
]


async def collect_notifications(
    ws_url: str,
    run,
    drain_seconds: float = 0.3,
) -> list[dict[str, Any]]:
    """Everything the runtime socket carried while `run` ran.

    Use it to count deliveries — one channel too many shows up here as
    duplicates, one too few as nothing at all.
    """
    seen: list[dict[str, Any]] = []

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
                    try:
                        payload = json.loads(message.get("value", "null"))
                    except (TypeError, ValueError):
                        continue
                    seen.append(
                        {"instanceId": message.get("instanceId"), "payload": payload}
                    )

            reader_task = asyncio.create_task(reader())
            await run()
            await asyncio.sleep(drain_seconds)
            reader_task.cancel()

    return seen


def flow_count(seen: list[dict[str, Any]], instance_id: str, state: str) -> int:
    """How often `instance_id` reported the given flow state."""
    return len(
        [
            entry
            for entry in seen
            if entry["instanceId"] == instance_id
            and isinstance(entry["payload"], dict)
            and (entry["payload"].get("__internal") or {}).get("state") == state
        ]
    )


async def _create_board(session, base_url: str) -> str:
    async with session.post(
        f"{base_url}/runtimes",
        json={
            "id": "rt-1",
            "name": "Python",
            "services": [
                {
                    "serviceId": SUB_SERVICE_DESCRIPTOR.service_id,
                    "uuid": "sub-1",
                    "state": {"pipeline": NESTED_PIPELINE},
                }
            ],
        },
    ) as response:
        assert response.status == 200
        body = await response.json()
    return body["runtimes"][0]["outputUrl"]


@pytest.mark.asyncio
async def test_reports_a_nested_services_state_to_an_attached_board(servers):
    # Regression: a nested runtime has no notification targets of its own, and
    # SubService had no host to carry its pipeline's notifications out to. An
    # attached board saw nested services report nothing at all.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await _create_board(session, base_url)

    async def drive():
        async with aiohttp.ClientSession() as session:
            # One tick, now, rather than waiting out the period.
            async with session.post(
                f"{base_url}/runtimes/rt-1/services/sub-1",
                json={
                    "configureService": {
                        "instanceId": "timer-1",
                        "state": {"immediate": True, "start": True},
                    }
                },
            ) as response:
                assert response.status == 200

    seen = await collect_notifications(ws_url, drive)

    holds = [
        entry["payload"]
        for entry in seen
        if entry["instanceId"] == "hold-1"
        and isinstance(entry["payload"], dict)
        and entry["payload"].get("writeCount") == 1
    ]
    # The tick reached the Hold behind it, and the Hold's own report got out.
    assert holds, f"no hold-1 state reached the board; saw {seen}"
    assert holds[-1]["held"] == 1


@pytest.mark.asyncio
async def test_reports_each_nested_notification_exactly_once(servers):
    # Zero means the nested runtime's reports are being dropped; two means they
    # travel by the host *and* by the callback passed into the pipeline. The
    # flow (`__internal`) reports are what double, since a service's own state
    # goes by the host alone.
    _server, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        ws_url = await _create_board(session, base_url)

    async def drive():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/runtimes/rt-1", json={"triggerCount": 1}
            ) as response:
                assert response.status == 200

    seen = await collect_notifications(ws_url, drive)

    assert flow_count(seen, "hold-1", "call-process") == 1
    assert flow_count(seen, "hold-1", "call-process-finished") == 1


class _RecordingService:
    """A nested service that says when it was destroyed."""

    service_id = "stub"
    service_name = "Stub"
    version = None
    capabilities: list[str] = []

    def __init__(self, config: ServiceConfiguration, destroyed: list[str]) -> None:
        self.uuid = config.uuid
        self._destroyed = destroyed

    def configure(self, _config):
        return {}

    def get_state(self):
        return {}

    def process(self, input, _notify):
        return input

    def destroy(self) -> None:
        self._destroyed.append(self.uuid)


def _recording_create_service(destroyed: list[str]):
    return lambda config: _RecordingService(config, destroyed)


def _pipeline_of(uuid: str) -> list[dict[str, Any]]:
    return [{"serviceId": "stub", "uuid": uuid}]


# Nested teardown is asserted on destroy() reaching the nested services rather
# than on what stops arriving at the board: a leaked pipeline is unreachable, so
# it goes quiet either way. Silence proves nothing; the call does.
def test_destroys_a_sub_services_nested_services_with_it():
    destroyed: list[str] = []
    sub = SubService(
        ServiceConfiguration(
            service_id=SUB_SERVICE_DESCRIPTOR.service_id,
            uuid="sub-1",
            state={"pipeline": _pipeline_of("nested-1")},
        ),
        _recording_create_service(destroyed),
    )

    sub.destroy()
    assert destroyed == ["nested-1"]


def test_destroys_the_pipeline_a_sub_service_replaces():
    # The frequent path: every reconfiguration builds a new pipeline, and the
    # old one keeps its timers and sockets unless it is torn down.
    destroyed: list[str] = []
    sub = SubService(
        ServiceConfiguration(
            service_id=SUB_SERVICE_DESCRIPTOR.service_id,
            uuid="sub-1",
            state={"pipeline": _pipeline_of("nested-1")},
        ),
        _recording_create_service(destroyed),
    )

    sub.configure({"pipeline": _pipeline_of("nested-2")})
    assert destroyed == ["nested-1"]

    sub.destroy()
    assert destroyed == ["nested-1", "nested-2"]


def test_destroys_an_http_servers_nested_services_with_it():
    destroyed: list[str] = []
    endpoint = HttpServerSubservicesService(
        ServiceConfiguration(
            service_id=HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
            uuid="http-1",
            state={"pipeline": _pipeline_of("nested-1")},
        ),
        _recording_create_service(destroyed),
    )

    endpoint.configure({"pipeline": _pipeline_of("nested-2")})
    assert destroyed == ["nested-1"]

    endpoint.destroy()
    assert destroyed == ["nested-1", "nested-2"]
