"""Hold: the latest value one side produced, replayed to the other.

Ported from hkp-node/tests/hold.test.ts.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.hold import HOLD_DESCRIPTOR, HoldService
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR
from hkp.services.map_service import MAP_DESCRIPTOR
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


def make_hold(state: dict[str, Any] | None = None) -> HoldService:
    return HoldService(
        ServiceConfiguration(service_id="hold", uuid="hold-1", state=state)
    )


def noop(_payload: Any, _instance_id: str | None = None) -> None:
    return None


# What an http-server request arrives as: no producer property in sight.
REQUEST = {"meta": {"method": "GET", "path": "/", "query": {}}}


def test_holds_the_named_property_and_emits_it_under_the_same_name():
    hold = make_hold({"property": "triggerCount"})
    assert hold.process({"triggerCount": 1}, noop) == {"triggerCount": 1}
    state = hold.get_state()
    assert state["held"] == 1
    assert state["writeCount"] == 1


def test_emits_the_same_shape_whichever_side_calls():
    hold = make_hold({"property": "triggerCount"})
    written = hold.process({"triggerCount": 4}, noop)
    read = hold.process(REQUEST, noop)
    # What the services after Hold see does not say which side called; only the
    # counts, which nothing downstream sees, tell them apart.
    assert read == written
    assert hold.get_state()["readCount"] == 1
    assert hold.get_state()["writeCount"] == 1


def test_replays_without_consuming():
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 4}, noop)
    assert hold.process(REQUEST, noop) == {"triggerCount": 4}
    assert hold.process(REQUEST, noop) == {"triggerCount": 4}


def test_keeps_the_newest_value_written():
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 1}, noop)
    hold.process({"triggerCount": 2}, noop)
    assert hold.process(REQUEST, noop) == {"triggerCount": 2}


def test_drops_everything_but_the_held_property():
    # A producer's other fields are not part of what is held.
    hold = make_hold({"property": "triggerCount"})
    assert hold.process({"triggerCount": 5, "note": "ignored"}, noop) == {
        "triggerCount": 5
    }


def test_stops_while_nothing_is_held():
    hold = make_hold({"property": "triggerCount"})
    assert hold.process(REQUEST, noop) is None
    assert hold.get_state()["held"] is None


def test_reads_on_inputs_that_cannot_carry_a_property():
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 6}, noop)
    assert hold.process("a string", noop) == {"triggerCount": 6}
    assert hold.process([1, 2, 3], noop) == {"triggerCount": 6}


def test_reads_on_a_none_value_which_is_nothing_to_hold():
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 2}, noop)
    assert hold.process({"triggerCount": None}, noop) == {"triggerCount": 2}
    assert hold.get_state()["readCount"] == 1
    assert hold.get_state()["writeCount"] == 1


def test_passes_input_through_while_no_property_is_configured():
    hold = make_hold()
    assert hold.process(REQUEST, noop) == REQUEST
    assert hold.get_state()["held"] is None


def test_forgets_the_held_value_and_the_counts_on_clear():
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 3}, noop)
    hold.process(REQUEST, noop)
    hold.configure({"action": "clear"})
    # The counts described the value that was just discarded.
    state = hold.get_state()
    assert state["held"] is None
    assert state["readCount"] == 0
    assert state["writeCount"] == 0
    assert hold.process(REQUEST, noop) is None


def test_forgets_the_held_value_and_the_counts_when_the_property_changes():
    # What was held belonged to the old property name.
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 3}, noop)
    hold.configure({"property": "counter"})
    assert hold.get_state()["held"] is None
    assert hold.get_state()["writeCount"] == 0
    assert hold.process(REQUEST, noop) is None


def test_keeps_the_counts_when_the_property_is_configured_to_what_it_already_is():
    hold = make_hold({"property": "triggerCount"})
    hold.process({"triggerCount": 3}, noop)
    hold.configure({"property": "triggerCount"})
    assert hold.get_state()["held"] == 3
    assert hold.get_state()["writeCount"] == 1


def test_describes_a_held_value_that_cannot_travel_as_json():
    hold = make_hold({"property": "payload"})
    unserializable = object()
    hold.process({"payload": unserializable}, noop)
    assert hold.get_state()["held"] == "[object]"
    # The value itself is untouched — only the reported state is a description.
    assert hold.process(REQUEST, noop) == {"payload": unserializable}


@pytest.mark.asyncio
async def test_serves_a_nested_timers_latest_tick_to_callers(servers):
    server, base_url = await servers()

    sub_pipeline = [
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
        {
            "serviceId": MAP_DESCRIPTOR.service_id,
            "uuid": "map-1",
            "state": {
                "mode": "replace",
                "template": {"=": "'tick ' + params.triggerCount"},
            },
        },
    ]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [
                    {
                        "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                        "uuid": "http-1",
                        "state": {
                            "bypass": False,
                            "mode": "process_on_session",
                            "pipeline": sub_pipeline,
                        },
                    }
                ],
            },
        ) as response:
            assert response.status == 200

        async with session.get(f"{base_url}/runtimes/rt-1/services/http-1") as response:
            mount_url = (await response.json())["__hkpMount"]

        # Before the first tick there is nothing held, and the read stops.
        async with session.get(mount_url) as response:
            assert await response.json() is None

        # One tick, now, rather than waiting out the period.
        async with session.post(
            f"{base_url}/runtimes/rt-1/services/http-1",
            json={
                "configureService": {
                    "instanceId": "timer-1",
                    "state": {"immediate": True, "start": True},
                }
            },
        ) as response:
            assert response.status == 200

        import asyncio

        await asyncio.sleep(0.15)

        async with session.get(mount_url) as response:
            assert await response.json() == "tick 1"
