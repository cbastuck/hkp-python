"""Scopes: addressing into one, ending one, and what one keeps to itself.

Ported from hkp-node/tests/scoped-address.test.ts, stop-propagation.test.ts and
scope-slots.test.ts. A scope is a `sub-service` that can decline to pass its
result on and can hold state its children share — the two flows on one runtime
that a Stopper between them used to mark by convention.
"""
from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.hold import HOLD_DESCRIPTOR
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR
from hkp.services.map_service import MAP_DESCRIPTOR
from hkp.services.monitor import MONITOR_DESCRIPTOR
from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR


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


async def collect_notifications(ws_url: str, run, drain_seconds: float = 0.3):
    """Everything the runtime socket carried while `run` ran."""
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
                    try:
                        payload = json.loads(message.get("value", "null"))
                    except (TypeError, ValueError):
                        continue
                    seen.append(
                        {"instanceId": message.get("instanceId"), "payload": payload}
                    )

            task = asyncio.create_task(reader())
            await run()
            await asyncio.sleep(drain_seconds)
            task.cancel()
    return seen


async def post(session, url: str, body):
    async with session.post(url, json=body) as response:
        return response.status, await response.json()


async def get(session, url: str):
    async with session.get(url) as response:
        return response.status, await response.json()


def scope(uuid: str, pipeline, **state):
    return {
        "serviceId": SUB_SERVICE_DESCRIPTOR.service_id,
        "uuid": uuid,
        "state": {"pipeline": pipeline, **state},
    }


MARK = {
    "serviceId": MAP_DESCRIPTOR.service_id,
    "uuid": "mark",
    "state": {"mode": "replace", "template": {"mark": True}},
}


@pytest.mark.asyncio
async def test_reads_and_configures_a_service_inside_a_scope(servers):
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        await post(
            session,
            f"{base}/runtimes",
            {
                "id": "rt-1",
                "name": "Python",
                "services": [
                    scope(
                        "outer",
                        [
                            {
                                "serviceId": HOLD_DESCRIPTOR.service_id,
                                "uuid": "hold-1",
                                "state": {"property": "triggerCount"},
                            },
                            scope("inner", [MARK]),
                        ],
                    )
                ],
            },
        )

        status, body = await get(
            session, f"{base}/runtimes/rt-1/services/outer.hold-1"
        )
        assert status == 200
        assert body["property"] == "triggerCount"

        # Two levels down, which is what makes an address a path rather than a
        # single hop into a container.
        status, body = await get(
            session, f"{base}/runtimes/rt-1/services/outer.inner.mark"
        )
        assert status == 200
        assert body["mode"] == "replace"

        status, _ = await post(
            session,
            f"{base}/runtimes/rt-1/services/outer.hold-1",
            {"property": "other"},
        )
        assert status == 200
        _, body = await get(session, f"{base}/runtimes/rt-1/services/outer.hold-1")
        assert body["property"] == "other"

        # A partial address is a miss, not a match.
        async with session.get(
            f"{base}/runtimes/rt-1/services/outer.nope"
        ) as response:
            assert response.status == 404


@pytest.mark.asyncio
async def test_enters_a_scope_at_one_of_its_services(servers):
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        await post(
            session,
            f"{base}/runtimes",
            {
                "id": "rt-1",
                "name": "Python",
                "services": [scope("outer", [scope("inner", [MARK])])],
            },
        )

        status, body = await post(
            session,
            f"{base}/runtimes/rt-1/services/outer.inner/process",
            {"ignored": True},
        )
        assert status == 200
        assert body == {"mark": True}


@pytest.mark.asyncio
async def test_reports_a_nested_service_under_its_address_at_every_depth(servers):
    # A Monitor because it reports on every call and passes its input on; a Map
    # only answers, so watching one says nothing about whether it ran.
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [
                    scope(
                        "outer",
                        [
                            scope(
                                "inner",
                                [
                                    {
                                        "serviceId": MONITOR_DESCRIPTOR.service_id,
                                        "uuid": "witness",
                                        "state": {},
                                    }
                                ],
                            )
                        ],
                    )
                ],
            },
        ) as response:
            assert response.status == 200
            ws_url = (await response.json())["runtimes"][0]["outputUrl"]

    async def drive():
        async with aiohttp.ClientSession() as session:
            await post(session, f"{base}/runtimes/rt-1", {"go": 1})

    seen = await collect_notifications(ws_url, drive)
    assert any(entry["instanceId"] == "outer.inner.witness" for entry in seen), seen


@pytest.mark.asyncio
async def test_a_scope_that_stops_propagation(servers):
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        await post(
            session,
            f"{base}/runtimes",
            {
                "id": "rt-1",
                "name": "Python",
                "services": [
                    scope("ends", [MARK], stopPropagation=True),
                    {
                        "serviceId": HOLD_DESCRIPTOR.service_id,
                        "uuid": "after",
                        "state": {"slot": "seen", "op": "write"},
                    },
                ],
            },
        )

        status, body = await post(session, f"{base}/runtimes/rt-1", {"go": 1})
        assert status == 200
        assert body is None

        # The services after it did not run, which is what the Stopper between
        # two flows used to say about the gap rather than about either side.
        _, after = await get(session, f"{base}/runtimes/rt-1/services/after")
        assert after["writeCount"] == 0


@pytest.mark.asyncio
async def test_absent_stop_propagation_passes_the_result_on(servers):
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        await post(
            session,
            f"{base}/runtimes",
            {
                "id": "rt-1",
                "name": "Python",
                "services": [scope("open", [MARK])],
            },
        )

        # The default is the pipeline a board already has.
        status, body = await post(session, f"{base}/runtimes/rt-1", {"go": 1})
        assert status == 200
        assert body == {"mark": True}

        _, state = await get(session, f"{base}/runtimes/rt-1/services/open")
        assert state["stopPropagation"] is False
        assert state["scope"] == {"slots": "own"}


@pytest.mark.asyncio
async def test_what_a_scope_keeps_to_itself(servers):
    _server, base = await servers()

    def board(scope_state):
        return {
            "id": "rt-1",
            "name": "Python",
            "services": [
                scope(
                    "writes",
                    [
                        {
                            "serviceId": HOLD_DESCRIPTOR.service_id,
                            "uuid": "writer",
                            "state": {"slot": "shared", "op": "write"},
                        }
                    ],
                    **scope_state,
                ),
                {
                    "serviceId": HOLD_DESCRIPTOR.service_id,
                    "uuid": "reader",
                    "state": {"slot": "shared", "op": "read"},
                },
            ],
        }

    async with aiohttp.ClientSession() as session:
        await post(session, f"{base}/runtimes", board({}))
        await post(session, f"{base}/runtimes/rt-1", {"value": 7})
        # The name is the same on both sides and still reaches a different cell.
        _, reader = await get(session, f"{base}/runtimes/rt-1/services/reader")
        assert reader["held"] is None

        async with session.delete(f"{base}/runtimes/rt-1"):
            pass

        await post(session, f"{base}/runtimes", board({"scope": {"slots": "inherit"}}))
        await post(session, f"{base}/runtimes/rt-1", {"value": 7})
        _, reader = await get(session, f"{base}/runtimes/rt-1/services/reader")
        assert reader["held"] == {"value": 7}


def endpoint(uuid: str, mount_name: str | None = None):
    """An endpoint that answers with whatever the board last handed it.

    ``bypass: False`` is spelled out because an endpoint defaults to bypassed,
    and a bypassed one holds no address.
    """
    state = {
        "bypass": False,
        "onProcess": [
            {
                "serviceId": HOLD_DESCRIPTOR.service_id,
                "instanceId": "keep",
                "state": {"slot": "document", "op": "write"},
            }
        ],
        "onRequest": [
            {
                "serviceId": HOLD_DESCRIPTOR.service_id,
                "instanceId": "serve",
                "state": {"slot": "document", "op": "read"},
            }
        ],
    }
    if mount_name:
        state["mountName"] = mount_name
    return {
        "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
        "uuid": uuid,
        "state": state,
    }


@pytest.mark.asyncio
async def test_an_endpoint_inside_a_scope_is_given_an_address(servers):
    # A nested runtime has no server of its own, so until mounts were delegated
    # the way secrets and slots are, an endpoint inside a sub-pipeline
    # published no address at all — which made a scope something a board could
    # not put an endpoint in.
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        await post(
            session,
            f"{base}/runtimes",
            {
                "id": "rt-1",
                "name": "Python",
                "services": [
                    scope(
                        "serve-scope",
                        [
                            {
                                "serviceId": MAP_DESCRIPTOR.service_id,
                                "uuid": "body",
                                "state": {
                                    "mode": "replace",
                                    "template": {
                                        "meta": {"status": 200},
                                        "body": "from a scope",
                                    },
                                },
                            },
                            endpoint("serve", "in-scope"),
                        ],
                    )
                ],
            },
        )

        _, body = await get(
            session, f"{base}/runtimes/rt-1/services/serve-scope.serve"
        )
        url = body["__hkpMount"]
        assert "/hosted/" in url

        # Drive the board so the endpoint is handed a document to keep.
        await post(session, f"{base}/runtimes/rt-1", {"go": 1})

        async with session.get(url) as response:
            assert await response.text() == "from a scope"


@pytest.mark.asyncio
async def test_two_copies_of_one_scope_get_different_addresses(servers):
    # The name a mount is derived from falls back to the scoped address, so a
    # pipeline used twice does not derive one address between the two copies
    # and let either take the other's callers.
    _server, base = await servers()
    async with aiohttp.ClientSession() as session:
        await post(
            session,
            f"{base}/runtimes",
            {
                "id": "rt-1",
                "name": "Python",
                "services": [
                    scope("left", [endpoint("serve")]),
                    scope("right", [endpoint("serve")]),
                ],
            },
        )

        _, first = await get(session, f"{base}/runtimes/rt-1/services/left.serve")
        _, second = await get(session, f"{base}/runtimes/rt-1/services/right.serve")
        assert first["__hkpMount"]
        assert second["__hkpMount"]
        assert first["__hkpMount"] != second["__hkpMount"]
