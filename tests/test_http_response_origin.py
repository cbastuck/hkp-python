"""What forms the HTTP response of ``http-server-subservices``.

The same three rows are pinned in every runtime that implements this service
(hkp-node, hkp-python, hkp-rt), because a board written against one must behave
the same on the others:

    | nested pipeline | service after the server | answer comes from |
    |-----------------|--------------------------|-------------------|
    | yes             | no                       | nested pipeline   |
    | yes             | yes                      | nested pipeline   |
    | no              | yes                      | the outer runtime |

The middle row is the point: configuring a nested pipeline declares a handler,
and services added behind the server must not silently rewrite what an external
caller receives.

What declares a handler is ``onRequest`` — a pipeline for requests. The rows
above say "nested pipeline" because a board that names no entry points has only
one, and the legacy ``mode`` decides which sides enter it. An endpoint that
declares ``onProcess`` alone has a pipeline and no handler, which is the fourth
row and the one the older shape could not express.

Ported from hkp-node/tests/http-response-origin.test.ts.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR
from hkp.services.map_service import MAP_DESCRIPTOR

NESTED = {
    "instanceId": "inner",
    "serviceId": MAP_DESCRIPTOR.service_id,
    "serviceName": "Inner",
    "state": {"mode": "replace", "template": {"from": "subservice"}},
}

OUTER = {
    "serviceId": MAP_DESCRIPTOR.service_id,
    "uuid": "outer-1",
    "state": {"mode": "replace", "template": {"from": "outer"}},
}


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


async def endpoint_with(
    servers,
    *,
    subservices: bool,
    after: list[dict[str, Any]],
    entries: dict[str, Any] | None = None,
) -> str:
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Node",
                "services": [
                    {
                        "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                        "uuid": "http-1",
                        "state": entries
                        or {
                            "bypass": False,
                            "mode": "process_on_session",
                            "pipeline": [NESTED] if subservices else [],
                        },
                    },
                    *after,
                ],
            },
        ) as res:
            assert res.status == 200
        async with session.get(
            f"{base_url}/runtimes/rt-1/services/http-1"
        ) as res:
            assert res.status == 200
            return (await res.json())["__hkpMount"]


async def get_json(url: str) -> Any:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as res:
            return await res.json()


@pytest.mark.asyncio
async def test_nested_pipeline_answers_when_it_is_the_only_handler(servers):
    url = await endpoint_with(servers, subservices=True, after=[])
    assert await get_json(url) == {"from": "subservice"}


@pytest.mark.asyncio
async def test_nested_pipeline_still_answers_when_services_follow(servers):
    # The outer service runs — it is a side effect of having served a request —
    # but it does not get to rewrite the answer.
    url = await endpoint_with(servers, subservices=True, after=[OUTER])
    assert await get_json(url) == {"from": "subservice"}


@pytest.mark.asyncio
async def test_outer_runtime_answers_without_a_nested_pipeline(servers):
    # Inversion of control: with no handler configured, the rest of the board is
    # the handler.
    url = await endpoint_with(servers, subservices=False, after=[OUTER])
    assert await get_json(url) == {"from": "outer"}


@pytest.mark.asyncio
async def test_a_pipeline_for_passes_alone_leaves_the_chain_answering(servers):
    # The distinction a single unnamed pipeline could not draw: this endpoint
    # has something to run, but nothing to run *for a request*, so the board
    # still answers. Declared the old way the same board would have served
    # whatever its one pipeline returned, because having one at all decided.
    url = await endpoint_with(
        servers,
        subservices=False,
        after=[OUTER],
        entries={"bypass": False, "onProcess": [NESTED]},
    )
    assert await get_json(url) == {"from": "outer"}


@pytest.mark.asyncio
async def test_a_pipeline_for_requests_answers_with_one_for_passes_beside_it(servers):
    url = await endpoint_with(
        servers,
        subservices=False,
        after=[OUTER],
        entries={
            "bypass": False,
            "onProcess": [{**NESTED, "instanceId": "on-pass"}],
            "onRequest": [{**NESTED, "instanceId": "on-request"}],
        },
    )
    assert await get_json(url) == {"from": "subservice"}
