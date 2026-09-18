"""What an endpoint may answer with, beside JSON.

The request and the response are the same envelope read in two directions, and
the pair of tests that matter are the two ends of that: that a handler can say
what it is answering with, and that a handler which simply passed its input
through still answers the way it always did.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.http_server import (
    HTTP_SERVER_SUBSERVICES_DESCRIPTOR,
    _requested_range,
    _to_answer,
)


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(options: dict[str, Any] | None = None):
        server = create_runtime_server({"external_host": "127.0.0.1", **(options or {})})
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


async def _mount_answering(session, base_url, template: dict[str, Any]) -> str:
    """An endpoint whose whole pipeline is a Map returning `template`."""
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
                        "pipeline": [
                            {
                                "instanceId": "answer",
                                "serviceId": "map",
                                "serviceName": "Answer",
                                "state": {"mode": "replace", "template": template},
                            }
                        ],
                    },
                }
            ],
        },
    ) as res:
        assert res.status == 200
    async with session.get(f"{base_url}/runtimes/rt-1/services/http-1") as res:
        return (await res.json())["__hkpMount"]


@pytest.mark.asyncio
async def test_a_handler_can_say_what_it_is_answering_with(servers):
    _, base_url = await servers()
    feed = '<?xml version="1.0"?><rss version="2.0"></rss>'

    async with aiohttp.ClientSession() as session:
        mount = await _mount_answering(
            session,
            base_url,
            {
                "meta": {"status": 200, "contentType": "application/rss+xml"},
                "body": feed,
            },
        )
        async with session.get(f"{mount}/feed.xml") as res:
            assert res.status == 200
            assert res.headers["Content-Type"].startswith("application/rss+xml")
            assert await res.text() == feed


@pytest.mark.asyncio
async def test_a_status_travels_too(servers):
    _, base_url = await servers()

    async with aiohttp.ClientSession() as session:
        mount = await _mount_answering(
            session,
            base_url,
            {"meta": {"status": 404}, "body": {"error": "no such episode"}},
        )
        async with session.get(f"{mount}/missing.mp3") as res:
            assert res.status == 404
            assert (await res.json())["error"] == "no such episode"


@pytest.mark.asyncio
async def test_a_request_passed_through_is_still_answered_as_json(servers):
    # A request envelope is the same shape as a response one; only a status
    # tells them apart. Without this, an echo would answer with the caller's
    # own content type and every board that had one would change behaviour.
    _, base_url = await servers()

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
                            "pipeline": [],
                        },
                    }
                ],
            },
        ) as res:
            assert res.status == 200
        async with session.get(f"{base_url}/runtimes/rt-1/services/http-1") as res:
            mount = (await res.json())["__hkpMount"]

        async with session.post(
            f"{mount}/echo", json={"hello": "world"}
        ) as res:
            assert res.status == 200
            assert res.headers["Content-Type"].startswith("application/json")
            echoed = await res.json()
            assert echoed["body"] == {"hello": "world"}


def test_bytes_are_answered_as_bytes():
    # The one value whose own type decides: JSON encoding a bytes object
    # produces nothing anybody wanted.
    status, headers, payload = _to_answer(b"\x00\x01\x02")

    assert status == 200
    assert headers["content-type"] == "application/octet-stream"
    assert payload == b"\x00\x01\x02"


def test_an_envelope_carrying_bytes_keeps_the_type_it_declared():
    status, headers, payload = _to_answer(
        {"meta": {"status": 200, "contentType": "audio/mpeg"}, "binary": b"\xff\xfb"}
    )

    assert (status, headers["content-type"], payload) == (200, "audio/mpeg", b"\xff\xfb")


def test_anything_without_a_status_is_json():
    status, headers, payload = _to_answer({"meta": {"contentType": "audio/mpeg"}, "body": "x"})

    assert status == 200
    assert headers["content-type"] == "application/json"
    assert payload == b'{"meta": {"contentType": "audio/mpeg"}, "body": "x"}'


@pytest.mark.parametrize(
    "header,length,expected",
    [
        ("bytes=0-9", 100, (0, 9)),
        # An open end means "to the end of what there is".
        ("bytes=90-", 100, (90, 99)),
        # A suffix range is the last n bytes, not the first.
        ("bytes=-10", 100, (90, 99)),
        # Clamped rather than refused: a player asking past the end gets what
        # there is.
        ("bytes=95-200", 100, (95, 99)),
        ("bytes=200-", 100, None),
        ("items=0-9", 100, None),
        (None, 100, None),
    ],
)
def test_a_range_is_read_the_way_a_player_writes_one(header, length, expected):
    assert _requested_range(header, length) == expected


@pytest.mark.asyncio
async def test_an_endpoint_answers_its_own_document(servers):
    # `process_on_data` serves what the board handed it. If the chain's tail
    # answered instead, an endpoint could only ever be the last service in its
    # runtime, and a runtime could publish exactly one document.
    _, base_url = await servers()

    def document(body: str) -> dict[str, Any]:
        return {
            "mode": "replace",
            "template": {
                "meta": {"status": 200, "contentType": "text/plain"},
                "body": body,
            },
        }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [
                    {"serviceId": "map", "uuid": "doc", "state": document("the document")},
                    {
                        "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                        "uuid": "http-1",
                        "state": {"bypass": False, "mode": "process_on_data", "pipeline": []},
                    },
                    # What a board does after serving — here something that would
                    # be a fine answer, if answers came from the chain's tail.
                    {"serviceId": "map", "uuid": "after", "state": document("something else")},
                ],
            },
        ) as res:
            assert res.status == 200

        async with session.post(
            f"{base_url}/runtimes/rt-1/services/doc/process", json={}
        ) as res:
            assert res.status == 200

        async with session.get(f"{base_url}/runtimes/rt-1/services/http-1") as res:
            mount = (await res.json())["__hkpMount"]

        async with session.get(mount) as res:
            assert res.status == 200
            assert await res.text() == "the document"
