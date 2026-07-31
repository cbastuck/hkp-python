"""Service endpoint (mount) tests — ported from hkp-node/tests/mounts.test.ts."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import aiohttp
import pytest
import pytest_asyncio

from hkp.auth import AuthenticatedUser
from hkp.server import create_runtime_server
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR

ALICE = "auth0|alice"
BOB = "auth0|bob"


class TwoPrincipalAuthenticator:
    """A bearer token is the sub it authenticates as; see test_tenancy."""

    def __init__(self, resolve_opaque_token: Callable[[str], Any]) -> None:
        self._resolve_opaque_token = resolve_opaque_token
        self._known = {ALICE, BOB}

    async def verify_token(self, token: str | None) -> AuthenticatedUser | None:
        if not token:
            return None
        opaque = self._resolve_opaque_token(token)
        if opaque:
            return opaque
        return AuthenticatedUser(sub=token) if token in self._known else None


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(options: dict[str, Any] | None = None):
        server = create_runtime_server(
            {"external_host": "127.0.0.1", **(options or {})}
        )
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


def auth(sub: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {sub}"}


HTTP_SERVICE = {
    "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
    "uuid": "http-1",
    "state": {"bypass": False, "mode": "process_on_session", "pipeline": []},
}


async def make_endpoint(
    session: aiohttp.ClientSession,
    base_url: str,
    runtime_id: str = "rt-1",
    headers: dict[str, str] | None = None,
) -> str:
    async with session.post(
        f"{base_url}/runtimes",
        headers=headers or {},
        json={"id": runtime_id, "name": "Node", "services": [HTTP_SERVICE]},
    ) as res:
        assert res.status == 200
    async with session.get(
        f"{base_url}/runtimes/{runtime_id}/services/http-1", headers=headers or {}
    ) as res:
        assert res.status == 200
        return (await res.json())["__hkpMount"]


@pytest.mark.asyncio
async def test_endpoint_is_served_without_a_token(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        endpoint = await make_endpoint(session, base_url)
        assert re.search(r"/hosted/[0-9a-f]{32}$", endpoint), endpoint

        # Mounts exist to be called by outside parties, so no Authorization.
        async with session.get(f"{endpoint}/hello?a=1") as res:
            assert res.status == 200
            received = await res.json()

        assert received["meta"] == {
            "method": "GET",
            "path": "/hello",
            "query": {"a": "1"},
        }
        # No body at all, so neither representation is carried.
        assert "binary" not in received
        assert "body" not in received


@pytest.mark.asyncio
async def test_two_tenants_get_separate_endpoints(servers):
    _server, base_url = await servers(
        {"build_authenticator": TwoPrincipalAuthenticator}
    )
    async with aiohttp.ClientSession() as session:
        alice_url = await make_endpoint(session, base_url, "node", auth(ALICE))
        bob_url = await make_endpoint(session, base_url, "node", auth(BOB))

        assert alice_url != bob_url
        # Both live: neither tenant's endpoint displaced the other's, which is
        # exactly what a shared port would have done.
        for url in (alice_url, bob_url):
            async with session.get(url) as res:
                assert res.status == 200


@pytest.mark.asyncio
async def test_endpoint_stops_when_its_runtime_is_removed(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        endpoint = await make_endpoint(session, base_url)
        async with session.get(f"{endpoint}/hello") as res:
            assert res.status == 200

        async with session.delete(f"{base_url}/runtimes/rt-1") as res:
            assert res.status == 200

        # The endpoint must not outlive the runtime that published it.
        async with session.get(f"{endpoint}/hello") as res:
            assert res.status == 404


@pytest.mark.asyncio
async def test_endpoint_released_on_bypass(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        endpoint = await make_endpoint(session, base_url)

        async with session.post(
            f"{base_url}/runtimes/rt-1/services/http-1", json={"bypass": True}
        ) as res:
            assert res.status == 200
            assert (await res.json())["__hkpMount"] == ""

        async with session.get(f"{endpoint}/hello") as res:
            assert res.status == 404


@pytest.mark.asyncio
async def test_body_is_decoded_by_content_type(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        endpoint = await make_endpoint(session, base_url)

        async def post(content_type: str, payload: bytes) -> dict[str, Any]:
            async with session.post(
                f"{endpoint}/x",
                data=payload,
                headers={"Content-Type": content_type},
            ) as res:
                assert res.status == 200
                return await res.json()

        # Charset parameters must not defeat the match.
        decoded = await post("application/json; charset=utf-8", b'{"a": 1}')
        assert decoded["body"] == {"a": 1}
        # The raw bytes would only restate the decoded value at twice the size.
        assert "binary" not in decoded

        assert (await post("application/vnd.api+json", b'{"a": 2}'))["body"] == {"a": 2}
        assert (await post("text/plain", b"hello"))["body"] == "hello"
        assert (await post("application/x-www-form-urlencoded", b"a=1&b=two"))[
            "body"
        ] == {"a": "1", "b": "two"}

        # Not textual: only the raw bytes, no decoded body to be wrong about.
        raw = await post("application/octet-stream", b"abc")
        assert "body" not in raw
        assert raw["binary"] is not None

        # Malformed input leaves the bytes to inspect rather than failing the
        # request — a public endpoint takes whatever it is given.
        broken = await post("application/json", b"{not json")
        assert "body" not in broken
        assert broken["binary"] is not None


@pytest.mark.asyncio
async def test_filename_is_surfaced_separately_from_path(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        endpoint = await make_endpoint(session, base_url)
        async with session.post(
            f"{endpoint}/upload",
            data=b'{"hello": "world"}',
            headers={
                "Content-Type": "application/json",
                "Content-Disposition": 'attachment; filename="notes.txt"',
            },
        ) as res:
            received = await res.json()

        assert received["meta"]["path"] == "/upload"
        assert received["meta"]["filename"] == "notes.txt"
        assert received["body"] == {"hello": "world"}


@pytest.mark.asyncio
async def test_oversized_body_is_refused(servers):
    _server, base_url = await servers({"quotas": {"max_request_body_bytes": 64}})
    async with aiohttp.ClientSession() as session:
        endpoint = await make_endpoint(session, base_url)

        # The endpoint takes no token, so an unbounded read would be available
        # to anyone holding the URL.
        async with session.post(f"{endpoint}/upload", data=b"x" * 1024) as res:
            assert res.status == 413


@pytest.mark.asyncio
async def test_unknown_mount_is_404(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/hosted/{'0' * 32}/x") as res:
            assert res.status == 404
