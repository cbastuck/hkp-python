"""Multi-tenancy tests — ported from hkp-node/tests/auth.test.ts.

Runtimes are namespaced by the authenticated ``sub``. These exercise that a
token can only ever reach its own namespace, using two principals resolved
offline so no JWKS endpoint is contacted.
"""
from __future__ import annotations

from typing import Any, Callable

import aiohttp
import pytest
import pytest_asyncio

from hkp.auth import AuthenticatedUser
from hkp.server import create_runtime_server

ALICE = "auth0|alice"
BOB = "auth0|bob"


class TwoPrincipalAuthenticator:
    """Stands in for JWKS verification: a bearer token is simply the sub it
    authenticates as. Session tokens are still resolved first through the
    resolver the server owns, so the delegation path stays under test rather
    than being stubbed out."""

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
        options = {
            "external_host": "127.0.0.1",
            "build_authenticator": TwoPrincipalAuthenticator,
            **(options or {}),
        }
        server = create_runtime_server(options)
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


def auth(sub: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {sub}"}


async def create_runtime_as(
    session: aiohttp.ClientSession,
    base_url: str,
    sub: str,
    runtime_id: str,
    service_uuid: str,
) -> None:
    """Create a runtime owned by ``sub`` carrying one monitor service."""
    async with session.post(
        f"{base_url}/runtimes",
        headers=auth(sub),
        json={
            "id": runtime_id,
            "name": "Node",
            "services": [{"serviceId": "monitor", "uuid": service_uuid}],
        },
    ) as res:
        assert res.status == 200


# ── Isolation ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_same_runtime_id_gives_each_user_their_own(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        # Boards ship stable, human-readable runtime ids, so this collision is
        # the normal case whenever two users load the same board.
        await create_runtime_as(session, base_url, ALICE, "node", "alice-monitor")
        await create_runtime_as(session, base_url, BOB, "node", "bob-monitor")

        for sub, expected in ((ALICE, "alice-monitor"), (BOB, "bob-monitor")):
            async with session.get(
                f"{base_url}/runtimes/node/services", headers=auth(sub)
            ) as res:
                assert res.status == 200
                assert [s["uuid"] for s in await res.json()] == [expected]


@pytest.mark.asyncio
async def test_list_returns_only_the_callers_runtimes(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "alice-rt", "m1")
        await create_runtime_as(session, base_url, BOB, "bob-rt", "m2")

        async with session.get(f"{base_url}/runtimes", headers=auth(ALICE)) as res:
            body = await res.json()
            assert [r["id"] for r in body["runtimes"]] == ["alice-rt"]
            # The registry describes the build, not the tenant.
            assert len(body["registry"]) > 0


@pytest.mark.asyncio
async def test_foreign_runtime_is_404_everywhere(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "alice-rt", "m1")

        # 404 rather than 403 — Bob must not be able to probe which ids exist.
        gets = [
            "/runtimes/alice-rt",
            "/runtimes/alice-rt/services",
            "/runtimes/alice-rt/services/m1",
            "/runtimes/alice-rt/services/m1/property/bypass",
        ]
        for path in gets:
            async with session.get(f"{base_url}{path}", headers=auth(BOB)) as res:
                assert res.status == 404, path

        async with session.post(
            f"{base_url}/runtimes/alice-rt", headers=auth(BOB), json={"hello": "world"}
        ) as res:
            assert res.status == 404
        async with session.post(
            f"{base_url}/runtimes/alice-rt/services/m1",
            headers=auth(BOB),
            json={"bypass": True},
        ) as res:
            assert res.status == 404
        async with session.post(
            f"{base_url}/runtimes/alice-rt/rearrange", headers=auth(BOB), json=["m1"]
        ) as res:
            assert res.status == 404
        async with session.post(
            f"{base_url}/runtimes/alice-rt/session-token", headers=auth(BOB)
        ) as res:
            assert res.status == 404
        async with session.delete(
            f"{base_url}/runtimes/alice-rt/services/m1", headers=auth(BOB)
        ) as res:
            assert res.status == 404

        # Alice's runtime and its service survived all of that.
        async with session.get(
            f"{base_url}/runtimes/alice-rt/services/m1", headers=auth(ALICE)
        ) as res:
            assert res.status == 200


@pytest.mark.asyncio
async def test_delete_all_stays_inside_the_callers_namespace(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "alice-rt", "m1")
        await create_runtime_as(session, base_url, BOB, "bob-rt", "m2")

        async with session.delete(f"{base_url}/runtimes", headers=auth(BOB)) as res:
            assert res.status == 200

        async with session.get(
            f"{base_url}/runtimes/alice-rt", headers=auth(ALICE)
        ) as res:
            assert res.status == 200
        async with session.get(f"{base_url}/runtimes/bob-rt", headers=auth(BOB)) as res:
            assert res.status == 404


@pytest.mark.asyncio
async def test_cannot_delete_another_tenants_runtime(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "alice-rt", "m1")

        async with session.delete(
            f"{base_url}/runtimes/alice-rt", headers=auth(BOB)
        ) as res:
            assert res.status == 404

        async with session.get(
            f"{base_url}/runtimes/alice-rt", headers=auth(ALICE)
        ) as res:
            assert res.status == 200


@pytest.mark.asyncio
async def test_websocket_onto_another_tenants_runtime_is_rejected(servers):
    _server, base_url = await servers()
    ws_url = base_url.replace("http", "ws") + "/alice-rt"
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "alice-rt", "m1")

        with pytest.raises(aiohttp.WSServerHandshakeError):
            async with session.ws_connect(ws_url, headers=auth(BOB)):
                pass

        async with session.ws_connect(ws_url, headers=auth(ALICE)) as ws:
            assert not ws.closed


@pytest.mark.asyncio
async def test_session_token_resolves_into_the_minters_namespace(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "alice-rt", "m1")

        async with session.post(
            f"{base_url}/runtimes/alice-rt/session-token", headers=auth(ALICE)
        ) as res:
            assert res.status == 200
            token = (await res.json())["token"]

        # This is the coordinator's machine path: the token stands in for Alice.
        async with session.get(f"{base_url}/runtimes", headers=auth(token)) as res:
            assert res.status == 200
            body = await res.json()
            assert [r["id"] for r in body["runtimes"]] == ["alice-rt"]


@pytest.mark.asyncio
async def test_no_auth_collapses_to_a_single_namespace(servers):
    # The dev/loopback path must keep behaving exactly as it did before tenancy.
    _server, base_url = await servers({"build_authenticator": None})
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={"id": "rt-1", "name": "Node", "services": []},
        ) as res:
            assert res.status == 200
        async with session.get(f"{base_url}/runtimes") as res:
            body = await res.json()
            assert [r["id"] for r in body["runtimes"]] == ["rt-1"]
        async with session.get(f"{base_url}/runtimes/rt-1") as res:
            assert res.status == 200


# ── Quotas ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_cap_is_per_tenant(servers):
    _server, base_url = await servers({"quotas": {"max_runtimes_per_user": 1}})
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "rt-1", "m1")

        async with session.post(
            f"{base_url}/runtimes",
            headers=auth(ALICE),
            json={"id": "rt-2", "name": "Node", "services": []},
        ) as res:
            assert res.status == 429

        # Bob has his own allowance; Alice exhausting hers must not spend it.
        await create_runtime_as(session, base_url, BOB, "rt-1", "m2")


@pytest.mark.asyncio
async def test_recreating_an_existing_runtime_is_allowed_at_the_cap(servers):
    _server, base_url = await servers({"quotas": {"max_runtimes_per_user": 1}})
    async with aiohttp.ClientSession() as session:
        await create_runtime_as(session, base_url, ALICE, "rt-1", "m1")

        # Re-POSTing an existing id is how a browser reattaches after a reload;
        # the cap must not turn that into a failure.
        async with session.post(
            f"{base_url}/runtimes",
            headers=auth(ALICE),
            json={"id": "rt-1", "name": "Node", "services": []},
        ) as res:
            assert res.status == 200


@pytest.mark.asyncio
async def test_service_cap_applies_on_create_and_on_add(servers):
    _server, base_url = await servers({"quotas": {"max_services_per_runtime": 2}})
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            headers=auth(ALICE),
            json={
                "id": "rt-big",
                "name": "Node",
                "services": [
                    {"serviceId": "monitor", "uuid": "m1"},
                    {"serviceId": "monitor", "uuid": "m2"},
                    {"serviceId": "monitor", "uuid": "m3"},
                ],
            },
        ) as res:
            assert res.status == 429

        async with session.post(
            f"{base_url}/runtimes",
            headers=auth(ALICE),
            json={
                "id": "rt-1",
                "name": "Node",
                "services": [
                    {"serviceId": "monitor", "uuid": "m1"},
                    {"serviceId": "monitor", "uuid": "m2"},
                ],
            },
        ) as res:
            assert res.status == 200

        async with session.post(
            f"{base_url}/runtimes/rt-1/services",
            headers=auth(ALICE),
            json={"serviceId": "monitor", "uuid": "m3"},
        ) as res:
            assert res.status == 429


@pytest.mark.asyncio
async def test_no_limits_by_default(servers):
    _server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        for rid in ("rt-1", "rt-2", "rt-3"):
            async with session.post(
                f"{base_url}/runtimes",
                headers=auth(ALICE),
                json={
                    "id": rid,
                    "name": "Node",
                    "services": [
                        {"serviceId": "monitor", "uuid": f"{rid}-a"},
                        {"serviceId": "monitor", "uuid": f"{rid}-b"},
                    ],
                },
            ) as res:
                assert res.status == 200
