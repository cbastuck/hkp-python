"""Whether a public endpoint keeps its address.

Ported from hkp-node/tests/mount-stability.test.ts.

A mount is reached by an outside party that was configured with its URL by hand
— a webhook in somebody else's product. So the address is part of the contract
with that party, and an address that changed whenever a board was loaded meant
reconfiguring them after every restart, which nobody would keep doing. It is
derived from what identifies the mount, keyed by a secret only the server holds:
stable for the same mount, unguessable without the key, different for anything
else.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR

SECRET = "test-secret"


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(secret: str = SECRET):
        server = create_runtime_server(
            {"external_host": "127.0.0.1", "mount_secret": secret}
        )
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return address["base_url"]

    yield start
    for server in started:
        await server.stop()


async def mount_of(
    base_url: str,
    board_name: str = "Offer Intake",
    runtime_id: str = "node",
    uuid: str = "http-1",
    mount_name: str | None = None,
) -> str:
    """The path of the endpoint a board of this shape is given."""
    state: dict[str, Any] = {
        "bypass": False,
        "mode": "process_on_session",
        "pipeline": [],
    }
    if mount_name is not None:
        state["mountName"] = mount_name

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": runtime_id,
                "name": "Python",
                "boardName": board_name,
                "services": [
                    {
                        "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                        "uuid": uuid,
                        "state": state,
                    }
                ],
            },
        ) as res:
            assert res.status == 200
        async with session.get(
            f"{base_url}/runtimes/{runtime_id}/services/{uuid}"
        ) as res:
            assert res.status == 200
            return urlparse((await res.json())["__hkpMount"]).path


@pytest.mark.asyncio
async def test_address_survives_reloading_the_board(servers):
    base_url = await servers()

    first = await mount_of(base_url)
    second = await mount_of(base_url)

    assert second == first


@pytest.mark.asyncio
async def test_address_survives_restarting_the_server(servers):
    # The case that matters: the process the webhook was configured against is
    # gone, and a new one has to answer on the same URL.
    first = await mount_of(await servers())
    second = await mount_of(await servers())

    assert second == first


@pytest.mark.asyncio
async def test_address_differs_between_tenants():
    # Not observable through the server used here, which has auth off and
    # therefore one tenant, so it is asserted where the derivation happens.
    from hkp.mounts import MountRegistry

    registry = MountRegistry(lambda path: f"http://host{path}", SECRET)

    async def handler(_request, _context):
        raise AssertionError("not called")

    mine = registry.register(
        "auth0|alice", "node", "http-1", handler, board_name="Offer Intake"
    )
    theirs = registry.register(
        "auth0|bob", "node", "http-1", handler, board_name="Offer Intake"
    )

    assert mine is not None and theirs is not None
    assert mine.path != theirs.path


@pytest.mark.asyncio
async def test_address_differs_between_boards(servers):
    base_url = await servers()

    offers = await mount_of(base_url, board_name="Offer Intake")
    invoices = await mount_of(base_url, board_name="Invoices", runtime_id="other")

    assert invoices != offers


@pytest.mark.asyncio
async def test_address_differs_between_two_endpoints_on_one_board(servers):
    base_url = await servers()

    first = await mount_of(base_url, uuid="http-1")
    second = await mount_of(base_url, uuid="http-2", runtime_id="n2")

    assert second != first


@pytest.mark.asyncio
async def test_address_cannot_be_worked_out_without_the_key(servers):
    # The address is the capability to reach an unauthenticated endpoint, so
    # knowing the board, the runtime and the service must not be enough.
    known = await mount_of(await servers("one-secret"))
    elsewhere = await mount_of(await servers("another-secret"))

    assert elsewhere != known


@pytest.mark.asyncio
async def test_naming_an_endpoint_rotates_that_address_and_nothing_else(servers):
    # The deliberate lever: a board that wants a new URL for one endpoint
    # renames it, rather than having every address change underneath it.
    base_url = await servers()

    by_uuid = await mount_of(base_url)
    named = await mount_of(base_url, mount_name="missive-intake")

    assert named != by_uuid
    # And the new name is as stable as the old identity was.
    assert await mount_of(base_url, mount_name="missive-intake") == named


@pytest.mark.asyncio
async def test_renamed_endpoint_stops_answering_on_the_old_address(servers):
    base_url = await servers()

    before = await mount_of(base_url)
    after = await mount_of(base_url, mount_name="missive-intake")

    origin = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{origin}{before}") as res:
            assert res.status == 404
        async with session.get(f"{origin}{after}") as res:
            assert res.status == 200
