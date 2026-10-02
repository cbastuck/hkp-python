"""More than one claim to one address.

A mount's address is derived from what the mount is called, so a board open in
a client and the same board deployed claim the same one, as does a runtime
rebuilt under its id before the one it replaces lets go. One claim answers;
releasing any of them leaves the address with whoever still holds it.
"""

from __future__ import annotations

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from hkp.mounts import MountRegistry
from hkp.runtime import board_space

OWNER = "user-1"
BOARD = "doorbell"
DEPLOYED = board_space(OWNER, BOARD)


def registry() -> MountRegistry:
    return MountRegistry(lambda path: f"http://h{path}", "secret")


def claim(mounts: MountRegistry, name: str, space: str | None = None):
    async def handler(request, context):
        return web.Response(text=name)

    return mounts.register(
        OWNER, "py", "hook", handler, board_name=BOARD, space=space
    )


async def ask(mounts: MountRegistry, path: str) -> str:
    """Who answers a request to the address, or nobody."""
    mount_id = path.rsplit("/", 1)[-1]
    request = make_mocked_request("GET", path, match_info={"mount_id": mount_id})
    try:
        return (await mounts.handle(request)).text
    except web.HTTPNotFound:
        return "nobody"


def test_is_one_address_whichever_space_the_runtime_is_in():
    mounts = registry()
    assert claim(mounts, "client").url == claim(mounts, "deployed", DEPLOYED).url


async def test_is_answered_by_the_deployed_board_whoever_claimed_last():
    mounts = registry()
    deployed = claim(mounts, "deployed", DEPLOYED)

    claim(mounts, "client")

    assert await ask(mounts, deployed.path) == "deployed"


async def test_stays_the_deployed_boards_when_the_client_that_opened_it_leaves():
    # Opening a deployed board in the playground and closing it again must not
    # take its endpoint down.
    mounts = registry()
    deployed = claim(mounts, "deployed", DEPLOYED)
    client = claim(mounts, "client")

    client.release()

    assert await ask(mounts, deployed.path) == "deployed"


async def test_falls_to_the_clients_copy_when_the_deployed_board_lets_go():
    mounts = registry()
    client = claim(mounts, "client")
    deployed = claim(mounts, "deployed", DEPLOYED)

    deployed.release()

    assert await ask(mounts, client.path) == "client"


async def test_the_newer_of_two_claims_of_a_kind_answers_then_the_older_again():
    mounts = registry()
    first = claim(mounts, "first")
    second = claim(mounts, "second")
    assert await ask(mounts, first.path) == "second"

    second.release()

    assert await ask(mounts, first.path) == "first"


async def test_answers_nobody_once_every_claim_is_released():
    mounts = registry()
    client = claim(mounts, "client")
    deployed = claim(mounts, "deployed", DEPLOYED)

    client.release()
    client.release()
    deployed.release()

    assert await ask(mounts, client.path) == "nobody"


async def test_keeps_a_boards_claim_when_the_clients_runtimes_are_removed():
    mounts = registry()
    deployed = claim(mounts, "deployed", DEPLOYED)
    claim(mounts, "client")

    mounts.release_owner(OWNER)
    mounts.release_runtime(OWNER, "py")

    assert await ask(mounts, deployed.path) == "deployed"
