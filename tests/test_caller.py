"""Who began a run — ported from hkp-node/tests/caller.test.ts.

The caller is stated by the server that verified a token and by nothing else.
What is pinned here is each way a client begins a run: whatever a request says
about a caller is never read, and a server without authentication names none.
The context travels with the call rather than with the data, so it is observed
by recording what a service ran under.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import aiohttp
import pytest_asyncio

from hkp.auth import (
    AuthenticatedUser,
    identity_from_claims,
    may_own,
)
from hkp.data import FloatRingBuffer
from hkp.runtime import (
    caller_of,
    child_run,
    context_for_client,
    context_from_link,
    context_from_wire,
    context_to_wire,
)
from hkp.server import create_runtime_server
from hkp.types import Caller, PersonRunActor, ProcessContext, SourceRunActor
from hkp.yas import serialize_message

ALICE = AuthenticatedUser(sub="auth0|alice", email="alice@example.com")
#: Signed in, and without an address anybody verified.
CAROL = AuthenticatedUser(sub="auth0|carol")

FORGED = {
    "runId": "run-from-client",
    "actor": {
        "kind": "person",
        "sub": "auth0|bob",
        "email": "bob@example.com",
        "name": "Bob",
        "expiresAt": 9_999_999_999_999,
    },
}


class KnownPeople:
    """A bearer token is the ``sub`` it authenticates as."""

    def __init__(self, _resolve_opaque_token: Callable[[str], Any]) -> None:
        self._known = {user.sub: user for user in (ALICE, CAROL)}

    async def authorize_owner(self, token: str | None) -> AuthenticatedUser | None:
        return self._known.get(token or "")


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(**options: Any):
        server = create_runtime_server({"external_host": "127.0.0.1", **options})
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


def auth(user: AuthenticatedUser | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {user.sub}"} if user else {}


async def runtime_with_spy(
    server,
    session: aiohttp.ClientSession,
    base_url: str,
    user: AuthenticatedUser | None,
    method: str = "process",
) -> list[Any]:
    """A runtime of one monitor, recording the context each named call runs under."""
    async with session.post(
        f"{base_url}/runtimes",
        headers=auth(user),
        json={
            "id": "rt-1",
            "name": "Python",
            "services": [{"serviceId": "monitor", "uuid": "mon-1"}],
        },
    ) as res:
        assert res.status == 200
    runtime = server.runtime_app.get_runtime(user.sub if user else "anonymous", "rt-1")
    service = runtime.get_service("mon-1")
    seen: list[Any] = []
    called = getattr(service, method)

    def recording(*args: Any) -> Any:
        seen.append(runtime.current_context())
        return called(*args)

    setattr(service, method, recording)
    return seen


async def test_a_service_configure_call_is_the_tokens(servers):
    server, base_url = await servers(build_authenticator=KnownPeople)
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(
            server, session, base_url, ALICE, method="configure"
        )

        async with session.post(
            f"{base_url}/runtimes/rt-1/services/mon-1",
            headers=auth(ALICE),
            json={"__context": FORGED, "logToConsole": True},
        ) as res:
            assert res.status == 200

    assert len(seen) == 1
    assert seen[0].run_id != FORGED["runId"]
    assert seen[0].actor.sub == ALICE.sub
    assert seen[0].actor.email == ALICE.email


async def test_a_service_process_call_is_the_tokens_whatever_the_body_claims(servers):
    server, base_url = await servers(build_authenticator=KnownPeople)
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(server, session, base_url, ALICE)

        async with session.post(
            f"{base_url}/runtimes/rt-1/services/mon-1/process",
            headers=auth(ALICE),
            json={"__context": FORGED, "actor": FORGED["actor"]},
        ) as res:
            assert res.status == 200

    assert seen[0].run_id == "run-from-client"
    assert seen[0].actor.sub == ALICE.sub
    assert seen[0].actor.email == ALICE.email


async def test_a_runtime_process_call_is_the_tokens(servers):
    server, base_url = await servers(build_authenticator=KnownPeople)
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(server, session, base_url, ALICE)

        async with session.post(
            f"{base_url}/runtimes/rt-1",
            headers=auth(ALICE),
            json={"__context": FORGED, "context": FORGED},
        ) as res:
            assert res.status == 200

    assert seen[0].actor.sub == ALICE.sub
    assert seen[0].actor.email == ALICE.email


async def test_a_process_on_the_runtimes_socket_is_whoever_opened_it(servers):
    server, base_url = await servers(build_authenticator=KnownPeople)
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(server, session, base_url, ALICE)

        async with session.ws_connect(
            f"{base_url}/rt-1", headers=auth(ALICE)
        ) as ws:
            await ws.send_str(
                json.dumps(
                    {"type": "processRuntime", "params": {}, "context": FORGED}
                )
            )
            while True:
                message = json.loads((await ws.receive()).data)
                if message.get("type") == "result":
                    break

    assert seen[0].run_id == "run-from-client"
    assert seen[0].actor.sub == ALICE.sub
    assert seen[0].actor.email == ALICE.email


async def test_bytes_on_the_runtimes_socket_are_whoever_opened_it_too(servers):
    server, base_url = await servers(build_authenticator=KnownPeople)
    frame = serialize_message(FloatRingBuffer.from_floats([1.0, -1.0]), sender="")
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(server, session, base_url, ALICE)

        async with session.ws_connect(
            f"{base_url}/rt-1", headers=auth(ALICE)
        ) as ws:
            await ws.send_bytes(frame)
            # Notifications come as text; the result of bytes comes as bytes.
            while (await ws.receive(timeout=5)).type != aiohttp.WSMsgType.BINARY:
                pass

    # A frame of bytes names no run, so one begins here — as the person the
    # socket was opened by, the same as a text frame's.
    assert seen[0].run_id
    assert seen[0].actor.sub == ALICE.sub
    assert seen[0].actor.email == ALICE.email


async def test_a_caller_has_no_email_when_none_was_verified(servers):
    server, base_url = await servers(build_authenticator=KnownPeople)
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(server, session, base_url, CAROL)

        async with session.post(
            f"{base_url}/runtimes/rt-1", headers=auth(CAROL), json={}
        ) as res:
            assert res.status == 200

    assert seen[0].actor.sub == CAROL.sub
    assert seen[0].actor.email is None


async def test_a_server_without_authentication_names_no_caller(servers):
    # Not a caller called anonymous: otherwise everybody on a development
    # machine would be the same person.
    server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        seen = await runtime_with_spy(server, session, base_url, None)

        async with session.post(
            f"{base_url}/runtimes/rt-1/services/mon-1/process",
            json={"__context": FORGED},
        ) as res:
            assert res.status == 200

    assert seen[0].actor == SourceRunActor(kind="local")


def test_never_reads_a_caller_from_a_clients_context():
    assert context_from_wire(FORGED).actor == SourceRunActor(kind="local")
    assert context_for_client(FORGED, None).actor == SourceRunActor(
        kind="local"
    )
    actor = context_for_client(FORGED, ALICE).actor
    assert actor.kind == "person"
    assert actor.sub == ALICE.sub
    assert actor.email == ALICE.email
    assert context_for_client(None, ALICE).run_id


def test_takes_a_caller_as_stated_over_a_participant_link():
    context = context_from_link({**FORGED, "requestId": "reply-here"})
    assert context.run_id == "run-from-client"
    assert context.actor.to_wire() == FORGED["actor"]
    # The reply address belongs to whoever was waiting at the other end.
    assert context.request_id is None
    # A malformed person is expired, rather than gaining another actor kind.
    malformed = context_from_link(
        {"runId": "r", "actor": {"kind": "person", "email": "x@y.z"}}
    )
    assert malformed.actor == PersonRunActor(kind="person", sub="", expires_at=0)
    assert context_from_link(None) is None


def test_says_a_run_as_another_runtime_is_told_it():
    context = ProcessContext(
        run_id="r",
        parent_run_id="p",
        request_id="q",
        actor=PersonRunActor(
            kind="person", sub="s", name="N", expires_at=9_999_999_999_999
        ),
    )
    assert context_to_wire(context) == {
        "runId": "r",
        "parentRunId": "p",
        "actor": {
            "kind": "person",
            "sub": "s",
            "name": "N",
            "expiresAt": 9_999_999_999_999,
        },
    }
    assert context_to_wire(
        ProcessContext(run_id="r", actor=SourceRunActor(kind="board"))
    ) == {"runId": "r", "actor": {"kind": "board"}}
    assert context_to_wire(None) is None


def test_states_no_caller_for_the_anonymous_tenant():
    assert caller_of(AuthenticatedUser(sub="anonymous")) is None
    assert caller_of(None) is None
    assert caller_of(CAROL) == Caller(sub=CAROL.sub)


def test_hands_the_caller_down_to_a_child_run():
    parent = ProcessContext(
        run_id="outer",
        actor=PersonRunActor(
            kind="person", sub="s", email="e@x.y", expires_at=9_999_999_999_999
        ),
    )
    child = child_run(parent)
    assert child.parent_run_id == "outer"
    assert child.actor == parent.actor
    assert child_run(
        ProcessContext(run_id="outer", actor=SourceRunActor(kind="board"))
    ).actor == SourceRunActor(kind="board")


def test_an_identity_carries_an_email_only_when_it_is_verified():
    assert identity_from_claims(
        {"sub": "a", "email": " Alice@Example.COM ", "email_verified": True}
    ) == AuthenticatedUser(sub="a", email="alice@example.com")
    assert identity_from_claims({"sub": "a", "email": "a@x.y"}) == AuthenticatedUser(
        sub="a"
    )
    assert identity_from_claims({"email": "a@x.y"}) is None


def test_the_allowlist_is_asked_only_of_who_may_own():
    allowed = ["alice@example.com"]
    assert may_own(ALICE, allowed) is True
    assert may_own(CAROL, allowed) is False
    assert may_own(CAROL, None) is True
