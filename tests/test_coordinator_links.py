"""The runtime server's end of a coordinator connection.

A coordinator never dials a runtime server: a person's client tells the server
to connect to one, with a ticket, and the runtime that ticket speaks for is
built and driven over the connection the server opened.

The coordinator itself lives in hkp-node. What stands in for it here speaks the
same protocol (hkp-node/src/coordinator/participantProtocol.ts) and nothing
more: it checks the ticket on the upgrade, welcomes whoever says hello, and lets
a test send the requests a coordinator would.
"""
from __future__ import annotations

import asyncio
import json
import stat
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from hkp.binary_frame import decode_frame, encode_frame, from_binary, to_binary
from hkp.data import BinaryData, FloatRingBuffer
from hkp.coordinator_links import (
    CLOSE_TICKET_REVOKED,
    FileLinkStore,
    LinkRecord,
    MemoryLinkStore,
    join_url_for,
)
from hkp.runtime import board_space
from hkp.server import create_runtime_server
from hkp.services.monitor import MONITOR_DESCRIPTOR

OWNER = "anonymous"
FAST = {"reconnect_delay": 0.02, "max_reconnect_delay": 0.05, "introduce_timeout": 2}


class FakeCoordinator:
    """Accepts joins the way hkp-node's coordinator does."""

    def __init__(self) -> None:
        self.tickets: set[str] = {"hkpt_good"}
        self.hellos: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        # Binary frames, decoded: (header, shape, payload).
        self.binary: list[Any] = []
        self.connection: web.WebSocketResponse | None = None
        self._waiting: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._counter = 0
        self._runner: web.AppRunner | None = None
        self.url = ""
        self.port = 0

    async def start(self, port: int = 0) -> None:
        app = web.Application()
        app.router.add_get("/coordinator/join", self._join)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/coordinator"

    async def stop(self) -> None:
        if self.connection is not None and not self.connection.closed:
            await self.connection.close()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def _join(self, request: web.Request) -> web.StreamResponse:
        header = request.headers.get("Authorization", "")
        if header.removeprefix("Bearer ") not in self.tickets:
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connection = ws
        async for message in ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                self.binary.append(decode_frame(message.data))
                continue
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(message.data)
            if data["type"] == "hello":
                self.hellos.append(data)
                await ws.send_json(
                    {"type": "welcome", "boardName": "doorbell", "runtimeId": "py"}
                )
            elif data["type"] == "response":
                self._waiting.pop(data["requestId"]).set_result(data)
            else:
                self.events.append(data)
        return ws

    async def request(self, op: str, **payload: Any) -> dict[str, Any]:
        self._counter += 1
        request_id = f"req-{self._counter}"
        future = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = future
        assert self.connection is not None
        await self.connection.send_json(
            {"type": "request", "requestId": request_id, "op": op, **payload}
        )
        return await asyncio.wait_for(future, 3)

    async def send(self, message: dict[str, Any]) -> None:
        assert self.connection is not None
        await self.connection.send_json(message)

    async def send_binary(
        self, header: dict[str, Any], shape: dict[str, Any], payload: bytes
    ) -> None:
        assert self.connection is not None
        await self.connection.send_bytes(encode_frame(header, shape, payload))


async def eventually(check, what: str = "condition", timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        outcome = check()
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        if outcome:
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"Timed out waiting for {what}")
        await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def coordinator():
    fake = FakeCoordinator()
    await fake.start()
    yield fake
    await fake.stop()


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(**options: Any):
        server = create_runtime_server(
            {
                "external_host": "127.0.0.1",
                "coordinator_link_options": FAST,
                **options,
            }
        )
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


def introduction(coordinator: FakeCoordinator, **overrides: Any) -> dict[str, Any]:
    return {
        "coordinatorUrl": coordinator.url,
        "ticket": "hkpt_good",
        "boardName": "doorbell",
        "runtimeId": "py",
        **overrides,
    }


PROVISION = {
    "name": "Python",
    "boardName": "doorbell",
    "state": {},
    "services": [
        {
            "uuid": "mon-1",
            "serviceId": MONITOR_DESCRIPTOR.service_id,
            "serviceName": "Monitor",
            "state": {},
        }
    ],
}


def test_join_url_is_the_coordinators_join_endpoint():
    assert (
        join_url_for("http://127.0.0.1:8080/coordinator")
        == "ws://127.0.0.1:8080/coordinator/join"
    )
    assert (
        join_url_for("https://cloud.example/coordinator/")
        == "wss://cloud.example/coordinator/join"
    )


@pytest.mark.parametrize(
    "address", ["file:///etc/passwd", "hkp://remotes/local", "not a url", ""]
)
def test_join_url_refuses_what_is_not_an_http_address(address: str):
    with pytest.raises(ValueError):
        join_url_for(address)


async def test_says_it_can_be_introduced_to_a_coordinator(servers):
    _, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/runtimes") as res:
            body = await res.json()
    assert body["coordinatorLinks"] is True
    assert body["server"] == "python"


async def test_connects_with_the_ticket_and_reports_the_link_never_the_ticket(
    servers, coordinator
):
    _, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/coordinator-links", json=introduction(coordinator)
        ) as res:
            assert res.status == 201
            assert await res.json() == {"connected": True}
        async with session.get(f"{base_url}/coordinator-links") as res:
            text = await res.text()

    assert json.loads(text)["links"] == [
        {
            "boardName": "doorbell",
            "runtimeId": "py",
            "coordinatorUrl": coordinator.url,
            "connected": True,
            # Introduced, and not yet built by the coordinator.
            "running": False,
        }
    ]
    assert "hkpt_good" not in text
    assert coordinator.hellos[0]["server"] == "python"
    assert coordinator.hellos[0]["runtimeExists"] is False


async def test_says_why_when_the_ticket_is_not_accepted_and_keeps_nothing(
    servers, coordinator
):
    server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/coordinator-links",
            json=introduction(coordinator, ticket="hkpt_made-up"),
        ) as res:
            assert res.status == 502
            assert "did not accept the ticket" in (await res.json())["error"]
    assert server.coordinator_links.list(OWNER) == []


async def test_says_why_when_there_is_no_coordinator_there(servers):
    server, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/coordinator-links",
            json={
                "coordinatorUrl": "http://127.0.0.1:1/coordinator",
                "ticket": "hkpt_x",
                "boardName": "doorbell",
                "runtimeId": "py",
            },
        ) as res:
            assert res.status == 502
            assert (await res.json())["error"]
    assert server.coordinator_links.list(OWNER) == []


async def test_refuses_an_introduction_that_leaves_something_out(servers):
    _, base_url = await servers()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/coordinator-links", json={"ticket": "hkpt_x"}
        ) as res:
            assert res.status == 400


async def test_the_coordinator_builds_configures_and_releases_the_runtime(
    servers, coordinator
):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )

    built = await coordinator.request("provision", **PROVISION)
    assert built["ok"] is True
    assert [svc["uuid"] for svc in built["data"]["services"]] == ["mon-1"]
    assert built["data"]["missingSecrets"] == []
    runtime = server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py")
    # The coordinator's, until the coordinator says otherwise.
    assert runtime.garbage_collected is False

    described = await coordinator.request("describe")
    assert [svc["uuid"] for svc in described["data"]["services"]] == ["mon-1"]

    logging = await coordinator.request(
        "setState", state={"logging": True, "logLevel": "debug"}
    )
    assert logging["data"]["logging"] is True
    assert logging["data"]["logLevel"] == "debug"

    removed = await coordinator.request("remove")
    assert removed["ok"] is True
    assert server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py") is None


async def test_builds_for_the_board_it_was_introduced_for_whatever_it_is_told(
    servers, coordinator
):
    # A ticket speaks for one board: the board name a runtime is built under
    # decides its mount addresses.
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )

    await coordinator.request(
        "provision", **{**PROVISION, "boardName": "someone-elses-board"}
    )

    built = server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py")
    assert built.board_name == "doorbell"
    assert server.runtime_app.get_board_runtimes(OWNER) == [built]


async def test_answers_what_it_cannot_do_with_an_error_rather_than_silence(
    servers, coordinator
):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )

    unknown = await coordinator.request("shell")
    assert unknown["ok"] is False
    assert "Unknown operation" in unknown["error"]

    absent = await coordinator.request("describe")
    assert absent["ok"] is False

    bad = await coordinator.request(
        "provision",
        **{**PROVISION, "services": [{"uuid": "x", "serviceId": "no-such-service"}]},
    )
    assert bad["ok"] is False
    assert "no-such-service" in bad["error"]


async def test_is_driven_over_the_link_and_says_what_its_runtime_says(
    servers, coordinator
):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)

    await coordinator.send({"type": "processRuntime", "params": {"hello": "there"}})

    await eventually(
        lambda: any(event["type"] == "result" for event in coordinator.events),
        "the result",
    )
    result = next(e for e in coordinator.events if e["type"] == "result")
    assert result["data"] == {"hello": "there"}
    assert any(
        event["type"] == "notification" and event["serviceUuid"] == "mon-1"
        for event in coordinator.events
    )


ALICE = {"sub": "auth0|alice", "email": "alice@example.com", "name": "Alice"}


def _seen_by(server, service_uuid: str = "mon-1") -> list[Any]:
    """Records the context each call of a service runs under. The context
    travels with the call rather than with the data, so this is the only way
    to observe it from outside."""
    runtime = server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py")
    service = runtime.get_service(service_uuid)
    seen: list[Any] = []
    process = service.process

    def recording(data: Any, notify: Any) -> Any:
        seen.append(runtime.current_context())
        return process(data, notify)

    service.process = recording
    return seen


async def test_takes_the_run_and_its_caller_from_the_coordinator_and_hands_them_back(
    servers, coordinator
):
    # The link is the board's own, and the coordinator on it is what verified
    # the person — so here, and nowhere else, a caller is taken as stated. It
    # comes back with the result, which is how the board's next runtime learns
    # who began the run.
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)
    seen = _seen_by(server)
    run = {"runId": "run-1", "caller": ALICE}

    await coordinator.send(
        {"type": "processRuntime", "params": {"n": 1}, "context": run}
    )

    await eventually(
        lambda: any(event["type"] == "result" for event in coordinator.events),
        "the result",
    )
    assert seen[0].run_id == "run-1"
    assert seen[0].caller.to_wire() == ALICE
    result = next(e for e in coordinator.events if e["type"] == "result")
    assert result["context"] == run
    # What the service said while it ran is the caller's to hear, and says so.
    said = next(e for e in coordinator.events if e["type"] == "notification")
    assert said["caller"] == ALICE


async def test_a_run_nobody_began_names_nobody(servers, coordinator):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)

    await coordinator.send({"type": "processRuntime", "params": {"n": 1}})

    await eventually(
        lambda: any(event["type"] == "result" for event in coordinator.events),
        "the result",
    )
    result = next(e for e in coordinator.events if e["type"] == "result")
    assert "caller" not in result["context"]
    assert result["context"]["runId"]
    said = next(e for e in coordinator.events if e["type"] == "notification")
    assert "caller" not in said


async def test_begins_at_one_service_when_asked_and_says_what_came_of_it(
    servers, coordinator
):
    # What a facade's process action means on a deployed board: the answer
    # says the work was taken, and what the pipeline produced follows as the
    # runtime's output, in the same run.
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    two = {
        **PROVISION,
        "services": [
            {**PROVISION["services"][0], "uuid": "first"},
            {**PROVISION["services"][0], "uuid": "second"},
        ],
    }
    await coordinator.request("provision", **two)
    first = _seen_by(server, "first")
    second = _seen_by(server, "second")
    run = {"runId": "run-2", "caller": ALICE}

    answer = await coordinator.request(
        "processService", serviceUuid="second", params={"n": 2}, context=run
    )

    assert answer["ok"] is True
    assert answer["data"] == {"accepted": True}
    await eventually(
        lambda: any(event["type"] == "result" for event in coordinator.events),
        "the result",
    )
    assert first == []
    assert second[0].caller.to_wire() == ALICE
    result = next(e for e in coordinator.events if e["type"] == "result")
    assert result["data"] == {"n": 2}
    assert result["context"] == run


async def test_says_why_it_cannot_begin_at_a_service_that_is_not_there(
    servers, coordinator
):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)

    answer = await coordinator.request(
        "processService", serviceUuid="nobody", params={}
    )

    assert answer["ok"] is False
    assert 'no service "nobody"' in answer["error"]


async def test_is_built_with_the_assets_the_coordinator_sends(servers, coordinator):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    built = await coordinator.request(
        "provision",
        **{
            **PROVISION,
            "services": [
                {
                    "uuid": "asset-1",
                    "serviceId": "asset",
                    "serviceName": "Asset",
                    "state": {"asset": "hkp-asset://day"},
                }
            ],
            "assets": {"day": {"id": "day", "mediaType": "text/plain", "text": "sun"}},
        },
    )
    assert built["ok"] is True

    await coordinator.send({"type": "processRuntime", "params": {}})

    await eventually(
        lambda: any(event["type"] == "result" for event in coordinator.events),
        "the result",
    )
    result = next(e for e in coordinator.events if e["type"] == "result")
    assert result["data"]["meta"]["status"] == 200
    assert result["data"]["body"] == "sun"


async def test_bytes_arrive_as_bytes_and_leave_as_bytes(servers, coordinator):
    # As text they could only be described — a size, a type — and the runtime
    # after this one would be handed a description.
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)
    sent = bytes(range(256)) * 16

    await coordinator.send_binary({"type": "processRuntime"}, {"kind": "bytes"}, sent)

    await eventually(lambda: coordinator.binary, "the result")
    header, shape, payload = coordinator.binary[0]
    # The header is the message without its value: what it is, and the run it
    # was produced in — which a coordinator hands to the next runtime.
    assert header["type"] == "result"
    assert header["context"]["runId"]
    assert shape == {"kind": "bytes"}
    assert payload == sent
    # Nothing was also said as text.
    assert not any(event["type"] == "result" for event in coordinator.events)


async def test_a_ring_buffer_keeps_its_samples_and_what_identifies_it(
    servers, coordinator
):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)
    buffer = FloatRingBuffer.from_floats([0.5, -1.0, 0.25], id=7, ts=1234)

    await coordinator.send_binary(
        {"type": "processRuntime"},
        {"kind": "floatRingBuffer", "id": 7, "ts": 1234},
        buffer.samples,
    )

    await eventually(lambda: coordinator.binary, "the result")
    _, shape, payload = coordinator.binary[0]
    assert shape == {"kind": "floatRingBuffer", "id": 7, "ts": 1234}
    assert payload == buffer.samples


def test_reads_a_frame_as_the_coordinator_writes_it():
    # The fixture hkp-node's encoder produces; hkp-frontend's tests read the
    # same bytes.
    fixture = bytes.fromhex(
        "000000777b2274797065223a2270726f6365737352756e74696d65222c2272756e74696d654964223a227569222c22726571756573744964223a22722d31222c2262696e617279223a7b226b696e64223a226d69786564222c226a736f6e223a7b226d657461223a7b226e616d65223a22612e62696e227d7d7d7d0001feff"
    )

    header, shape, payload = decode_frame(fixture)

    assert header == {"type": "processRuntime", "runtimeId": "ui", "requestId": "r-1"}
    assert from_binary(shape, payload) == {
        "meta": {"name": "a.bin"},
        "binary": b"\x00\x01\xfe\xff",
    }


def test_what_travels_as_bytes_and_what_as_text():
    assert to_binary(b"\x01\x02") == ({"kind": "bytes"}, b"\x01\x02")
    assert to_binary(BinaryData(b"\x03")) == ({"kind": "bytes"}, b"\x03")
    assert to_binary({"meta": {"status": 200}, "binary": b"\x04"}) == (
        {"kind": "mixed", "json": {"meta": {"status": 200}}},
        b"\x04",
    )
    assert to_binary({"a": 1}) is None
    assert to_binary("text") is None
    assert to_binary(None) is None


def test_a_payload_becomes_the_value_a_service_expects():
    assert from_binary({"kind": "bytes"}, b"\x01") == BinaryData(b"\x01")
    assert from_binary({"kind": "mixed", "json": {"meta": {"n": 1}}}, b"\x02") == {
        "meta": {"n": 1},
        "binary": b"\x02",
    }
    buffer = from_binary({"kind": "floatRingBuffer", "id": 3, "ts": 9}, b"\x00" * 8)
    assert isinstance(buffer, FloatRingBuffer)
    assert (buffer.id, buffer.ts, buffer.num_samples) == (3, 9, 2)


@pytest.mark.parametrize(
    "raw",
    [
        b"\x00\x00",
        b"\x00\x00\x00\x32{",
        b"\x00\x00\x00\x01x",
        encode_frame({"type": "result"}, {"kind": "unknown"}, b""),
    ],
)
def test_a_frame_that_is_not_one_is_not_read(raw: bytes):
    assert decode_frame(raw) is None


async def test_reconnects_on_its_own_and_says_its_runtime_is_still_there(
    servers, coordinator
):
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)

    await coordinator.connection.close()

    await eventually(lambda: len(coordinator.hellos) == 2, "the reconnect")
    assert coordinator.hellos[1]["runtimeExists"] is True
    assert server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py") is not None


async def test_drops_the_link_and_the_runtime_when_its_ticket_is_revoked(
    servers, coordinator
):
    store = MemoryLinkStore()
    server, _ = await servers(coordinator_links=store)
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)
    assert len(store.load()) == 1

    await coordinator.connection.close(code=CLOSE_TICKET_REVOKED)

    await eventually(lambda: store.load() == [], "the ticket to be forgotten")
    assert server.coordinator_links.list(OWNER) == []
    assert server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py") is None


async def test_forgets_a_ticket_the_coordinator_no_longer_holds(servers, coordinator):
    # The board was deleted while this server was away.
    store = MemoryLinkStore()
    store.save([LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_stale")])
    server, _ = await servers(coordinator_links=store)

    server.coordinator_links.restore()

    await eventually(lambda: store.load() == [], "the ticket to be forgotten")


async def test_reconnects_after_a_restart_with_the_ticket_it_kept(servers, coordinator):
    store = MemoryLinkStore()
    first, _ = await servers(coordinator_links=store)
    await first.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await first.stop()

    second, _ = await servers(coordinator_links=store)
    second.coordinator_links.restore()

    await eventually(lambda: len(coordinator.hellos) == 2, "the reconnect")
    # An empty process holding the ticket: the coordinator rebuilds from here.
    assert coordinator.hellos[1]["runtimeExists"] is False


async def test_keeps_the_connection_it_has_when_introduced_again(servers, coordinator):
    # Being introduced is the first step of a deploy that may yet fail.
    server, _ = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )

    # Not even a ticket the coordinator would refuse costs it the link.
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_never-presented")
    )

    assert len(coordinator.hellos) == 1
    links = server.coordinator_links.list(OWNER)
    assert [link["connected"] for link in links] == [True]


async def test_refuses_to_be_another_coordinators_naming_the_first(
    servers, coordinator
):
    other = FakeCoordinator()
    await other.start()
    try:
        server, _ = await servers()
        await server.coordinator_links.introduce(
            LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
        )

        with pytest.raises(ConnectionError) as refused:
            await server.coordinator_links.introduce(
                LinkRecord(OWNER, "doorbell", "py", other.url, "hkpt_good")
            )

        assert "already deployed here by" in str(refused.value)
        assert coordinator.url in str(refused.value)
        assert other.hellos == []
        links = server.coordinator_links.list(OWNER)
        assert [link["coordinatorUrl"] for link in links] == [coordinator.url]
    finally:
        await other.stop()


async def test_a_runtime_id_two_boards_share_is_a_link_of_each(servers, coordinator):
    # Boards ship the same handful of ids. Being introduced for a second board
    # must not cost the first one its link.
    server, _ = await servers()
    for board in ("doorbell", "garden"):
        await server.coordinator_links.introduce(
            LinkRecord(OWNER, board, "py", coordinator.url, "hkpt_good")
        )

    links = server.coordinator_links.list(OWNER)
    assert sorted(link["boardName"] for link in links) == ["doorbell", "garden"]
    assert all(link["connected"] for link in links)


async def test_a_boards_runtime_is_not_the_one_a_client_creates_under_its_id(
    servers, coordinator
):
    # What opening the same board in the playground does: it posts a runtime
    # under the id the deployed board uses, and deletes it when it leaves.
    server, base_url = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)
    deployed = server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py")

    async with aiohttp.ClientSession() as session:
        created = {"id": "py", "name": "Python", "boardName": "doorbell", "services": []}
        async with session.post(f"{base_url}/runtimes", json=created) as res:
            assert res.status == 200
        async with session.get(f"{base_url}/runtimes") as res:
            listed = (await res.json())["runtimes"]
        async with session.delete(f"{base_url}/runtimes/py") as res:
            assert res.status == 200
        async with session.delete(f"{base_url}/runtimes") as res:
            assert res.status == 200

    # The client saw its own runtime and never the board's.
    assert [runtime["services"] for runtime in listed] == [[]]
    assert server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py") is deployed
    described = await coordinator.request("describe")
    assert [svc["uuid"] for svc in described["data"]["services"]] == ["mon-1"]


async def test_leaves_a_board_when_asked_dropping_the_runtime(servers, coordinator):
    server, base_url = await servers()
    await server.coordinator_links.introduce(
        LinkRecord(OWNER, "doorbell", "py", coordinator.url, "hkpt_good")
    )
    await coordinator.request("provision", **PROVISION)

    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{base_url}/coordinator-links/doorbell/py") as res:
            assert res.status == 200
        async with session.delete(f"{base_url}/coordinator-links/doorbell/py") as res:
            assert res.status == 404

    assert server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py") is None
    assert server.coordinator_links.list(OWNER) == []


async def test_credentials_come_from_the_client_and_missing_ones_are_named(
    servers, coordinator
):
    server, base_url = await servers()
    with_secrets = {
        **PROVISION,
        "services": [
            {
                "uuid": "client",
                "serviceId": "http-client",
                "serviceName": "HTTP",
                "state": {
                    "url": "https://api.example",
                    "token": "{{secret.api.key}}",
                    "other": "{{secret.not.sent}}",
                },
            }
        ],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/coordinator-links",
            json=introduction(coordinator, secrets={"api.key": {"value": "s3cret"}}),
        ) as res:
            assert res.status == 201

    built = await coordinator.request("provision", **with_secrets)

    assert built["data"]["missingSecrets"] == ["not.sent"]
    assert server.runtime_app.get_runtime(board_space(OWNER, "doorbell"), "py").secrets().aliases() == [
        "api.key"
    ]
    # Never to the coordinator: not in what it was told, not in what it asks.
    assert "s3cret" not in json.dumps(built)
    assert "s3cret" not in json.dumps(await coordinator.request("describe"))
    assert "s3cret" not in json.dumps(coordinator.hellos)


def test_tickets_on_disk_are_readable_by_their_owner_only(tmp_path):
    file = tmp_path / "nested" / "coordinator-links.json"
    store = FileLinkStore(file)
    record = LinkRecord(OWNER, "doorbell", "py", "http://c/coordinator", "hkpt_x")

    store.save([record])

    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert stat.S_IMODE(file.parent.stat().st_mode) == 0o700
    assert store.load() == [record]


def test_reads_back_nothing_from_a_file_that_is_missing_or_not_what_it_wrote(tmp_path):
    file = tmp_path / "coordinator-links.json"
    assert FileLinkStore(file).load() == []

    file.write_text("{ not json")
    assert FileLinkStore(file).load() == []

    file.write_text(json.dumps([{"owner": "x"}, "junk", {"owner": 1}]))
    assert FileLinkStore(file).load() == []
