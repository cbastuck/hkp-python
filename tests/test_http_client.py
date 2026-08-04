"""HTTP Client — calling an endpoint, including one whose address a runtime
assigns.

Ported from hkp-node/tests/http-client.test.ts; the two runtimes implement the
same service and must behave the same way.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web

from hkp.mount import is_mount_reference, join_mount_path
from hkp.services.http_client import HttpClientService
from hkp.types import ServiceConfiguration


class HostSpy:
    """Captures what the service pushes through the rest of the pipeline."""

    def __init__(self) -> None:
        self.pushed: list[Any] = []
        self.emitted: list[Any] = []

    def process_from(self, _uuid: str, data: Any, _on_notification: Any) -> Any:
        self.pushed.append(data)
        return data

    def notify(self, _payload: Any, _instance_id: str) -> None:
        pass

    def emit_result(self, output: Any) -> None:
        self.emitted.append(output)


class Endpoint:
    """A server that records what it was sent and answers what it was told to."""

    def __init__(self, url: str, received: list[dict[str, Any]]) -> None:
        self.url = url
        self.received = received


@pytest_asyncio.fixture
async def endpoint():
    started: list[web.AppRunner] = []

    async def start(
        reply: Any = None,
    ) -> Endpoint:
        received: list[dict[str, Any]] = []

        async def handler(request: web.Request) -> web.Response:
            received.append(
                {
                    "method": request.method,
                    "path": request.path,
                    "contentType": request.headers.get("content-type"),
                    "userAgent": request.headers.get("user-agent"),
                    "body": await request.read(),
                }
            )
            answer = (reply or (lambda _p: {"body": "ok"}))(request.path)
            return web.Response(
                status=answer.get("status", 200),
                body=answer.get("body", ""),
                content_type=answer.get("contentType"),
            )

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        started.append(runner)
        port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
        return Endpoint(f"http://127.0.0.1:{port}", received)

    yield start
    for runner in started:
        await runner.cleanup()


def make_client(state: dict[str, Any]) -> tuple[HttpClientService, HostSpy]:
    service = HttpClientService(
        ServiceConfiguration(service_id="http-client", uuid="client-1", state=state)
    )
    host = HostSpy()
    service.set_host(host)
    return service, host


async def next_push(host: HostSpy) -> Any:
    """Wait until the service has pushed a result, or fail."""
    for _ in range(200):
        if host.pushed:
            return host.pushed[0]
        await asyncio.sleep(0.01)
    raise AssertionError("service pushed no result")


def test_mount_references_are_told_apart_from_addresses():
    assert is_mount_reference("hkp-mount://chat-node/peer-svc")
    # An address is the other form of the same field, never a reference.
    assert not is_mount_reference("http://127.0.0.1:8080/hosted/abc")
    # A bare pair is indistinguishable from a relative URL, so it is not one.
    assert not is_mount_reference("chat-node/peer-svc")
    assert not is_mount_reference("")


def test_a_path_is_joined_without_doubling_the_separator():
    assert join_mount_path("http://h:8080/hosted/abc", "/x") == "http://h:8080/hosted/abc/x"
    assert join_mount_path("http://h:8080/hosted/abc/", "/x") == "http://h:8080/hosted/abc/x"
    assert join_mount_path("http://h:8080/hosted/abc", "") == "http://h:8080/hosted/abc"


@pytest.mark.asyncio
async def test_calls_the_address_it_was_configured_with(endpoint):
    target = await endpoint(
        lambda _p: {"contentType": "application/json", "body": '{"ok": true}'}
    )
    service, host = make_client({"__hkpMount": target.url, "path": "/hello"})

    # The pipeline is synchronous, so the response cannot be returned from
    # process — it does not exist yet. It arrives through process_from instead.
    assert service.process(None, lambda _n: None) is None

    result = await next_push(host)
    assert target.received[0]["path"] == "/hello"
    assert result["meta"]["status"] == 200
    assert result["body"] == {"ok": True}
    # Nothing downstream stopped, so the runtime's result is emitted onward.
    assert len(host.emitted) == 1


@pytest.mark.asyncio
async def test_waits_while_a_reference_is_unresolved():
    service, host = make_client({"__hkpMount": "hkp-mount://other/peer-1"})
    notifications: list[Any] = []

    # Only the coordinator can turn a reference into an address. Until it does,
    # there is nothing to call — a normal state while a board comes up.
    assert service.process(None, notifications.append) is None
    await asyncio.sleep(0.05)
    assert host.pushed == []
    assert "hkp-mount://other/peer-1" in notifications[0]["error"]


@pytest.mark.asyncio
async def test_calls_the_address_once_the_coordinator_hands_it_over(endpoint):
    target = await endpoint(lambda _p: {"contentType": "text/plain", "body": "live"})
    service, host = make_client({"__hkpMount": "hkp-mount://other/peer-1"})
    service.process(None, lambda _n: None)

    # This is what a coordinator does once the owner publishes its address.
    state = service.configure({"__hkpMount": target.url})
    assert state["__hkpMount"] == target.url

    service.process(None, lambda _n: None)
    assert (await next_push(host))["body"] == "live"


@pytest.mark.asyncio
async def test_lets_a_mount_win_over_a_typed_url(endpoint):
    # Naming a service is the more specific instruction, and its address is not
    # knowable when the board is written.
    mounted = await endpoint(lambda _p: {"body": "mounted"})
    typed = await endpoint(lambda _p: {"body": "typed"})
    service, host = make_client({"url": typed.url, "__hkpMount": mounted.url})

    service.process(None, lambda _n: None)
    await next_push(host)

    assert len(mounted.received) == 1
    assert typed.received == []


@pytest.mark.asyncio
async def test_waits_on_an_unresolved_mount_instead_of_falling_back_to_url(endpoint):
    # Falling back would silently call something the board did not ask for.
    typed = await endpoint(lambda _p: {"body": "typed"})
    service, host = make_client(
        {"url": typed.url, "__hkpMount": "hkp-mount://other/svc"}
    )

    service.process(None, lambda _n: None)
    await asyncio.sleep(0.05)

    assert typed.received == []
    assert host.pushed == []


@pytest.mark.asyncio
async def test_sends_a_string_as_text_and_an_object_as_json(endpoint):
    target = await endpoint()

    async def send(payload: Any) -> None:
        service, host = make_client({"__hkpMount": target.url, "method": "post"})
        service.process(payload, lambda _n: None)
        await next_push(host)

    await send("plain text")
    assert "text/plain" in target.received[0]["contentType"]
    assert target.received[0]["body"] == b"plain text"

    await send({"a": 1})
    assert target.received[1]["contentType"] == "application/json"
    assert target.received[1]["body"] == b'{"a": 1}'


@pytest.mark.asyncio
async def test_forwards_a_request_received_by_an_http_server_unchanged(endpoint):
    # {meta, body} is what http-server-subservices produces, so a request taken
    # in on one runtime can be sent on from another without reshaping it.
    target = await endpoint()
    service, host = make_client({"__hkpMount": target.url, "method": "post"})

    service.process(
        {"meta": {"contentType": "application/json"}, "body": {"forwarded": True}},
        lambda _n: None,
    )
    await next_push(host)

    assert target.received[0]["contentType"] == "application/json"
    assert target.received[0]["body"] == b'{"forwarded": true}'


@pytest.mark.asyncio
async def test_sends_no_body_on_get(endpoint):
    target = await endpoint()
    service, host = make_client({"__hkpMount": target.url})

    service.process({"ignored": True}, lambda _n: None)
    await next_push(host)

    assert target.received[0]["body"] == b""


@pytest.mark.asyncio
async def test_decodes_what_the_content_type_explains_and_keeps_the_rest(endpoint):
    target = await endpoint(
        lambda path: (
            {"contentType": "application/json", "body": '{"n": 1}'}
            if path == "/json"
            else {"contentType": "application/octet-stream", "body": bytes([1, 2, 3])}
        )
    )

    async def call(path: str) -> Any:
        service, host = make_client({"__hkpMount": target.url, "path": path})
        service.process(None, lambda _n: None)
        return await next_push(host)

    decoded = await call("/json")
    assert decoded["body"] == {"n": 1}
    assert "binary" not in decoded

    raw = await call("/blob")
    assert "body" not in raw
    assert raw["binary"] == bytes([1, 2, 3])


@pytest.mark.asyncio
async def test_passes_a_failure_status_on_as_a_result(endpoint):
    # The request completed; what the server said is the pipeline's business.
    target = await endpoint(lambda _p: {"status": 404, "body": "nope"})
    service, host = make_client({"__hkpMount": target.url})

    service.process(None, lambda _n: None)

    assert (await next_push(host))["meta"]["status"] == 404


@pytest.mark.asyncio
async def test_pushes_nothing_when_the_request_itself_fails():
    service, host = make_client(
        # Nothing is listening here.
        {"__hkpMount": "http://127.0.0.1:1/", "timeoutMs": 500}
    )
    notifications: list[Any] = []

    service.process(None, notifications.append)
    await asyncio.sleep(0.3)

    # No fabricated result travels down the pipeline.
    assert host.pushed == []
    assert any("error" in n for n in notifications)


@pytest.mark.asyncio
async def test_passes_input_through_when_bypassed():
    service, _host = make_client({"__hkpMount": "http://127.0.0.1:1/", "bypass": True})
    assert service.process({"untouched": True}, lambda _n: None) == {"untouched": True}


def test_reports_its_target_and_settings_in_state():
    service, _host = make_client(
        {
            "__hkpMount": "hkp-mount://node/http-1",
            "method": "POST",
            "path": "/upload",
            "headers": {"x-token": "abc", "dropped": 1},
        }
    )

    assert service.get_state() == {
        "url": "",
        "__hkpMount": "hkp-mount://node/http-1",
        "path": "/upload",
        # Stored lower case, as hkp-rt's http-client does and the shared UI expects.
        "method": "post",
        "headers": {"x-token": "abc"},
        "userAgent": "",
        "body": "",
        "timeoutMs": 10000,
        "bypass": False,
    }
