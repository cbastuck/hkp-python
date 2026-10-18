"""Where a board may make a hosted server connect to.

A server several people's boards run on must not dial its own loopback, its
private network or the cloud metadata endpoint on a board's say-so. What is
pinned here is that, with ``HKP_EGRESS=public``, those are refused at the
connection — whatever name led there, however many redirects it took — through
both ways this runtime makes requests (aiohttp and urllib), and that without it
nothing changes. The same cases are pinned for hkp-node in
``hkp-node/tests/egress.test.ts``.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web

from hkp.egress import (
    OPEN,
    failure_reason,
    is_public_address,
    read_egress_policy,
    refusal_in,
    set_egress_policy,
    client_session,
    urlopen,
)
from hkp.services.http_client import HttpClientService
from hkp.types import ServiceConfiguration


@pytest.fixture(autouse=True)
def open_again():
    yield
    set_egress_policy(OPEN)


def hosted(allow: str = "") -> None:
    set_egress_policy(read_egress_policy({"HKP_EGRESS": "public", "HKP_EGRESS_ALLOW": allow}))


@pytest_asyncio.fixture
async def serve():
    """Starts a server on a host, answering "reached" or a redirect."""
    runners: list[web.AppRunner] = []

    async def start(host: str, redirect_to: str | None = None) -> int:
        async def handler(_request: web.Request) -> web.StreamResponse:
            if redirect_to:
                raise web.HTTPFound(redirect_to)
            return web.Response(text="reached")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, 0)
        await site.start()
        runners.append(runner)
        return site._server.sockets[0].getsockname()[1]  # noqa: SLF001

    yield start
    for runner in runners:
        await runner.cleanup()


async def fetch(url: str) -> str:
    async with client_session() as session:
        async with session.get(url) as response:
            return await response.text()


def read(url: str) -> str:
    with urlopen(urllib.request.Request(url), timeout=5) as response:
        return response.read().decode()


# ── what counts as the public internet ───────────────────────────────────────


@pytest.mark.parametrize(
    "address", ["8.8.8.8", "1.1.1.1", "172.32.0.0", "100.128.0.0", "2606:4700:4700::1111"]
)
def test_public_addresses(address: str) -> None:
    assert is_public_address(address)


@pytest.mark.parametrize(
    "address",
    [
        "0.0.0.0",
        "127.0.0.1",
        "127.255.255.254",
        "10.0.0.1",
        "172.16.0.0",
        "172.31.255.255",
        "192.168.1.1",
        "100.64.0.1",
        "100.100.100.200",
        "169.254.169.254",
        "169.254.170.2",
        "192.0.0.192",
        "198.18.0.1",
        "224.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "fe80::1",
        "fc00::1",
        "fd00:ec2::254",
        "ff02::1",
        # IPv6 forms carrying an IPv4 address are judged by the IPv4 inside.
        "::ffff:127.0.0.1",
        "::ffff:7f00:1",
        "64:ff9b::a9fe:a9fe",
        "2002:7f00:1::",
        # Not an address at all.
        "localhost",
        "",
    ],
)
def test_not_public(address: str) -> None:
    assert not is_public_address(address)


def test_a_carried_public_address_is_public() -> None:
    assert is_public_address("::ffff:8.8.8.8")


# ── the policy a server is given ─────────────────────────────────────────────


def test_open_unless_told_otherwise() -> None:
    assert read_egress_policy({}) == OPEN
    assert read_egress_policy({"HKP_EGRESS": "open"}) == OPEN


def test_refuses_to_start_on_a_setting_it_does_not_understand() -> None:
    with pytest.raises(ValueError, match="HKP_EGRESS"):
        read_egress_policy({"HKP_EGRESS": "private"})
    # A name could be repointed at anything once allowed.
    with pytest.raises(ValueError, match="not an address or a range"):
        read_egress_policy({"HKP_EGRESS": "public", "HKP_EGRESS_ALLOW": "ollama.internal"})


# ── aiohttp ──────────────────────────────────────────────────────────────────


async def test_aiohttp_does_not_reach_a_private_address(serve: Any) -> None:
    port = await serve("127.0.0.1")
    hosted()
    for url in (f"http://127.0.0.1:{port}/", f"http://localhost:{port}/"):
        with pytest.raises(Exception) as caught:
            await fetch(url)
        assert "does not reach private addresses" in failure_reason(caught.value), url


async def test_aiohttp_reaches_what_was_allowed(serve: Any) -> None:
    port = await serve("127.0.0.1")
    hosted("127.0.0.1")
    assert await fetch(f"http://127.0.0.1:{port}/") == "reached"


async def test_aiohttp_is_refused_where_a_redirect_leads_somewhere_private(serve: Any) -> None:
    inside = await serve("::1")
    outside = await serve("127.0.0.1", redirect_to=f"http://[::1]:{inside}/")
    hosted("127.0.0.1")
    with pytest.raises(Exception) as caught:
        await fetch(f"http://127.0.0.1:{outside}/")
    assert "does not reach private addresses (::1)" in failure_reason(caught.value)


async def test_aiohttp_unchanged_when_not_hosted(serve: Any) -> None:
    port = await serve("127.0.0.1")
    assert await fetch(f"http://localhost:{port}/") == "reached"


# ── urllib ───────────────────────────────────────────────────────────────────


async def test_urllib_does_not_reach_a_private_address(serve: Any) -> None:
    port = await serve("127.0.0.1")
    hosted()
    for url in (f"http://127.0.0.1:{port}/", f"http://localhost:{port}/"):
        with pytest.raises(urllib.error.URLError) as caught:
            await asyncio.to_thread(read, url)
        assert refusal_in(caught.value) is not None, url


async def test_urllib_reaches_what_was_allowed(serve: Any) -> None:
    port = await serve("127.0.0.1")
    hosted("127.0.0.1")
    assert await asyncio.to_thread(read, f"http://127.0.0.1:{port}/") == "reached"


async def test_urllib_is_refused_where_a_redirect_leads_somewhere_private(serve: Any) -> None:
    inside = await serve("::1")
    outside = await serve("127.0.0.1", redirect_to=f"http://[::1]:{inside}/")
    hosted("127.0.0.1")
    with pytest.raises(urllib.error.URLError) as caught:
        await asyncio.to_thread(read, f"http://127.0.0.1:{outside}/")
    assert "(::1)" in failure_reason(caught.value)


async def test_urllib_unchanged_when_not_hosted(serve: Any) -> None:
    port = await serve("127.0.0.1")
    assert await asyncio.to_thread(read, f"http://localhost:{port}/") == "reached"


# ── a service ────────────────────────────────────────────────────────────────


class _Host:
    def secrets(self) -> Any:
        return None

    def spawn(self, coro: Any) -> bool:
        asyncio.get_running_loop().create_task(coro)
        return True

    def process_from(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def current_context(self) -> Any:
        return None

    def notify(self, _payload: Any, _instance_id: str) -> None:
        pass

    def emit_result(self, _output: Any) -> None:
        pass


async def test_the_http_client_says_why(serve: Any) -> None:
    port = await serve("127.0.0.1")
    hosted()
    service = HttpClientService(
        ServiceConfiguration(
            service_id="http-client",
            uuid="client-1",
            state={"url": f"http://127.0.0.1:{port}/", "method": "get"},
        )
    )
    service.set_host(_Host())
    said: list[dict[str, Any]] = []
    service.process({}, said.append)
    for _ in range(50):
        if any(entry.get("status") == 0 for entry in said):
            break
        await asyncio.sleep(0.02)
    failed = [entry for entry in said if entry.get("status") == 0]
    assert failed and failed[0]["error"] == (
        "This server does not reach private addresses (127.0.0.1)"
    )
