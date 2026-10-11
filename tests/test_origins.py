"""Which pages may call a server, and when a request carrying no credential is
let in for where it comes from. See hkp/origins.py.

The same rows are pinned in every runtime server (hkp-node, hkp-python,
hkp-rt): a board's runtimes must be equally closed to a foreign page whichever
of them they run on. The first half is the rule, row by row; the second sends
requests over a real socket to a listening server, with the headers a browser
would put on them — a page cannot choose its ``Origin``, its ``Sec-Fetch-Site``
or its ``Host``, which is what makes them worth checking.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.auth import AuthConfig
from hkp.origins import (
    AllowedOrigins,
    admits_without_credential,
    allows_origin,
    allows_origin_without_credential,
    is_known_host,
    is_loopback_origin,
    parse_allowed_origins,
)
from hkp.server import create_runtime_server
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR

# A non-resolvable domain keeps this offline: a request without a token is
# rejected before any key is fetched.
JWT_AUTH = AuthConfig(mode="jwt", domain="auth.invalid", audience="test-audience")
NO_AUTH = AuthConfig(mode="none")

EVIL = "https://evil.example"
# A runtime a page would create to get at the machine.
RUNTIME = {"id": "planted", "name": "planted", "services": []}


def admits(
    allowed: AllowedOrigins,
    origin: str | None = None,
    sec_fetch_site: str | None = None,
    host: str | None = "127.0.0.1:8080",
) -> bool:
    return admits_without_credential(origin, sec_fetch_site, host, allowed, [])


# ── The rule ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://localhost",
        "http://127.0.0.1:8555",
        "https://127.0.0.1:8443",
        "http://[::1]:3000",
        "HTTP://LOCALHOST:5173",
    ],
)
def test_a_page_served_from_this_machine_is_a_loopback_origin_on_any_port(origin):
    assert is_loopback_origin(origin)


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "http://127.evil.example",
        "http://evil.example:8887",
        "http://localhost:5173@evil.example",
        "http://192.168.1.20:5173",
        "file://localhost",
        "null",
        "",
    ],
)
def test_a_name_that_merely_starts_like_a_loopback_one_is_somebody_elses(origin):
    assert not is_loopback_origin(origin)


def test_with_nothing_said_the_apps_and_local_pages_may_call_and_no_site_may():
    allowed = parse_allowed_origins(None)

    assert admits(allowed, "saucer://embedded")
    assert admits(allowed, "hkp://app")
    assert admits(allowed, "https://appassets.androidplatform.net")
    assert admits(allowed, "http://localhost:5173")

    assert not admits(allowed, "https://evil.example")
    # The project's own website is a site like any other until somebody says so.
    assert not admits(allowed, "https://readymadeit.com")
    # What a sandboxed frame, a file:// page and a data: URL all send.
    assert not admits(allowed, "null")


def test_a_caller_that_is_not_a_browser_says_nothing_and_is_let_in():
    assert admits(parse_allowed_origins(""))
    # No Host either: HTTP/1.0, or a client that left it off.
    assert admits(parse_allowed_origins(""), host=None)


def test_a_list_names_exactly_who_may_call():
    allowed = parse_allowed_origins(" https://app.example , https://readymadeit.com ")

    assert admits(allowed, "https://app.example")
    assert admits(allowed, "https://readymadeit.com")
    assert admits(allowed, "HTTPS://APP.EXAMPLE")
    assert not admits(allowed, "https://evil.example")
    # Replaced, not extended: what was allowed unasked is not, once a list is given.
    assert not admits(allowed, "http://localhost:5173")
    assert not admits(allowed, "saucer://embedded")


def test_a_star_lets_any_page_call_with_a_credential_and_none_without():
    allowed = parse_allowed_origins("*")

    assert allows_origin("https://evil.example", allowed)
    assert not allows_origin_without_credential("https://evil.example", allowed)
    assert not admits(allowed, "https://evil.example")
    # What it reads as for such a request: nothing was said.
    assert admits(allowed, "http://localhost:5173")
    assert admits(allowed, "saucer://embedded")


def test_a_request_without_an_origin_is_refused_when_the_browser_says_cross_site():
    allowed = parse_allowed_origins("")

    assert not admits(allowed, None, "cross-site")
    assert not admits(allowed, None, "Cross-Site")
    # Typed into the address bar, or asked for by the server's own page.
    assert admits(allowed, None, "none")
    assert admits(allowed, None, "same-origin")
    # An allowed page is allowed however the browser classifies the request.
    assert admits(allowed, "http://localhost:5173", "cross-site")


def test_a_server_is_addressed_by_an_address_localhost_or_a_name_it_was_given():
    assert is_known_host("127.0.0.1:8080", [])
    assert is_known_host("localhost:8080", [])
    assert is_known_host("localhost", [])
    assert is_known_host("192.168.1.5:8080", [])
    assert is_known_host("[::1]:8080", [])
    assert is_known_host("[fe80::1]", [])
    assert is_known_host("py.example.com:8080", ["py.example.com"])
    assert is_known_host("Py.Example.com", ["py.example.com"])


@pytest.mark.parametrize(
    "host",
    [
        "attacker.example:8080",
        "127.0.0.1.attacker.example:8080",
        "localhost.attacker.example",
        "999.1.1.1",
        "1.2.3",
        "localhost:80@attacker.example",
    ],
)
def test_a_name_somebody_else_resolves_to_this_machine_is_not_one_it_answers_to(host):
    # DNS rebinding: the page is same-origin with the server as far as the
    # browser can tell, so there is no Origin to refuse — only this.
    assert not is_known_host(host, [])
    assert not is_known_host("attacker.example", ["py.example.com"])
    allowed = parse_allowed_origins("")
    assert not admits(allowed, None, "same-origin", "attacker.example:8080")
    # An allowed page does not make up for it.
    assert not admits(allowed, "http://localhost:5173", None, "attacker.example:8080")


# ── A listening server ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def listening():
    started = []
    sessions = []

    async def start(options: dict[str, Any]):
        server = create_runtime_server({"external_host": "127.0.0.1", **options})
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        session = aiohttp.ClientSession()
        sessions.append(session)
        return _Client(session, address["base_url"])

    yield start
    for session in sessions:
        await session.close()
    for server in started:
        await server.stop()


class _Reply:
    def __init__(self, status: int, headers: Any, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def json(self) -> Any:
        import json

        return json.loads(self.body)


class _Client:
    def __init__(self, session: aiohttp.ClientSession, base_url: str) -> None:
        self._session = session
        self.port = base_url.rsplit(":", 1)[1]
        # Whatever name the server gives out, it is this machine it listens on.
        self.base_url = f"http://127.0.0.1:{self.port}"

    async def send(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        json: Any = None,
        data: Any = None,
    ) -> "_Reply":
        async with self._session.request(
            method, f"{self.base_url}{path}", headers=headers, json=json, data=data
        ) as response:
            return _Reply(response.status, response.headers, await response.read())

    async def upgrade(self, headers: dict[str, str] | None = None) -> str:
        try:
            async with self._session.ws_connect(
                self.base_url.replace("http", "ws") + "/planted", headers=headers
            ) as ws:
                await ws.close()
                return "open"
        except aiohttp.ClientError:
            return "rejected"

    async def runtime_count(self) -> int:
        response = await self.send("GET", "/runtimes")
        assert response.status == 200
        return len(response.json()["runtimes"])


async def test_a_caller_that_is_not_a_browser_drives_the_server_as_before(listening):
    rt = await listening({"auth": NO_AUTH})

    listed = await rt.send("GET", "/runtimes")
    assert listed.status == 200
    assert "Access-Control-Allow-Origin" not in listed.headers

    assert (await rt.send("POST", "/runtimes", json=RUNTIME)).status == 200
    assert await rt.runtime_count() == 1
    assert await rt.upgrade() == "open"


async def test_a_foreign_page_cannot_create_a_runtime_without_a_preflight(listening):
    rt = await listening({"auth": NO_AUTH})

    # text/plain is one of the types a browser sends cross-origin without
    # asking first.
    import json as _json

    refused = await rt.send(
        "POST",
        "/runtimes",
        headers={"Origin": EVIL, "Content-Type": "text/plain"},
        data=_json.dumps(RUNTIME),
    )

    assert refused.status == 403
    # Nothing that lets the page read the answer, so it cannot tell this from
    # a server that is not running.
    assert refused.body == b""
    assert "Access-Control-Allow-Origin" not in refused.headers
    assert await rt.runtime_count() == 0


async def test_a_foreign_page_is_refused_whatever_it_asks_for(listening):
    rt = await listening({"auth": NO_AUTH})
    assert (await rt.send("POST", "/runtimes", json=RUNTIME)).status == 200

    evil = {"Origin": EVIL}
    attempts = [
        ("GET", "/runtimes", None),
        ("POST", "/runtimes", {"id": "second", "name": "second", "services": []}),
        ("POST", "/runtimes/planted/services", {"serviceId": "http-client", "instanceId": "out"}),
        ("POST", "/runtimes/planted", {"url": "http://169.254.169.254/"}),
        ("POST", "/runtimes/planted/session-token", None),
        ("DELETE", "/runtimes/planted", None),
    ]
    for method, path, body in attempts:
        refused = await rt.send(method, path, headers=evil, json=body)
        assert refused.status == 403, f"{method} {path}"
        assert "Access-Control-Allow-Origin" not in refused.headers

    assert await rt.runtime_count() == 1


async def test_a_foreign_pages_preflight_is_answered_with_nothing_that_lets_it_proceed(
    listening,
):
    rt = await listening({"auth": NO_AUTH})

    preflight = await rt.send(
        "OPTIONS",
        "/runtimes",
        headers={
            "Origin": EVIL,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert "Access-Control-Allow-Origin" not in preflight.headers


async def test_a_cross_site_request_sent_without_an_origin_is_refused(listening):
    rt = await listening({"auth": NO_AUTH})

    # What an <img> or <script> on another site's page produces.
    cross = await rt.send("GET", "/runtimes", headers={"Sec-Fetch-Site": "cross-site"})
    assert cross.status == 403
    # The address typed into the address bar.
    typed = await rt.send("GET", "/runtimes", headers={"Sec-Fetch-Site": "none"})
    assert typed.status == 200


async def test_a_page_that_resolved_its_own_name_to_this_machine_is_refused(listening):
    rt = await listening({"auth": NO_AUTH})

    # DNS rebinding: same-origin to the browser, so no Origin on a GET and the
    # page's own on a POST.
    rebound = await rt.send("GET", "/runtimes", headers={"Host": "attacker.example:8080"})
    assert rebound.status == 403
    posted = await rt.send(
        "POST",
        "/runtimes",
        headers={"Host": "attacker.example:8080", "Origin": "http://attacker.example:8080"},
        json=RUNTIME,
    )
    assert posted.status == 403
    named = await rt.send("GET", "/runtimes", headers={"Host": f"localhost:{rt.port}"})
    assert named.status == 200
    assert await rt.runtime_count() == 0


async def test_a_server_answers_to_the_name_it_was_given(listening):
    rt = await listening({"auth": NO_AUTH, "external_host": "py.example.com"})

    assert (await rt.send("GET", "/runtimes", headers={"Host": "py.example.com"})).status == 200
    assert (await rt.send("GET", "/runtimes", headers={"Host": "attacker.example"})).status == 403


async def test_the_apps_and_pages_served_from_this_machine_are_answered_and_told_so(
    listening,
):
    rt = await listening({"auth": NO_AUTH})

    for origin in [
        "http://localhost:5173",
        "http://127.0.0.1:8555",
        "saucer://embedded",
        "hkp://app",
        "https://appassets.androidplatform.net",
    ]:
        listed = await rt.send("GET", "/runtimes", headers={"Origin": origin})
        assert listed.status == 200, origin
        assert listed.headers["Access-Control-Allow-Origin"] == origin
        assert "Origin" in listed.headers["Vary"]

    preflight = await rt.send(
        "OPTIONS",
        "/runtimes",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert "Authorization" in preflight.headers["Access-Control-Allow-Headers"]


async def test_the_website_is_a_site_like_any_other_until_the_server_is_told_otherwise(
    listening,
):
    website = {"Origin": "https://readymadeit.com"}

    closed = await listening({"auth": NO_AUTH})
    assert (await closed.send("GET", "/runtimes", headers=website)).status == 403

    opened = await listening(
        {"auth": NO_AUTH, "allowed_origins": ["https://readymadeit.com"]}
    )
    listed = await opened.send("GET", "/runtimes", headers=website)
    assert listed.status == 200
    assert listed.headers["Access-Control-Allow-Origin"] == "https://readymadeit.com"


async def test_a_star_does_not_open_a_server_without_auth_to_every_page(listening):
    rt = await listening({"auth": NO_AUTH, "allowed_origins": "*"})

    refused = await rt.send("POST", "/runtimes", headers={"Origin": EVIL}, json=RUNTIME)
    assert refused.status == 403
    assert "Access-Control-Allow-Origin" not in refused.headers
    local = await rt.send("GET", "/runtimes", headers={"Origin": "http://localhost:5173"})
    assert local.status == 200
    assert await rt.runtime_count() == 0


async def test_a_list_replaces_who_may_call_unasked(listening):
    rt = await listening({"auth": NO_AUTH, "allowed_origins": ["https://app.example"]})

    assert (await rt.send("GET", "/runtimes", headers={"Origin": "https://app.example"})).status == 200
    assert (await rt.send("GET", "/runtimes", headers={"Origin": "http://localhost:5173"})).status == 403
    assert (await rt.send("GET", "/runtimes")).status == 200


async def test_a_foreign_page_cannot_open_a_runtimes_socket(listening):
    rt = await listening({"auth": NO_AUTH})
    assert (await rt.send("POST", "/runtimes", json=RUNTIME)).status == 200

    assert await rt.upgrade({"Origin": EVIL}) == "rejected"
    assert (
        await rt.upgrade(
            {"Origin": "http://attacker.example:8080", "Host": "attacker.example:8080"}
        )
        == "rejected"
    )
    assert await rt.upgrade({"Origin": "http://localhost:5173"}) == "open"
    assert await rt.upgrade() == "open"


async def test_a_mount_is_reached_by_anyone_as_it_is_meant_to_be(listening):
    rt = await listening({"auth": NO_AUTH})
    created = await rt.send(
        "POST",
        "/runtimes",
        json={
            "id": "hook",
            "name": "hook",
            "services": [
                {
                    "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                    "uuid": "endpoint",
                    "state": {
                        "bypass": False,
                        "mode": "process_on_session",
                        "pipeline": [],
                    },
                }
            ],
        },
    )
    assert created.status == 200
    state = await rt.send("GET", "/runtimes/hook/services/endpoint")
    mount = state.json()["__hkpMount"]
    assert "/hosted/" in mount

    # An outside caller holds no token and is on no list; the address is what
    # lets it in. Whatever the pipeline answers, it is not this server's refusal.
    path = "/hosted/" + mount.split("/hosted/", 1)[1]
    reached = await rt.send("POST", path, headers={"Origin": EVIL}, json={})
    assert reached.status != 403


async def test_with_auth_a_page_is_asked_for_a_token_wherever_it_is_from(listening):
    rt = await listening({"auth": JWT_AUTH})

    assert (await rt.send("GET", "/runtimes")).status == 401
    local = await rt.send("GET", "/runtimes", headers={"Origin": "http://localhost:5173"})
    assert local.status == 401

    # A page this server does not allow cannot read even that.
    foreign = await rt.send("GET", "/runtimes", headers={"Origin": EVIL})
    assert foreign.status == 401
    assert "Access-Control-Allow-Origin" not in foreign.headers
    assert await rt.upgrade({"Origin": EVIL}) == "rejected"


async def test_with_auth_and_a_star_any_page_may_read_that_it_needs_a_token(listening):
    rt = await listening({"auth": JWT_AUTH, "allowed_origins": "*"})

    asked = await rt.send("GET", "/runtimes", headers={"Origin": EVIL})
    assert asked.status == 401
    # What lets a page that does hold a token learn it was not accepted.
    assert asked.headers["Access-Control-Allow-Origin"] == EVIL
