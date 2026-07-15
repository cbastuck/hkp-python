"""Authentication tests — ported from hkp-node/tests/auth.test.ts.

A non-resolvable domain keeps these tests offline: every code path we exercise
either rejects before verifying a token (missing/blank bearer) or resolves an
opaque session token locally, so the JWKS endpoint is never contacted.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.auth import (
    AuthConfig,
    AuthenticatedUser,
    Authenticator,
    is_email_allowed,
    is_loopback_host,
    is_origin_allowed,
)
from hkp.server import SessionToken, create_runtime_server

JWT_AUTH = AuthConfig(mode="jwt", domain="auth.invalid", audience="test-audience")


# ── Fixtures / helpers ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start(options: dict[str, Any]):
        options = {"external_host": "127.0.0.1", **options}
        server = create_runtime_server(options)
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return server, address["base_url"]

    yield start
    for server in started:
        await server.stop()


async def _create_runtime(session: aiohttp.ClientSession, base_url: str) -> None:
    async with session.post(
        f"{base_url}/runtimes",
        json={"id": "rt-1", "name": "Python", "services": []},
    ) as resp:
        assert resp.status == 200


async def _ws_outcome(
    base_url: str,
    path: str,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    origin: str | None = None,
) -> str:
    """Returns "open" if the socket connected, "rejected" if the upgrade failed."""
    url = base_url.replace("http", "ws") + path
    if token:
        url += f"?access_token={token}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, headers=headers, origin=origin) as ws:
                await ws.close()
                return "open"
    except aiohttp.WSServerHandshakeError:
        return "rejected"
    except aiohttp.ClientError:
        return "rejected"


# ── Authentication ─────────────────────────────────────────────────────────────


async def test_rejects_http_requests_with_no_bearer_token_under_jwt_auth(servers):
    _, base_url = await servers({"auth": JWT_AUTH})
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/runtimes") as resp:
            assert resp.status == 401


async def test_resolves_opaque_session_tokens_before_jwt_verification():
    authenticator = Authenticator(
        JWT_AUTH,
        resolve_opaque_token=lambda token: (
            AuthenticatedUser(sub="auth0|user-1") if token == "sess-1" else None
        ),
    )
    assert await authenticator.verify_token("sess-1") == AuthenticatedUser(
        sub="auth0|user-1"
    )
    assert await authenticator.verify_token(None) is None


async def test_rejects_websocket_upgrades_without_valid_token_under_jwt_auth(servers):
    _, base_url = await servers({"auth": JWT_AUTH})
    # Auth is checked on the upgrade before the runtime is even looked up, so a
    # missing token is rejected regardless of the path.
    assert await _ws_outcome(base_url, "/rt-1") == "rejected"


async def test_mints_session_token_and_accepts_it_on_ws_authorization_header(servers):
    _, base_url = await servers({"auth": AuthConfig(mode="none")})
    async with aiohttp.ClientSession() as session:
        await _create_runtime(session, base_url)
        async with session.post(f"{base_url}/runtimes/rt-1/session-token") as resp:
            assert resp.status == 200
            body = await resp.json()
    assert isinstance(body["token"], str) and body["token"]

    # Coordinator-style: token in the Authorization header, not the URL.
    outcome = await _ws_outcome(
        base_url, "/rt-1", headers={"Authorization": f"Bearer {body['token']}"}
    )
    assert outcome == "open"


async def test_minted_session_token_authenticates_http_under_jwt_auth(servers):
    """Under JWT auth a minted opaque token must be accepted without ever
    contacting the JWKS endpoint (it resolves locally, before JWT verify)."""
    server, base_url = await servers({"auth": JWT_AUTH})
    # Mint route itself requires a JWT we can't produce offline, so bind the
    # token directly — this exercises resolution, not minting.
    server._session_tokens["sess-1"] = SessionToken(
        sub="auth0|user-1", runtime_id="rt-1"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/runtimes",
            headers={"Authorization": "Bearer sess-1"},
        ) as resp:
            assert resp.status == 200


async def test_rejects_websocket_upgrades_from_disallowed_origin(servers):
    _, base_url = await servers(
        {
            "auth": AuthConfig(mode="none"),
            "allowed_origins": ["https://app.example"],
        }
    )
    async with aiohttp.ClientSession() as session:
        await _create_runtime(session, base_url)
    outcome = await _ws_outcome(base_url, "/rt-1", origin="https://evil.example")
    assert outcome == "rejected"


def test_classifies_loopback_vs_public_bind_addresses():
    for host in ["127.0.0.1", "127.0.0.5", "::1", "[::1]", "localhost", "LOCALHOST"]:
        assert is_loopback_host(host)
    for host in ["0.0.0.0", "192.168.1.10", "10.0.0.4", "example.com"]:
        assert not is_loopback_host(host)


async def test_allows_everything_in_no_auth_mode(servers):
    _, base_url = await servers({"auth": AuthConfig(mode="none")})
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/runtimes") as resp:
            assert resp.status == 200
        await _create_runtime(session, base_url)
    assert await _ws_outcome(base_url, "/rt-1") == "open"


# ── JWT verification (offline: locally-signed RS256 + stubbed JWKS client) ────


class _StubJwksClient:
    def __init__(self, public_key: Any) -> None:
        self._key = public_key

    def get_signing_key_from_jwt(self, _token: str):
        return type("Key", (), {"key": self._key})()


def _jwt_authenticator(allowed_emails: list[str] | None = None):
    import jwt as pyjwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    config = AuthConfig(
        mode="jwt",
        domain="auth.invalid",
        audience="test-audience",
        allowed_emails=allowed_emails,
    )
    authenticator = Authenticator(config)
    authenticator._jwks_client = _StubJwksClient(private_key.public_key())

    def sign(claims: dict[str, Any]) -> str:
        return pyjwt.encode(claims, private_key, algorithm="RS256")

    return authenticator, sign


async def test_verifies_a_signed_jwt_and_extracts_the_principal():
    authenticator, sign = _jwt_authenticator()
    token = sign({"sub": "auth0|u1", "aud": "test-audience", "email": "a@x.com"})
    assert await authenticator.verify_token(token) == AuthenticatedUser(
        sub="auth0|u1", email="a@x.com"
    )


async def test_rejects_a_jwt_with_the_wrong_audience():
    authenticator, sign = _jwt_authenticator()
    token = sign({"sub": "auth0|u1", "aud": "other-audience"})
    assert await authenticator.verify_token(token) is None


async def test_enforces_the_email_allowlist_on_verified_jwts():
    authenticator, sign = _jwt_authenticator(allowed_emails=["alice@example.com"])
    allowed = sign(
        {
            "sub": "auth0|u1",
            "aud": "test-audience",
            "email": "alice@example.com",
            "email_verified": True,
        }
    )
    assert await authenticator.verify_token(allowed) is not None

    unverified = sign(
        {
            "sub": "auth0|u1",
            "aud": "test-audience",
            "email": "alice@example.com",
            "email_verified": False,
        }
    )
    assert await authenticator.verify_token(unverified) is None

    unlisted = sign(
        {
            "sub": "auth0|u2",
            "aud": "test-audience",
            "email": "mallory@evil.example",
            "email_verified": True,
        }
    )
    assert await authenticator.verify_token(unlisted) is None


# ── Email allowlist ────────────────────────────────────────────────────────────

ALLOWED = ["alice@example.com", "bob@example.com"]


def test_passes_everyone_when_no_allowlist_is_configured():
    assert is_email_allowed({}, None)
    assert is_email_allowed(
        {"email": "mallory@evil.example", "email_verified": True}, None
    )


def test_accepts_verified_listed_email_case_and_whitespace_insensitively():
    assert is_email_allowed(
        {"email": "alice@example.com", "email_verified": True}, ALLOWED
    )
    assert is_email_allowed(
        {"email": " Alice@Example.COM ", "email_verified": True}, ALLOWED
    )


def test_rejects_unlisted_emails():
    assert not is_email_allowed(
        {"email": "mallory@evil.example", "email_verified": True}, ALLOWED
    )


def test_fails_closed_on_missing_or_unverified_email_claim():
    # No email claim at all (e.g. an access token without the email scope).
    assert not is_email_allowed({}, ALLOWED)
    # Self-signup with someone else's address: email present but not verified.
    assert not is_email_allowed(
        {"email": "alice@example.com", "email_verified": False}, ALLOWED
    )
    assert not is_email_allowed({"email": "alice@example.com"}, ALLOWED)
    # Non-string junk in the claim.
    assert not is_email_allowed({"email": 42, "email_verified": True}, ALLOWED)


# ── Origin gate ────────────────────────────────────────────────────────────────


def test_origin_gate():
    assert is_origin_allowed("https://anything.example", "*")
    # Non-browser clients send no Origin and are gated by the token instead.
    assert is_origin_allowed(None, ["https://app.example"])
    assert is_origin_allowed("https://app.example", ["https://app.example"])
    assert not is_origin_allowed("https://evil.example", ["https://app.example"])


# ── Session token lifecycle ────────────────────────────────────────────────────


async def test_deleting_a_runtime_purges_its_session_tokens(servers):
    server, base_url = await servers({"auth": AuthConfig(mode="none")})
    async with aiohttp.ClientSession() as session:
        await _create_runtime(session, base_url)
        async with session.post(f"{base_url}/runtimes/rt-1/session-token") as resp:
            token = (await resp.json())["token"]
        assert token in server._session_tokens
        async with session.delete(f"{base_url}/runtimes/rt-1") as resp:
            assert resp.status == 200
        assert token not in server._session_tokens
