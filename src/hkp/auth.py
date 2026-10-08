"""Authentication & authorization — ported from hkp-node/src/auth.ts.

The same client (hkp-frontend, the hkp-node coordinator) talks to hkp-node and
hkp-python over the identical REST/WS protocol, so the auth surface is kept
byte-compatible:

- HTTP routes expect ``Authorization: Bearer <token>``.
- WebSocket upgrades accept the token either in the Authorization header
  (machine clients such as the coordinator) or as an ``?access_token=`` query
  parameter (browsers cannot set headers on a WS handshake), plus an Origin
  check against the allowed-origins list (CSWSH protection).
- Opaque session tokens are resolved locally *before* falling back to JWT
  verification.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

# "*" reflects any origin (only sensible for local/no-auth development).
AllowedOrigins = Union[str, list[str]]


@dataclass
class AuthenticatedUser:
    """Who a verified token speaks for.

    ``email`` is present only when the token carries one **and** says it is
    verified, and is normalised the way the lists it is compared with are. An
    address somebody merely typed while signing up is dropped rather than
    carried: everything that reads it — the server's allowlist, a run's caller
    — treats it as proof of who is asking.
    """

    sub: str
    email: str | None = None


def normalize_email(email: str) -> str:
    """An email as the lists it is compared with keep it."""
    return email.strip().lower()


#: Owner key used when authentication is disabled. Every request collapses into
#: this single tenant, which is exactly the pre-multi-tenancy behaviour.
ANONYMOUS_SUB = "anonymous"


def owner_key_of(user: AuthenticatedUser | None) -> str:
    """The tenant a request belongs to.

    Runtimes are namespaced by this key, so a runtime id is only ever resolved
    within the caller's own namespace. Matches the ``userId`` the coordinator
    uses, so a coordinator-provisioned runtime lands where the browser looks.
    """
    return user.sub if user else ANONYMOUS_SUB


@dataclass
class AuthConfig:
    """How requests are authenticated.

    - ``jwt``  — verify an Auth0 bearer token against the JWKS for
      ``domain``/``audience``. When ``allowed_emails`` is set, owning anything
      here additionally takes a **verified** ``email`` claim that is on the
      list. It gates who may own, not who may be identified.

      ``audience`` may list several accepted values. The frontend sends its
      id_token, whose ``aud`` is the Auth0 *client id* of whichever application
      signed the user in — and the web and native apps must be separate Auth0
      applications (only a SPA-type one can do the browser flows, only a
      Native-type one the RFC 8252 flow). One runtime serves users from both,
      so it accepts both client ids.
    - ``none`` — accept everything (no identity). Only ever resolved for a
      local development checkout that opts in via ALLOW_NO_AUTH, or a loopback
      bind (see resolve_server_auth_config in __main__.py).
    """

    mode: str  # "jwt" | "none"
    domain: str = ""
    audience: str | list[str] = ""
    allowed_emails: list[str] | None = None


# Resolves an opaque (non-JWT) bearer token to a principal, or None if unknown.
# Used for coordinator **session tokens**: short random strings the runtime
# itself mints (gated by a user JWT) and hands to the coordinator, so the
# coordinator can make long-lived machine calls on that user's behalf without a
# user JWT that would expire. The token resolves back to the user it was minted
# for, so there is no unscoped "service" superuser.
OpaqueTokenResolver = Callable[[str], Optional[AuthenticatedUser]]


def is_email_allowed(
    claims: dict[str, Any], allowed_emails: list[str] | None
) -> bool:
    """Email-allowlist gate applied after signature verification.

    Fail closed: when a list is configured, a token without an ``email`` claim
    — or with an unverified one — is rejected, because on tenants that allow
    self-signup an attacker could otherwise register an allowlisted address
    without owning it.
    """
    if allowed_emails is None:
        return True
    email = claims.get("email")
    if not isinstance(email, str) or claims.get("email_verified") is not True:
        return False
    return normalize_email(email) in allowed_emails


def may_own(user: AuthenticatedUser, allowed_emails: list[str] | None) -> bool:
    """The allowlist asked of somebody already identified. An identity carries
    an email only when it was verified, so this is ``is_email_allowed`` without
    the claims."""
    if allowed_emails is None:
        return True
    return bool(user.email) and user.email in allowed_emails


def identity_from_claims(claims: dict[str, Any]) -> AuthenticatedUser | None:
    """What a token's claims say about who it speaks for, or None without a
    ``sub``."""
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        return None
    email = claims.get("email")
    # Dropped, not refused: a person whose address is unverified can still sign
    # in wherever no list is asked.
    verified = (
        normalize_email(email)
        if isinstance(email, str) and claims.get("email_verified") is True
        else ""
    )
    return AuthenticatedUser(sub=sub, email=verified or None)


def is_loopback_host(host: str) -> bool:
    """True when the bind address is reachable only from the local machine.

    A loopback bind is itself an access-control boundary — nothing off-machine
    can connect — so running without authentication there is safe.
    """
    h = host.strip().lower()
    return h in ("localhost", "::1", "[::1]") or h.startswith("127.")


def is_origin_allowed(origin: str | None, allowed: AllowedOrigins) -> bool:
    """Cross-Site WebSocket Hijacking protection.

    Browsers always send an Origin header on the WS handshake, so a mismatched
    one is a cross-site attempt and is rejected. Non-browser clients (e.g. the
    coordinator) send no Origin; they are allowed through here and gated by
    the token check instead.
    """
    if allowed == "*":
        return True
    if origin is None:
        return True
    return origin in allowed


class Authenticator:
    """Resolved auth surface shared by HTTP and WebSocket entry points so both
    apply the exact same checks."""

    def __init__(
        self,
        config: AuthConfig,
        resolve_opaque_token: OpaqueTokenResolver | None = None,
    ) -> None:
        self._config = config
        self._resolve_opaque_token = resolve_opaque_token
        self._jwks_client: Any = None
        if config.mode == "jwt":
            from jwt import PyJWKClient

            self._jwks_client = PyJWKClient(
                f"https://{config.domain}/.well-known/jwks.json",
                cache_keys=True,
            )

    async def authorize_owner(self, token: str | None) -> AuthenticatedUser | None:
        """``identify_token``, then the server's allowlist: whether this person
        may own things here. What every route uses. The token is a raw string,
        from a WebSocket ``?access_token=`` query param or an Authorization
        bearer value."""
        if self._config.mode == "none":
            return AuthenticatedUser(sub=ANONYMOUS_SUB)
        if token and self._resolve_opaque_token is not None:
            # A session token was minted for somebody who had passed the
            # allowlist, and carries no email to ask it of again.
            opaque = self._resolve_opaque_token(token)
            if opaque is not None:
                return opaque
        user = await self.identify_token(token)
        if user is None or not may_own(user, self._config.allowed_emails):
            return None
        return user

    async def identify_token(self, token: str | None) -> AuthenticatedUser | None:
        """Who a raw token speaks for, and nothing about what they may do:
        signature, audience and ``sub``, with the email only when it is
        verified. Returns the principal or None.
        """
        if self._config.mode == "none":
            # Identity is irrelevant in no-auth mode; hand back a stable
            # principal so downstream code that reads `sub` still works.
            return AuthenticatedUser(sub="anonymous")

        if not token:
            return None

        # Session tokens are opaque and resolve locally without a network
        # round-trip, so check them before falling back to JWT verification.
        if self._resolve_opaque_token is not None:
            user = self._resolve_opaque_token(token)
            if user is not None:
                return user

        # JWKS fetch + signature verification are blocking (urllib + crypto);
        # keep them off the event loop.
        return await asyncio.get_running_loop().run_in_executor(
            None, self._verify_jwt, token
        )

    def _verify_jwt(self, token: str) -> AuthenticatedUser | None:
        import jwt

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            decoded = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._config.audience,
            )
        except Exception:
            return None

        return identity_from_claims(decoded)
