from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from .auth import AllowedOrigins, AuthConfig, is_loopback_host
from .server import create_runtime_server


def _read_integer(value: str | None, fallback: int) -> int:
    if not value:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def _load_env_file() -> None:
    """Minimal .env loader (KEY=VALUE lines next to the project root), matching
    hkp-node's dotenv behaviour: real environment variables win."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_allowed_emails(value: str | None) -> list[str] | None:
    if not value:
        return None
    emails = [email.strip().lower() for email in value.split(",")]
    emails = [email for email in emails if email]
    return emails or None


def _parse_allowed_origins(value: str | None) -> AllowedOrigins:
    if not value or value.strip() == "*":
        return "*"
    origins = [origin.strip() for origin in value.split(",")]
    return [origin for origin in origins if origin]


def _is_dev_checkout() -> bool:
    """A development checkout runs from the source tree; an installed package
    runs from site-packages/dist-packages. Only the former may opt out of
    authentication (mirrors hkp-node's node_modules check)."""
    parts = Path(__file__).resolve().parts
    return "site-packages" not in parts and "dist-packages" not in parts


def resolve_server_auth_config(host: str) -> AuthConfig:
    """Fail closed: refuse to start without authentication unless the server is
    reachable only locally (loopback bind) or this is a source checkout that
    explicitly opts in via ALLOW_NO_AUTH=true. An installed package bound to a
    public interface can never reach no-auth mode."""
    domain = os.environ.get("AUTH0_DOMAIN")
    audience = os.environ.get("AUTH0_AUDIENCE")
    allowed_emails = _parse_allowed_emails(os.environ.get("ALLOWED_EMAILS"))
    if domain and audience:
        if allowed_emails:
            print(
                f"[hkp-python] Access restricted to {len(allowed_emails)} allowlisted email(s)."
            )
        return AuthConfig(
            mode="jwt",
            domain=domain,
            audience=audience,
            allowed_emails=allowed_emails,
        )

    # An allowlist without JWT auth cannot be enforced; starting anyway would
    # silently grant access to everyone the operator meant to exclude.
    if allowed_emails:
        print(
            "[hkp-python] ALLOWED_EMAILS is set but AUTH0_DOMAIN/AUTH0_AUDIENCE are not. "
            "The email allowlist can only be enforced with Auth0 configured — refusing to start.",
            file=sys.stderr,
        )
        sys.exit(1)

    if is_loopback_host(host):
        print(
            f"[hkp-python] No Auth0 configured; bound to loopback ({host}), so the server "
            "is reachable only from this machine. Running without authentication.",
            file=sys.stderr,
        )
        return AuthConfig(mode="none")

    if _is_dev_checkout() and os.environ.get("ALLOW_NO_AUTH") == "true":
        print(
            "[hkp-python] AUTH0_DOMAIN/AUTH0_AUDIENCE not set and ALLOW_NO_AUTH=true — running "
            f"with NO AUTHENTICATION on a non-loopback bind ({host}). Local development only; never expose this.",
            file=sys.stderr,
        )
        return AuthConfig(mode="none")

    print(
        f"[hkp-python] Refusing to start without authentication on a non-loopback bind ({host}). "
        "Set AUTH0_DOMAIN and AUTH0_AUDIENCE, bind to 127.0.0.1 for local-only use, or "
        "(from a checkout) set ALLOW_NO_AUTH=true.",
        file=sys.stderr,
    )
    sys.exit(1)


async def main() -> None:
    if not os.environ.get("SKIP_LOADING_ENV"):
        _load_env_file()

    port = _read_integer(os.environ.get("PORT"), 8080)
    host = os.environ.get("HOST", "0.0.0.0")
    external_host = os.environ.get("EXTERNAL_HOST", "127.0.0.1")
    allowed_origins = _parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))
    auth_config = resolve_server_auth_config(host)

    server = create_runtime_server(
        {
            "allowed_origins": allowed_origins,
            "auth": auth_config,
            "external_host": external_host,
            "host": host,
        }
    )

    address = await server.start(port, host)
    print(f"hkp-python listening on {address['base_url']}")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await server.stop()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
