"""Which browser pages may call this server, and when a request that carries
no credential is let in.

Being reachable only from this machine is not the same as being called only by
this machine's owner: a page in their browser is a local caller too, and any
site can address ``127.0.0.1``. So a request admitted without a credential — a
server running without auth — must also not come from a foreign page, and must
address the server by a name the server knows. A browser states the first in
``Origin`` (or, on a request it sends without one, in ``Sec-Fetch-Site``) and
the second in ``Host``; a page cannot forge either.

A caller that is not a browser sends none of these, and is let in as before:
another process on this machine is not what this guards against.

The same rule, row for row, is in hkp-node (``src/origins.ts``) and hkp-rt
(``lib/include/origins.h``).
"""

from __future__ import annotations

import re
from typing import Sequence, Union

# The origins a server was told may call it, the way ``ALLOWED_ORIGINS`` gives
# them:
#
# - ``"default"`` — nothing was said: the origins Readymade's own apps run
#   from, and any page served from this machine;
# - a list — exactly those;
# - ``"*"`` — any origin, for a request that carries a credential. A request
#   that carries none is never let in on the strength of ``*``: for it, ``*``
#   reads as nothing was said.
AllowedOrigins = Union[str, list[str]]

DEFAULT_ORIGINS = "default"

# Where the Readymade apps load their pages from: desktop, iOS, Android.
_APP_ORIGINS = (
    "saucer://embedded",
    "hkp://app",
    "https://appassets.androidplatform.net",
)

_HOST_PORT = re.compile(r"^(\[[^\]]*\]|[^:]+)(?::(\d+))?$")
_IPV6_LITERAL = re.compile(r"^\[[0-9a-f:.]*:[0-9a-f:.]*\]$")
_HTTP_ORIGIN = re.compile(r"^https?://(.*)$")


def parse_allowed_origins(value: str | None) -> AllowedOrigins:
    """Reads ``ALLOWED_ORIGINS``."""
    trimmed = (value or "").strip()
    if not trimmed:
        return DEFAULT_ORIGINS
    if trimmed == "*":
        return "*"
    origins = [origin.strip().lower() for origin in trimmed.split(",")]
    return [origin for origin in origins if origin]


def _is_ipv4_literal(host: str) -> bool:
    """Four dot-separated numbers and nothing else. Strict, because the names
    this is asked about come from a request: ``127.0.0.1.example.com`` is a
    name somebody else resolves, not an address."""
    parts = host.split(".")
    return len(parts) == 4 and all(
        part.isascii() and part.isdigit() and len(part) <= 3 and int(part) <= 255
        for part in parts
    )


def _split_host(value: str) -> str | None:
    """The host of ``host[:port]`` — a ``Host`` header, or what follows an
    origin's scheme. None when it is neither."""
    match = _HOST_PORT.match(value)
    return match.group(1).lower() if match else None


def _is_loopback_name(host: str) -> bool:
    return (
        host == "localhost"
        or host == "[::1]"
        or (_is_ipv4_literal(host) and host.startswith("127."))
    )


def is_loopback_origin(origin: str) -> bool:
    """True for the origin of a page served from this machine, on any port."""
    match = _HTTP_ORIGIN.match(origin.lower())
    if not match:
        return False
    host = _split_host(match.group(1))
    return host is not None and _is_loopback_name(host)


def is_known_host(host_header: str | None, own_names: Sequence[str]) -> bool:
    """True when a ``Host`` header names the server by something a page cannot
    have arranged: an address, ``localhost``, or a name the server was told is
    its own.

    A page that resolves its own name to this machine (DNS rebinding) is
    same-origin with the server as far as the browser can tell, and sends no
    ``Origin`` to refuse — but the name it used is still in ``Host``. A request
    without the header was not sent by a browser.
    """
    value = (host_header or "").strip()
    if not value:
        return True
    host = _split_host(value)
    if host is None:
        return False
    if host == "localhost" or _is_ipv4_literal(host) or _IPV6_LITERAL.match(host):
        return True
    return any(name and name.lower() == host for name in own_names)


def allows_origin_without_credential(
    origin: str | None, allowed: AllowedOrigins
) -> bool:
    """Whether a page at ``origin`` may call the server with nothing but where
    it is calling from."""
    value = (origin or "").strip().lower()
    if not value:
        return False
    if isinstance(allowed, list):
        return value in allowed
    return value in _APP_ORIGINS or is_loopback_origin(value)


def allows_origin(origin: str | None, allowed: AllowedOrigins) -> bool:
    """Whether a page at ``origin`` may read what the server answers, and call
    it with a credential."""
    return allowed == "*" or allows_origin_without_credential(origin, allowed)


def admits_without_credential(
    origin: str | None,
    sec_fetch_site: str | None,
    host: str | None,
    allowed: AllowedOrigins,
    own_names: Sequence[str],
) -> bool:
    """Whether a request that carries no credential may be let in on the
    strength of where it comes from. See the top of this file."""
    if (origin or "").strip():
        if not allows_origin_without_credential(origin, allowed):
            return False
    elif (sec_fetch_site or "").strip().lower() == "cross-site":
        # A browser leaves ``Origin`` off a plain GET it makes for another
        # site's page — an image, a script — and says so here instead.
        return False
    return is_known_host(host, own_names)
