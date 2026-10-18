"""Where a board may make this server connect to.

A board names destinations — an HTTP Client's URL, an asset, a model's base
URL — and the server dials them. On a person's own machine that is the point.
On a server several people's boards run on it is not: this server's loopback,
its private network and a cloud provider's metadata endpoint
(``169.254.169.254``, which hands out the instance's credentials) are reachable
from here and are not a board's to reach.

So a server can be told it is hosted: with ``HKP_EGRESS=public``, a destination
is reached only when every address its name resolves to is a public internet
address. Unset, nothing is refused.

**Judged where the socket opens, on the address it opens to** — not on the
name a board wrote, which may resolve to something else by the time it is
dialled. That covers every hop of a redirect and every spelling of an address
too.

Services reach the network through what is here: ``client_session`` for
aiohttp, ``urlopen`` for urllib. ``tests/test_egress_sources.py`` fails on any
other way out. Connections the operator configured rather than a board — the
coordinator link, a model download from a fixed release address — are not a
board's destinations and are not judged here.

The same rule, range for range, is in hkp-node (``src/egress.ts``) and hkp-rt
(``lib/include/egress.h``).
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

import aiohttp

# What is not the public internet: this host, private networks, link-local
# (where the metadata endpoints are), shared address space, and what is not an
# address one connects to at all.
_NOT_PUBLIC = [
    ipaddress.ip_network(text)
    for text in (
        "0.0.0.0/8",
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "169.254.0.0/16",
        "192.0.0.0/24",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "fe80::/10",
        "fc00::/7",
        "ff00::/8",
    )
]

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


@dataclass(frozen=True)
class EgressPolicy:
    mode: str = "open"
    #: Addresses and ranges reached anyway, as the operator listed them.
    allow: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = field(
        default_factory=tuple
    )


OPEN = EgressPolicy()


def _address(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(text.strip("[]").split("%", 1)[0])
    except ValueError:
        return None


def _carried(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """An IPv6 address that carries an IPv4 one is that IPv4 address:
    ``::ffff:127.0.0.1`` is loopback."""
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped
        if address.sixtofour is not None:
            return address.sixtofour
        if address in _NAT64:
            return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address


def is_public_address(text: str) -> bool:
    """Whether ``text`` is a public internet address. Not an address: False."""
    address = _address(text)
    if address is None:
        return False
    address = _carried(address)
    return not any(address in network for network in _NOT_PUBLIC)


def read_egress_policy(env: Mapping[str, str]) -> EgressPolicy:
    """Reads ``HKP_EGRESS`` and ``HKP_EGRESS_ALLOW``.

    An entry of the list that is not an address or a range is an error rather
    than skipped: an operator who meant to allow something should learn now
    that nothing was. Names are not accepted, since an allowed name could be
    repointed.
    """
    mode = (env.get("HKP_EGRESS") or "").strip().lower()
    if mode in ("", "open"):
        return OPEN
    if mode != "public":
        raise ValueError(f'HKP_EGRESS must be "public" or unset, not "{env.get("HKP_EGRESS")}"')
    allow = []
    for entry in (env.get("HKP_EGRESS_ALLOW") or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            allow.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            raise ValueError(
                f'HKP_EGRESS_ALLOW: "{entry}" is not an address or a range '
                "(names are not accepted: an allowed name could be repointed)"
            ) from None
    return EgressPolicy(mode="public", allow=tuple(allow))


def may_reach(policy: EgressPolicy, text: str) -> bool:
    """Whether a destination at address ``text`` may be reached under ``policy``."""
    if policy.mode == "open":
        return True
    address = _address(text)
    if address is not None and any(
        address.version == network.version and address in network for network in policy.allow
    ):
        return True
    return is_public_address(text)


class EgressRefused(OSError):
    """A destination this server will not connect to, and why.

    An ``OSError``, so that the clients that report a failed connection report
    this one the same way — and keep it as the cause.
    """

    def __init__(self, host: str, address: str) -> None:
        message = (
            f"This server does not reach private addresses ({address})"
            if host == address
            else f"This server does not reach private addresses ({host} is {address})"
        )
        super().__init__(message)
        self.host = host
        self.address = address

    def __str__(self) -> str:
        return self.args[0]


_active: EgressPolicy = OPEN


def egress_policy() -> EgressPolicy:
    return _active


def set_egress_policy(policy: EgressPolicy) -> None:
    global _active
    _active = policy


def failure_reason(error: BaseException) -> str:
    """What to say about a failed connection: the refusal when this was one.

    The clients wrap what went wrong — aiohttp in a ``ClientConnectorError``,
    urllib in a ``URLError`` — and keep it as the cause or the reason.
    """
    refused = refusal_in(error)
    return str(refused) if refused else str(error)


def refusal_in(error: BaseException | None) -> EgressRefused | None:
    """The refusal an error is, or was caused by."""
    seen = 0
    current: Any = error
    while current is not None and seen < 6:
        if isinstance(current, EgressRefused):
            return current
        current = (
            getattr(current, "os_error", None)
            or getattr(current, "reason", None)
            or current.__cause__
            or current.__context__
        )
        if not isinstance(current, BaseException):
            return None
        seen += 1
    return None


def _check(host: str, addresses: list[str]) -> None:
    for address in addresses:
        if not may_reach(_active, address):
            raise EgressRefused(host, address)


# ── aiohttp ──────────────────────────────────────────────────────────────────


class _EgressConnector(aiohttp.TCPConnector):
    """A connector that refuses a destination any of whose addresses may not be
    reached.

    ``_resolve_host`` is where aiohttp turns a host — a name or an address
    literal — into the addresses it will connect to, for every connection and
    every redirect hop. It is private: ``tests/test_egress.py`` pins that a
    private address is refused through it, so an aiohttp that renamed it would
    fail there rather than let everything through.
    """

    async def _resolve_host(self, host: str, port: int, traces: Any = None) -> Any:
        resolved = await super()._resolve_host(host, port, traces=traces)
        _check(host, [entry["host"] for entry in resolved])
        return resolved


def client_session(**kwargs: Any) -> aiohttp.ClientSession:
    """An ``aiohttp.ClientSession`` for a board's destinations. Open: the
    default session. Hosted: one whose connector checks every connection."""
    if _active.mode != "open" and "connector" not in kwargs:
        kwargs["connector"] = _EgressConnector()
    return aiohttp.ClientSession(**kwargs)


# ── urllib ───────────────────────────────────────────────────────────────────


def _checked_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, **kwargs):
    """``socket.create_connection``, to an address the policy allows.

    Resolved here and connected to what was checked, so a second lookup cannot
    answer differently. TLS is wrapped around the socket afterwards by
    ``http.client``, against the name, so the certificate is still checked
    for the host that was asked for.
    """
    host, port = address
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    _check(host, [info[4][0] for info in infos])
    last: OSError | None = None
    for family, kind, proto, _, sockaddr in infos:
        sock = socket.socket(family, kind, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as error:
            last = error
            sock.close()
    raise last or OSError(f"could not connect to {host}")


class _CheckedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _checked_connection


class _CheckedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _checked_connection


class _CheckedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):  # noqa: D102 - urllib's own hook
        return self.do_open(_CheckedHTTPConnection, req)


class _CheckedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):  # noqa: D102 - urllib's own hook
        return self.do_open(_CheckedHTTPSConnection, req, context=self._context)


def urlopen(request: Any, timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT) -> Any:
    """``urllib.request.urlopen`` for a board's destinations. Open: exactly
    that. Hosted: every connection, redirects included, through the check."""
    if _active.mode == "open":
        return urllib.request.urlopen(request, timeout=timeout)
    opener = urllib.request.build_opener(_CheckedHTTPHandler, _CheckedHTTPSHandler)
    return opener.open(request, timeout=timeout)
