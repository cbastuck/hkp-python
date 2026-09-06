"""Secrets a runtime was given, and the one way to a value.

A board carries ``{{secret.<alias>}}`` references and never a value. The values
arrive separately — with the runtime's create payload, or on
``POST /runtimes/<id>/secrets`` — and are held here, apart from every service's
state. Nothing reads them back out: there is no route that returns one, they are
not in a serialized runtime, and a service obtains one only through
:meth:`SecretVault.resolve`, for one use, at the moment of that use.

That is what keeps a board safe to save. A service holds a reference, reports a
reference from ``get_state``, and the board it is serialized into never holds
anything else.

The format matches ``hkp-frontend/src/core/secrets.ts``, ``hkp-node/src/secrets.ts``
and ``hkp-rt/lib/include/secrets.h`` exactly: a board written against one runtime
has to open against another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

# ``{{secret.alias}}``, tolerating whitespace inside the braces. Dots are part of
# an alias rather than separators: ``secret.`` is a fixed prefix and ``}}``
# terminates, so ``{{secret.gmail.imap}}`` has exactly one reading.
REFERENCE = re.compile(r"\{\{\s*secret\.([A-Za-z0-9_.\-]+)\s*\}\}")


@dataclass
class SecretEntry:
    value: str
    #: Hosts this secret may be sent to. Empty means unconstrained — which is
    #: what an entry carrying no audience answers.
    audience: list[str] = field(default_factory=list)


@dataclass
class SecretRefusal:
    alias: str
    to: str
    audience: list[str]


@dataclass
class Resolved:
    value: Any
    missing: list[str]
    refused: list[SecretRefusal]


def destination_host(to: Any) -> str:
    """The host part of a destination.

    Callers hold destinations in whatever shape their own API uses — a request
    URL, a ``host:port`` pair, a bare hostname — and normalizing here is what
    keeps an audience a list of hosts rather than a list of spellings. Anything
    that does not yield a host answers ``""``.
    """
    if not isinstance(to, str):
        return ""
    trimmed = to.strip()
    if not trimmed:
        return ""
    candidate = trimmed if "://" in trimmed else f"hkp://{trimmed}"
    try:
        host = urlsplit(candidate).hostname
    except ValueError:
        return ""
    return (host or "").lower()


def audience_permits(audience: list[str], host: str) -> bool:
    """Whether an audience covers a host.

    An entry is either a host or a ``*.`` prefix standing for any subdomain of
    what follows it. The wildcard does not match the bare domain:
    ``*.example.com`` covers ``api.example.com`` and not ``example.com``, so
    widening one to the other stays a deliberate act.
    """
    if not audience:
        return True
    for entry in audience:
        allowed = entry.strip().lower()
        if not allowed:
            continue
        if allowed.startswith("*."):
            if host.endswith(allowed[1:]) and len(host) > len(allowed) - 1:
                return True
            continue
        if host == allowed:
            return True
    return False


def referenced_secrets(value: Any) -> list[str]:
    """Every alias a value refers to, however deeply it is nested."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for alias in REFERENCE.findall(node):
                if alias not in found:
                    found.append(alias)
            return
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    return found


class SecretVault:
    """The values one runtime was given, by alias."""

    def __init__(self) -> None:
        self._entries: dict[str, SecretEntry] = {}

    def replace(self, entries: dict[str, SecretEntry]) -> None:
        """Replaces everything held."""
        self._entries = dict(entries)

    def merge(self, entries: dict[str, SecretEntry]) -> None:
        """Adds or replaces individual entries, leaving the rest alone."""
        self._entries.update(entries)

    def aliases(self) -> list[str]:
        """The aliases held, for saying whether something is configured.

        Deliberately the only thing this answers about its contents.
        """
        return list(self._entries)

    def resolve(self, value: Any, to: str) -> Resolved:
        """A value with its references resolved, for one use.

        The result is transient: what a service passes to the call it is making,
        never something to assign back to its state. ``to`` is required and a
        caller that cannot name a destination gets nothing — every caller can,
        because a secret is used by sending it somewhere.

        An alias that is not held, or that may not go to this destination,
        becomes an empty string and is reported by name. Empty is what "not
        configured" already looks like to the code that takes a credential; the
        literal reference would be sent as one and fail far away, naming nothing.
        """
        missing: list[str] = []
        refused: list[SecretRefusal] = []
        host = destination_host(to)
        if not host:
            # No destination is not the same as no audience: without one there
            # is nothing to check a secret against, so nothing is released.
            return Resolved(value, referenced_secrets(value), refused)

        def substitute(match: re.Match[str]) -> str:
            alias = match.group(1)
            entry = self._entries.get(alias)
            if entry is None:
                if alias not in missing:
                    missing.append(alias)
                return ""
            if not audience_permits(entry.audience, host):
                if not any(r.alias == alias for r in refused):
                    refused.append(SecretRefusal(alias, host, list(entry.audience)))
                return ""
            return entry.value

        def walk(node: Any) -> Any:
            if isinstance(node, str):
                return REFERENCE.sub(substitute, node)
            if isinstance(node, dict):
                return {key: walk(child) for key, child in node.items()}
            if isinstance(node, list):
                return [walk(child) for child in node]
            return node

        return Resolved(walk(value), missing, refused)


@dataclass
class ResolvedCredential:
    #: The value to send, or None when there is none to send.
    value: Any
    #: Why there is none, or "" when there is.
    problem: str


def resolve_credential(
    vault: SecretVault | None, held: Any, to: str
) -> ResolvedCredential:
    """One credential, resolved for one use, or the reason there is none.

    What a service holds is either a reference or a literal, and either may be
    absent; the caller wants a value it can send or a sentence it can report.
    Separating those two outcomes here keeps every service that takes a
    credential from writing the same four branches.

    A literal is returned as it stands, which is what a runtime configured from
    a file holds. A reference needs a vault, and without one it resolves to
    nothing rather than being sent as its own text — a caller handed
    ``{{secret.…}}`` would offer it as a credential and fail somewhere far away.

    Takes a whole structure as readily as one string, because a credential is
    not always a field of its own: it can be one entry in a map of headers, or
    part of a larger string around it. On any failure it resolves nothing,
    rather than handing back a half-filled structure a caller might send anyway.
    """
    references = referenced_secrets(held)
    if not references:
        return ResolvedCredential(held, "")
    if vault is None:
        return ResolvedCredential(
            None, f"no secrets available to resolve {', '.join(references)}"
        )

    resolved = vault.resolve(held, to)
    if resolved.refused:
        refusal = resolved.refused[0]
        return ResolvedCredential(
            None, f"{refusal.alias} may not be sent to {refusal.to}"
        )
    if resolved.missing:
        return ResolvedCredential(
            None, f"no value stored for {', '.join(resolved.missing)}"
        )
    return ResolvedCredential(resolved.value, "")


def read_secrets_payload(value: Any) -> dict[str, SecretEntry]:
    """Reads a secrets payload off the wire.

    Tolerant of the short form — a bare string is a value with no audience —
    because that is what a client with nothing to say about destinations sends.
    Anything it cannot read is dropped rather than failing the request: a
    malformed entry costs one credential, and the service referencing it will
    report it as unavailable by name.
    """
    if not isinstance(value, dict):
        return {}
    entries: dict[str, SecretEntry] = {}
    for alias, entry in value.items():
        if isinstance(entry, str):
            entries[alias] = SecretEntry(entry)
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
            continue
        audience = entry.get("audience")
        entries[alias] = SecretEntry(
            entry["value"],
            [host for host in audience if isinstance(host, str)]
            if isinstance(audience, list)
            else [],
        )
    return entries
