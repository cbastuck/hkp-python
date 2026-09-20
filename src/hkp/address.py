"""Addressing a service inside a scope.

A runtime's services are a flat list, and a uuid names one of them. A service
holding a pipeline of its own — a SubService, an endpoint, Tracks — has
services inside it that the flat list does not reach, and until now nothing
outside could name one. A **scoped address** names it by the path through the
services containing it::

    read.kept-articles          the `kept-articles` inside the `read` scope
    read.list.feed-doc          two levels down

**The separator is a dot, and that is forced rather than chosen.** A service
address is carried in a URL path segment (``/runtimes/{id}/services/{id}``),
which a slash would split; the same constraint already picked a dot for the
separator between a unit's name and its runtime ids.

**A flat uuid is tried before the path is walked**, so a board whose service
uuid happens to contain a dot keeps resolving to that service rather than being
read as an address into something else.
"""

from __future__ import annotations

from typing import Any

ADDRESS_SEPARATOR = "."


def split_address(address: str) -> list[str]:
    """``["read", "kept-articles"]`` for ``"read.kept-articles"``."""
    return [part for part in address.split(ADDRESS_SEPARATOR) if part]


def join_address(owner: str, instance_id: str) -> str:
    """The address of ``instance_id`` inside ``owner``."""
    return f"{owner}{ADDRESS_SEPARATOR}{instance_id}" if owner else instance_id


def is_scoped_address(address: str) -> bool:
    """True where an address names something nested rather than a flat uuid."""
    return len(split_address(address)) > 1


def descend(service: Any, segments: list[str]) -> Any | None:
    """Walk the rest of an address down from a service already resolved.

    Answers None at the first segment nothing claims, rather than the nearest
    service it did reach: a partial address is a miss, not a match.
    """
    current = service
    for segment in segments:
        find_nested = getattr(current, "find_nested", None)
        current = find_nested(segment) if find_nested else None
        if current is None:
            return None
    return current
