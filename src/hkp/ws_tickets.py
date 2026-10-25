"""Single-use tickets that open a WebSocket in place of a token.

A browser cannot set a header on a WebSocket handshake, so a token it
authenticated one with would have to ride in the address — and an address is
what proxies and access logs keep. So a browser first asks for a ticket
(``POST /ws-tickets``, with its token in the ``Authorization`` header like any
request) and opens the socket with ``?ticket=`` instead. A ticket opens one
socket, within seconds, as whoever asked for it, and is worth nothing once
written down.

The same in hkp-node (``src/wsTickets.ts``) and hkp-rt (``http/server.cpp``).
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from typing import Callable, Generic, TypeVar

# Long enough for a browser to open the socket it asked for, and no longer.
TICKET_TTL_SECONDS = 30.0
# Outstanding tickets kept; past this the oldest go first.
_MAX_OUTSTANDING = 10_000

Holder = TypeVar("Holder")


class WsTickets(Generic[Holder]):
    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._tickets: OrderedDict[str, tuple[Holder, float]] = OrderedDict()

    def mint(self, holder: Holder) -> str:
        """A ticket for ``holder``, good for one socket within the TTL."""
        self._prune()
        while len(self._tickets) >= _MAX_OUTSTANDING:
            self._tickets.popitem(last=False)
        ticket = secrets.token_urlsafe(32)
        self._tickets[ticket] = (holder, self._now() + TICKET_TTL_SECONDS)
        return ticket

    def redeem(self, ticket: str) -> Holder | None:
        """Who a ticket was minted for — once. Unknown, used or expired: None."""
        held = self._tickets.pop(ticket, None)
        if held is None:
            return None
        holder, expires = held
        return holder if expires >= self._now() else None

    def _prune(self) -> None:
        now = self._now()
        for ticket in [t for t, (_, expires) in self._tickets.items() if expires < now]:
            del self._tickets[ticket]
