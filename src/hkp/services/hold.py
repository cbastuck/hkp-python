from __future__ import annotations

# Service Documentation
# Service ID: hold
# Service Name: Hold
# Runtime: hkp-python
# Modes: none — either a slot with a declared role, or a property that discriminates
# Key Config: slot + op, or property
# IO: in=anything -> out=the held value, or None while nothing is held
# Arrays: a list carries no property, so it reads
# Binary: holdable in a slot; a property cannot discriminate on one
#
# Sample-and-hold: a pipeline entered from two sides — a producer that runs on
# its own schedule and a consumer that arrives whenever it arrives — needs the
# producer's latest value to survive between runs. Hold keeps it.
#
# **Which side is calling can be said two ways**, and a board picks one.
#
# With a `slot`, the board says outright: `op` is `write` or `read`, and two
# Holds naming one slot are the two ends of it. Nothing inspects the value, so
# anything can be held — bytes, a document, None — and the two ends may sit in
# pipelines that never meet, which is what an endpoint's separate entry points
# are. Where the cells live is the host's to decide (RuntimeHost.slots): the
# service owning both pipelines, or failing that the runtime.
#
# With a `property` and no slot, the input says: an input carrying that property
# is the producer, its value replaces what is held, and every call — that one
# included — emits the held value under the same property name, so the services
# after Hold cannot tell the two sides apart. That is the older arrangement, and
# it is the only one available where the two sides share one pipeline, since
# there is nothing but the value to tell them apart. A None held value is an
# empty one, so a producer cannot hold None: an input carrying the property as
# None reads like any other.
#
# Mirrors hkp-node's and hkp-rt's `hold`.

import json
from typing import Any

from ..types import JsonRecord, NotifyCallback, RuntimeHost, ServiceConfiguration, ServiceRegistryEntry

HOLD_DESCRIPTOR = ServiceRegistryEntry(
    service_id="hold",
    service_name="Hold",
    version="v1",
    capabilities=[],
)

_MISSING = object()


class HoldService:
    service_id = HOLD_DESCRIPTOR.service_id
    service_name = HOLD_DESCRIPTOR.service_name
    version = HOLD_DESCRIPTOR.version
    capabilities = HOLD_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._host: RuntimeHost | None = None

        self._property = ""
        #: See the header: named, this is a cell the host owns rather than this one.
        self._slot = ""
        self._op = "read"
        #: What is held when no slot names somewhere else to hold it.
        self._own: Any = None
        self._read_count = 0
        self._write_count = 0

        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def get_state(self) -> JsonRecord:
        # Only the arrangement in use is reported. A state property a service
        # does not act on is one a board keeps and a reader has to discount, and
        # an omitted one is erased from the board the next time it is saved —
        # which is what should happen to the half of this service a board is not
        # using.
        which = {"slot": self._slot, "op": self._op} if self._slot else {"property": self._property}
        return {
            **which,
            "held": _reportable(self._read()),
            "readCount": self._read_count,
            "writeCount": self._write_count,
        }

    def configure(self, config: JsonRecord) -> JsonRecord:
        prop = config.get("property")
        if isinstance(prop, str):
            if prop != self._property:
                # What is held belongs to the property it was written for.
                self._forget()
            self._property = prop

        slot = config.get("slot")
        if isinstance(slot, str):
            if slot != self._slot:
                # A slot is an address, and what was held belongs to the old one
                # — but it belongs to whoever else is still reading it, so only
                # this service's own cell is cleared, never the host's.
                self._own = None
                self._read_count = 0
                self._write_count = 0
            self._slot = slot

        op = config.get("op")
        if op in ("read", "write"):
            self._op = op

        if config.get("action") == "clear":
            self._forget()

        state = self.get_state()
        self._notify(state)
        return state

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        if self._slot:
            return self._use_slot(input)

        # Nothing named is nothing to hold: an unconfigured Hold is a wire.
        if not self._property:
            return input

        incoming = _carried_value(input, self._property)
        if incoming is not _MISSING and incoming is not None:
            self._write(incoming)
            self._write_count += 1
        else:
            self._read_count += 1

        self._notify(self.get_state())

        held = self._read()
        if held is None:
            return None

        return {self._property: held}

    def destroy(self) -> None:
        # Only what this service holds itself. A slot belongs to the host, and
        # the other end of it outlives this one — a pipeline rebuilt while a
        # board is running destroys the services in it, and that must not empty
        # a cell the service on the other side is still answering from.
        self._own = None
        self._read_count = 0
        self._write_count = 0

    # ── Private ──────────────────────────────────────────────────────────────

    def _use_slot(self, input: Any) -> Any:
        """A call on a Hold whose role is declared rather than inferred.

        A write emits **its input unchanged**, so the pass it belongs to carries
        on as though the Hold were not there; a read emits what is held, **raw**,
        so it can be the whole of what a pipeline answers with. Neither looks at
        the value, which is what lets a slot hold what a property never could.
        """
        if self._op == "write":
            self._write(input)
            self._write_count += 1
            self._notify(self.get_state())
            return input

        self._read_count += 1
        self._notify(self.get_state())
        # Nothing held is nothing to pass on, the same as everywhere else — a
        # consumer that arrives before the producer has run stops here.
        return self._read()

    def _read(self) -> Any:
        """What is held, from wherever this Hold holds it."""
        if not self._slot:
            return self._own
        store = self._store()
        return store.get(self._slot) if store else self._own

    def _write(self, value: Any) -> None:
        store = self._store() if self._slot else None
        if store is not None:
            store.set(self._slot, value)
            return
        # No store to share through — a Hold outside any host that provides one
        # still holds, for itself alone, rather than dropping what it was given.
        self._own = value

    def _store(self) -> Any:
        # Optional on the host, as mounts are: a host that lends no cells is a
        # host this Hold keeps its value to itself in.
        slots = getattr(self._host, "slots", None)
        return slots() if callable(slots) else None

    def _forget(self) -> None:
        """Back to how the service started.

        The counts go with the value: they say how often each side has called
        for what is held now, and left running across a clear they would
        describe a value that is gone.
        """
        self._own = None
        if self._slot:
            store = self._store()
            if store is not None:
                store.set(self._slot, None)
        self._read_count = 0
        self._write_count = 0

    def _notify(self, payload: JsonRecord) -> None:
        if self._host:
            self._host.notify(payload, self.uuid)


def _carried_value(input: Any, prop: str) -> Any:
    """The value an input carries for the held property, if it carries one at
    all — anything else makes the call a read rather than a write."""
    if not isinstance(input, dict):
        return _MISSING
    return input.get(prop, _MISSING)


#: Beyond this, what is held is described rather than sent.
_REPORTABLE_LIMIT = 2048


def _reportable(value: Any) -> Any:
    """What is held, as something safe to put in state.

    State is read back into the board and sent to everyone watching, so what a
    Hold reports has to be worth carrying. A value that is bytes, or simply
    large — a rendered document, an audio buffer — is described instead of
    copied: the size is what a reader is looking for at that point, and the
    value itself is on its way to whatever asked for it regardless.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[{len(bytes(value))} bytes]"
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return f"[{type(value).__name__}]"
    if len(encoded) > _REPORTABLE_LIMIT:
        return f"[{type(value).__name__}, {len(encoded)} characters]"
    return value
