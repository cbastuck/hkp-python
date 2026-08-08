from __future__ import annotations

# Service Documentation
# Service ID: hold
# Service Name: Hold
# Runtime: hkp-python
# Modes: none — a call carrying the property writes, every call reads
# Key Config: property
# IO: in=anything -> out={property: held value}, or None while nothing is held
# Arrays: a list carries no property, so it reads
# Binary: reads; holding non-JSON values is not supported yet
#
# Sample-and-hold: a pipeline entered from two sides — a producer that runs on
# its own schedule and a consumer that arrives whenever it arrives — needs the
# producer's latest value to survive between runs. Hold keeps it.
#
# One property name is the whole configuration. An input carrying that property
# is the producer, and its value replaces what is held. Every call, that one
# included, then emits the held value under the same property name — so the
# services after Hold receive the same shape whichever side called, and cannot
# tell the two apart. That is the point: the ordered list itself cannot say
# where a call came from, and with Hold in front of them nothing downstream
# needs to.
#
# A None held value is an empty one, the way None is nothing to pass on
# everywhere else, so a producer cannot hold None: an input carrying the
# property as None reads like any other.
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
        self._held: Any = None
        self._read_count = 0
        self._write_count = 0

        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def get_state(self) -> JsonRecord:
        return {
            "property": self._property,
            "held": _reportable(self._held),
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

        if config.get("action") == "clear":
            self._forget()

        state = self.get_state()
        self._notify(state)
        return state

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        # Nothing named is nothing to hold: an unconfigured Hold is a wire.
        if not self._property:
            return input

        incoming = _carried_value(input, self._property)
        if incoming is not _MISSING and incoming is not None:
            self._held = incoming
            self._write_count += 1
        else:
            self._read_count += 1

        self._notify(self.get_state())

        if self._held is None:
            return None

        return {self._property: self._held}

    def destroy(self) -> None:
        self._forget()

    # ── Private ──────────────────────────────────────────────────────────────

    def _forget(self) -> None:
        """Back to how the service started.

        The counts go with the value: they say how often each side has called
        for what is held now, and left running across a clear they would
        describe a value that is gone.
        """
        self._held = None
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


def _reportable(value: Any) -> Any:
    """The held value as it can be reported over REST.

    State travels as JSON, so a value that does not survive the trip is
    described rather than sent.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return f"[{type(value).__name__}]"
