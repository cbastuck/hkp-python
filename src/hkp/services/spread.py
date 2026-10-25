from __future__ import annotations

# Service Documentation
# Service ID: spread
# Service Name: Spread
# Runtime: hkp-python
# Modes: none
# Key Config: itemsFrom, bypass
# IO: in=a list, or a single item -> out=one {item, group, index, total} per item, each sent on by itself
# Arrays: the point of the service
# Binary: an item may be bytes; it travels untouched under `item`
# MixedData: not native in runtime

"""One value in, as many out as it has items.

A value leaving a runtime travels as one message, however many items it holds.
Spread sends each item down the rest of the board on its own — the services
after it, and the runtimes after this one — so each crosses as a message of its
own, and a Gather on the far side puts them back together.

It is not an Iterator: it holds no pipeline, and nothing comes back. It returns
None and continues the run itself, once per item — the pull pattern — so every
item goes on in the run that reached it.

**Each item goes on wrapped**, as ``{item, group, index, total}``. **A single
item is a list of one.** Items go on one at a time, in order.

Matches ``spread`` in hkp-node and ``hookup.to/service/spread`` in the browser.
"""

import uuid as _uuid_mod
from typing import Any

from ..types import (
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    ServiceConfiguration,
    ServiceRegistryEntry,
)
from .iterator import value_at

SPREAD_DESCRIPTOR = ServiceRegistryEntry(
    service_id="spread",
    service_name="Spread",
    version="v1",
    capabilities=[],
)


class SpreadService:
    service_id = SPREAD_DESCRIPTOR.service_id
    service_name = SPREAD_DESCRIPTOR.service_name
    version = SPREAD_DESCRIPTOR.version
    capabilities = SPREAD_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._bypass = False
        self._items_from = ""
        self._last_total = 0
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def get_state(self) -> JsonRecord:
        return {
            "bypass": self._bypass,
            "itemsFrom": self._items_from,
            "lastTotal": self._last_total,
        }

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("bypass"), bool):
            self._bypass = config["bypass"]
        if isinstance(config.get("itemsFrom"), str):
            self._items_from = config["itemsFrom"]
        return self.get_state()

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if self._bypass or not self._host:
            return input
        host = self._host
        run = host.current_context()

        items = self._items(input)
        if items:
            group = str(_uuid_mod.uuid4())
            total = len(items)
            for index, item in enumerate(items):
                output = host.process_from(
                    self.uuid,
                    {"item": item, "group": group, "index": index, "total": total},
                    lambda _n: None,
                    run,
                )
                # An item the rest of the pipeline stopped goes no further; the
                # others still do.
                if output is not None:
                    host.emit_result(output)
            self._last_total = total
            notify({"lastTotal": total}, None)
        # Every item has gone on already; there is nothing left to pass.
        return None

    def _items(self, input: Any) -> list[Any]:
        source = value_at(input, self._items_from) if self._items_from else input
        if source is None:
            return []
        return source if isinstance(source, list) else [source]
