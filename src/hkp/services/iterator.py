from __future__ import annotations

# Service Documentation
# Service ID: iterator
# Service Name: Iterator
# Runtime: hkp-python
# Modes: one pass of the nested pipeline per item
# Key Config: itemsFrom, limit, pipeline, bypass
# IO: in=an array, or a single item -> out=the results, or nothing at all
# Arrays: the point of the service
# Binary: passed to the nested pipeline untouched
# MixedData: not native in runtime

"""Iteration as a service, so a board can express it without a service having
to grow its own loop.

A pipeline pass carries one value, which is a problem for anything that
produces several: a transcription returns its segments, a query its rows.
Iterator runs its nested pipeline once per item and collects the answers, so
iterating is a service you can see in the board rather than a loop buried in
whichever service happened to need it first.

**A single item is an array of one**, so a board that grows from one value to
several does not change shape.

What the nested pipeline returns for an item is collected, and an item whose
pipeline **stopped** contributes nothing — so a filter-shaped sub-pipeline makes
this a filter. Nothing collected at all stops the outer pipeline here.

**One item failing does not end the loop.** Failures are counted and reported
rather than raised.

Matches ``iterator`` in hkp-node, hkp-rt and the browser runtime.
"""

from typing import Any

from ..runtime import child_run
from ..types import (
    JsonRecord,
    NotifyCallback,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
)
from .sub_service import SubService

ITERATOR_DESCRIPTOR = ServiceRegistryEntry(
    service_id="iterator",
    service_name="Iterator",
    version="v1",
    # Same as SubService: this service holds a pipeline, and the board's UI
    # needs to know that to let anyone look inside it.
    capabilities=["subservices"],
)


def value_at(input: Any, path: str) -> Any:
    """The value at a dotted path, or None where the path does not lead."""
    current = input
    for step in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(step)
    return current


class IteratorService(SubService):
    service_id = ITERATOR_DESCRIPTOR.service_id
    service_name = ITERATOR_DESCRIPTOR.service_name
    version = ITERATOR_DESCRIPTOR.version
    capabilities = ITERATOR_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        # Before the base constructor, which configures from the board's state
        # and so reaches the override below.
        self._items_from = ""
        self._limit = 0
        self._last_items = 0
        self._last_failed = 0
        super().__init__(config, create_service)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("itemsFrom"), str):
            self._items_from = config["itemsFrom"]
        limit = config.get("limit")
        if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit >= 0:
            self._limit = int(limit)
        # The pipeline, bypass and the editing commands belong to the base.
        super().configure(config)
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            **super().get_state(),
            "itemsFrom": self._items_from,
            "limit": self._limit,
            "lastItems": self._last_items,
            "lastFailed": self._last_failed,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if self._bypass or not self._pipeline or not self._pipeline.list_services():
            # Nothing to run the items through: what an empty SubService does.
            return None if self._stop_propagation else input

        items = self._items(input)
        parent = self._host.current_context() if self._host else None
        results: list[Any] = []
        failed = 0

        for item in items:
            try:
                # A run of its own per item, descended from the run that reached
                # the Iterator, so a log answers "what happened to this one".
                result = self._pipeline.process(item, lambda _n: None, child_run(parent))
                if result is not None:
                    results.append(result)
            except Exception as error:  # noqa: BLE001 - counted and reported
                failed += 1
                if self._host:
                    self._host.log(
                        "error",
                        "service.failed",
                        {"message": f"iterator item failed: {error}"},
                    )

        self._last_items = len(items)
        self._last_failed = failed
        notify({"items": len(items), "results": len(results), "failed": failed}, None)

        if self._stop_propagation:
            return None
        return results if results else None

    def _items(self, input: Any) -> list[Any]:
        source = value_at(input, self._items_from) if self._items_from else input
        if source is None:
            return []
        items = source if isinstance(source, list) else [source]
        return items[: self._limit] if self._limit > 0 else items
