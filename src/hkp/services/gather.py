from __future__ import annotations

# Service Documentation
# Service ID: gather
# Service Name: Gather
# Runtime: hkp-python
# Modes: none — the items as an array, or what the nested pipeline makes of them
# Key Config: pipeline, maxBytes, timeoutMs, bypass
# IO: in=one part a Spread sent on -> out=the whole group, once every part is in; None until then
# Arrays: what a group becomes, in the order it was spread
# Binary: an item may be bytes; joining bytes is the nested pipeline's to do
# MixedData: not native in runtime

"""The other half of Spread: waits until it has seen every part of a group and
turns them back into one value.

Spread sends each item as ``{item, group, index, total}``. Gather collects the
parts of one ``group``, puts each at its ``index``, and once it holds ``total``
of them the group is complete. Parts may arrive in any order; a part sent twice
counts once. Until then it passes nothing on. Anything that is not a part
passes through, so the same Gather sits on a board that sometimes sends a value
whole.

**What the group becomes is a pipeline, not a list of strategies**, the same
answer Tracks gives its reducer. With no pipeline the items travel on as a list;
with one, the pipeline is given ``{"group", "items"}`` and its answer is what
goes on.

**Collecting a group holds it in memory.** ``maxBytes`` bounds one group, and a
group still incomplete after ``timeoutMs`` is let go — both said in the log,
because a group that never arrives looks, from outside, exactly like nothing
having been sent.

Matches ``gather`` in hkp-node, hkp-rt and the browser runtime.
"""

import json
import time
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

GATHER_DESCRIPTOR = ServiceRegistryEntry(
    service_id="gather",
    service_name="Gather",
    version="v1",
    capabilities=["subservices"],
)

#: As much as the coordinator lets through in one frame by default.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_MS = 60_000


def is_part(value: Any) -> bool:
    """Whether a value is a part a Spread sent on, rather than one to pass through."""
    if not isinstance(value, dict) or "item" not in value:
        return False
    group, index, total = value.get("group"), value.get("index"), value.get("total")
    return (
        isinstance(group, str)
        and group != ""
        and _is_int(total)
        and total > 0
        and _is_int(index)
        and 0 <= index < total
    )


def _is_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return value.is_integer()
    return isinstance(value, int)


def _size_of(item: Any) -> int:
    """What a part costs to hold, near enough to bound memory by."""
    if isinstance(item, str):
        return len(item.encode("utf-8"))
    if isinstance(item, (bytes, bytearray)):
        return len(item)
    if item is None:
        return 0
    try:
        return len(json.dumps(item).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


class _Collecting:
    def __init__(self, total: int, started_at: float) -> None:
        self.items: dict[int, Any] = {}
        self.total = total
        self.bytes = 0
        self.started_at = started_at


class GatherService(SubService):
    service_id = GATHER_DESCRIPTOR.service_id
    service_name = GATHER_DESCRIPTOR.service_name
    version = GATHER_DESCRIPTOR.version
    capabilities = GATHER_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        # Before the base constructor, which configures from the board's state
        # and so reaches the override below.
        self._max_bytes = DEFAULT_MAX_BYTES
        self._timeout_ms = DEFAULT_TIMEOUT_MS
        self._last_error = ""
        self._collecting: dict[str, _Collecting] = {}
        #: Groups given up on, and when. The parts of a group still on their way
        #: when it was refused would otherwise start it again.
        self._refused: dict[str, float] = {}
        super().__init__(config, create_service)

    def configure(self, config: JsonRecord) -> JsonRecord:
        max_bytes = config.get("maxBytes")
        if isinstance(max_bytes, (int, float)) and not isinstance(max_bytes, bool) and max_bytes > 0:
            self._max_bytes = int(max_bytes)
        timeout_ms = config.get("timeoutMs")
        if isinstance(timeout_ms, (int, float)) and not isinstance(timeout_ms, bool) and timeout_ms > 0:
            self._timeout_ms = int(timeout_ms)
        # The pipeline, bypass and the editing commands belong to the base.
        super().configure(config)
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            **super().get_state(),
            "maxBytes": self._max_bytes,
            "timeoutMs": self._timeout_ms,
            "pending": len(self._collecting),
            "error": self._last_error,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if self._bypass or not is_part(input):
            return input

        now = time.monotonic()
        self._let_go_of_stale(now, notify)

        group = input["group"]
        index = int(input["index"])
        total = int(input["total"])
        item = input["item"]
        if group in self._refused:
            return None

        collecting = self._collecting.get(group)
        if collecting is None:
            collecting = _Collecting(total, now)
            self._collecting[group] = collecting

        if collecting.total != total:
            self._refuse(group, now, f"group {group} said {collecting.total} parts, then {total}", notify)
            return None
        if index in collecting.items:
            # Sent twice; the first one counts.
            return None

        collecting.bytes += _size_of(item)
        if collecting.bytes > self._max_bytes:
            self._refuse(group, now, f"group {group} is larger than {self._max_bytes} bytes", notify)
            return None

        collecting.items[index] = item
        notify(
            {
                "group": group,
                "received": len(collecting.items),
                "total": total,
                "pending": len(self._collecting),
            },
            None,
        )

        if len(collecting.items) < total:
            return None

        del self._collecting[group]
        items = [collecting.items[i] for i in range(total)]
        if not self._pipeline or not self._pipeline.list_services():
            return None if self._stop_propagation else items
        # The run that brought the last part is the one the whole arrives in;
        # the pipeline making it runs as a run of its own beneath it.
        result = self._pipeline.process(
            {"group": group, "items": items},
            lambda _n: None,
            child_run(self._host.current_context() if self._host else None),
        )
        return None if self._stop_propagation else result

    def destroy(self) -> None:
        self._collecting.clear()
        self._refused.clear()
        super().destroy()

    # ── Private ────────────────────────────────────────────────────────────────

    def _refuse(self, group: str, now: float, message: str, notify: NotifyCallback) -> None:
        self._collecting.pop(group, None)
        self._refused[group] = now
        self._last_error = message
        if self._host:
            self._host.log("error", "gather.refused", {"group": group, "message": message})
        notify({"error": message, "pending": len(self._collecting)}, None)

    def _let_go_of_stale(self, now: float, notify: NotifyCallback) -> None:
        """Lets go of groups that stopped arriving, and of refusals that are old."""
        timeout = self._timeout_ms / 1000
        for group, collecting in list(self._collecting.items()):
            if now - collecting.started_at > timeout:
                del self._collecting[group]
                received = len(collecting.items)
                self._last_error = (
                    f"group {group} timed out with {received} of {collecting.total} parts"
                )
                if self._host:
                    self._host.log(
                        "warn",
                        "gather.timed-out",
                        {"group": group, "received": received, "total": collecting.total},
                    )
                notify({"error": self._last_error, "pending": len(self._collecting)}, None)
        for group, at in list(self._refused.items()):
            if now - at > timeout:
                del self._refused[group]
