"""What a call report says of the value a service was called with, or produced.

Every call is bracketed by a pair of reports (``call-process``,
``call-process-finished``), and each carries the value — what the flow
inspector shows of the call. That value travels wherever the reports do: to
every client watching the runtime and, on a deployed board, to its coordinator,
whose frame limit drops a message above 64 MiB and closes the connection a
little above that. A value that never leaves its runtime would still leave it
as a report, once for every service it passes through.

So a report carries a value only up to ``CALL_REPORT_LIMIT``. Above it, a
description takes its place — what it is, how large, and how it begins — which
the inspector shows and will not run again. The same limit, and the same
description, in every runtime: ``callReport.ts`` in hkp-node and hkp-frontend,
``call_report.h`` in hkp-rt.

Measuring stops once the limit is passed, so a value of a hundred megabytes
costs no more to judge than one of five. A size counted only that far is said
to be not ``exact``.

**And only while the board is captured.** What passes between the services of a
run is private; only what a board outputs is public. So a report carries its
value only in Capture mode — switched on by someone building the board, never
saved — and otherwise says only that the call happened and, when it ended,
whether it passed nothing on (``stopped``). See ``outgoing_payload``, and
``plans/TODO-CAPTURE-MODE.md``.
"""

from __future__ import annotations

import json
from typing import Any

from .data import NullData, TextData, UndefinedData

#: The most a report carries of one value: 5 MiB.
CALL_REPORT_LIMIT = 5 * 1024 * 1024

#: How much of a value too large to report its description begins with.
PREVIEW_LENGTH = 1024

#: Counted, and shown, as this much and no more: anything that is not data —
#: an instance, binary already summarised — is not looked inside.
_OPAQUE_SIZE = 16


def is_too_large(value: Any) -> bool:
    """Whether a reported value is the description of one too large to carry."""
    return isinstance(value, dict) and "__hkpTooLarge" in value


def outgoing_payload(payload: Any, capture: bool) -> Any:
    """A notification's payload as it may leave for anyone listening: a call
    report carrying its value only while ``capture`` is on, and then no more of
    it than ``reported_value`` allows; every other payload as it is."""
    if not isinstance(payload, dict):
        return payload
    internal = payload.get("__internal")
    if not isinstance(internal, dict) or internal.get("state") not in (
        "call-process",
        "call-process-finished",
    ):
        return payload
    report = {key: value for key, value in internal.items() if key != "data"}
    if capture:
        if "data" in internal:
            report["data"] = reported_value(internal["data"])
    elif report["state"] == "call-process-finished" and _passes_nothing(internal.get("data")):
        report["stopped"] = True
    return {**payload, "__internal": report}


def _passes_nothing(value: Any) -> bool:
    return value is None or isinstance(value, (NullData, UndefinedData))


def reported_value(value: Any, limit: int = CALL_REPORT_LIMIT) -> Any:
    """``value`` as a report may carry it: itself, or its description."""
    text = value.text if isinstance(value, TextData) else value
    if isinstance(text, str):
        size = _utf8_length_up_to(text, limit)
        # A text longer than the limit in characters is not counted in bytes.
        return (
            _too_large("string", size, len(text) <= limit, _head(text))
            if size > limit
            else value
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        size = value.nbytes if isinstance(value, memoryview) else len(value)
        return _too_large("binary", size, True) if size > limit else value
    if not isinstance(value, (dict, list, tuple)):
        return value
    size = _size_up_to(value, limit)
    return _too_large("json", size, False, _preview(value)) if size > limit else value


def _too_large(kind: str, size: int, exact: bool, preview: str | None = None) -> dict:
    described: dict[str, Any] = {"type": kind, "bytes": size, "exact": exact}
    if preview is not None:
        described["preview"] = preview
    return {"__hkpTooLarge": described}


def _utf8_length_up_to(text: str, budget: int) -> int:
    """The UTF-8 length of ``text``, counted no further than ``budget``: a text
    with more characters than that is past it already, whatever they encode to,
    and is answered with its character count."""
    if len(text) > budget:
        return len(text)
    return len(text.encode("utf-8", errors="replace"))


def _size_up_to(value: Any, budget: int) -> int:
    """What ``value`` takes as compact JSON, near enough, counted until it
    passes ``budget``. A container's own brackets and commas are counted before
    what is in it, so a list of a million items is past a small budget without
    one of them being visited."""
    total = 0
    pending: list[Any] = [value]
    while pending and total <= budget:
        item = pending.pop()
        if isinstance(item, str):
            total += _utf8_length_up_to(item, budget - total) + 2
        elif isinstance(item, bool):
            total += 4 if item else 5
        elif isinstance(item, (int, float)):
            total += len(str(item))
        elif item is None:
            total += 4
        elif isinstance(item, (bytes, bytearray)):
            total += len(item)
        elif isinstance(item, TextData):
            total += _utf8_length_up_to(item.text, budget - total) + 2
        elif isinstance(item, (list, tuple)):
            total += 2 + max(0, len(item) - 1)
            if total > budget:
                break
            pending.extend(item)
        elif isinstance(item, dict):
            total += 2
            for key, entry in item.items():
                total += _utf8_length_up_to(str(key), budget - total) + 4
                if total > budget:
                    break
                pending.append(entry)
        else:
            total += _OPAQUE_SIZE
    return total


def _head(text: str, length: int = PREVIEW_LENGTH) -> str:
    """The beginning of ``text``."""
    return text[:length]


def _preview(value: Any) -> str:
    """How ``value`` begins, as compact JSON of at most ``PREVIEW_LENGTH``
    characters and an ellipsis — written only that far, never the whole of it."""
    out: list[str] = []
    written = 0

    def put(text: str) -> None:
        nonlocal written
        out.append(text)
        written += len(text)

    # False once the preview is full: nothing more is written, not even the
    # brackets that would close what was begun.
    def write(item: Any) -> bool:
        if written >= PREVIEW_LENGTH:
            return False
        if isinstance(item, TextData):
            item = item.text
        if isinstance(item, str):
            put(json.dumps(_head(item, PREVIEW_LENGTH - written), ensure_ascii=False))
        elif isinstance(item, (bytes, bytearray)):
            put(f'"<{len(item)} bytes>"')
        elif isinstance(item, (list, tuple)):
            put("[")
            for i, entry in enumerate(item):
                if i > 0:
                    put(",")
                if not write(entry):
                    return False
            put("]")
        elif isinstance(item, dict):
            put("{")
            for i, (key, entry) in enumerate(item.items()):
                put(("," if i > 0 else "") + json.dumps(str(key), ensure_ascii=False) + ":")
                if not write(entry):
                    return False
            put("}")
        elif item is None or isinstance(item, (bool, int, float)):
            put(json.dumps(item))
        else:
            put(f'"<{type(item).__name__}>"')
        return written < PREVIEW_LENGTH

    complete = write(value)
    text = "".join(out)
    return text if complete else f"{_head(text)}…"
