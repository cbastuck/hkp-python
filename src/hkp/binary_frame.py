"""How a value that is not JSON crosses a coordinator connection.

A coordinator link carries JSON as text frames. A value holding bytes travels
as one binary frame instead:

    [ 4 bytes: header length, big-endian ][ header: UTF-8 JSON ][ payload ]

The header is the message that would have been sent as text, with the value
left out and a ``binary`` field in its place saying what the payload is:

    bytes            the payload is the value
    floatRingBuffer  little-endian float32 samples; ``id`` and ``ts`` in the header
    mixed            a dict with bytes under ``binary``; the header carries the
                     rest of the dict, the payload those bytes

The coordinator forwards the payload without reading it. Mirrors
hkp-node/src/coordinator/binaryFrame.ts; the two must agree.
"""
from __future__ import annotations

import json
import struct
from typing import Any

from .data import BinaryData, FloatRingBuffer

_LENGTH = struct.Struct(">I")
_BYTES = (bytes, bytearray, memoryview)


def _raw(value: Any) -> bytes | None:
    if isinstance(value, BinaryData):
        return bytes(value.data)
    if isinstance(value, _BYTES):
        return bytes(value)
    return None


def to_binary(value: Any) -> tuple[dict[str, Any], bytes] | None:
    """What a pipeline value travels as — ``(shape, payload)`` — or None when
    it is JSON and travels as text."""
    if isinstance(value, FloatRingBuffer):
        return (
            {"kind": "floatRingBuffer", "id": value.id, "ts": value.ts},
            bytes(value.samples),
        )
    raw = _raw(value)
    if raw is not None:
        return {"kind": "bytes"}, raw
    if isinstance(value, dict):
        carried = _raw(value.get("binary"))
        if carried is not None:
            rest = {key: item for key, item in value.items() if key != "binary"}
            return {"kind": "mixed", "json": rest}, carried
    return None


def from_binary(shape: dict[str, Any], payload: bytes) -> Any:
    """The pipeline value a payload stands for; see ``to_binary``."""
    kind = shape.get("kind")
    if kind == "floatRingBuffer":
        return FloatRingBuffer(
            samples=payload,
            id=int(shape.get("id") or 0),
            ts=int(shape.get("ts") or 0),
        )
    if kind == "mixed":
        rest = shape.get("json")
        return {**(rest if isinstance(rest, dict) else {}), "binary": payload}
    return BinaryData(payload)


def encode_frame(header: dict[str, Any], shape: dict[str, Any], payload: bytes) -> bytes:
    head = json.dumps({**header, "binary": shape}).encode("utf-8")
    return _LENGTH.pack(len(head)) + head + payload


def decode_frame(raw: bytes) -> tuple[dict[str, Any], dict[str, Any], bytes] | None:
    """``(header, shape, payload)``, or None for a frame that is not one of
    these. Nothing is raised at a peer."""
    if len(raw) < _LENGTH.size:
        return None
    (head_length,) = _LENGTH.unpack_from(raw, 0)
    end = _LENGTH.size + head_length
    if len(raw) < end:
        return None
    try:
        header = json.loads(raw[_LENGTH.size : end].decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(header, dict):
        return None
    shape = header.pop("binary", None)
    if not isinstance(shape, dict) or shape.get("kind") not in (
        "bytes",
        "floatRingBuffer",
        "mixed",
    ):
        return None
    return header, shape, raw[end:]
