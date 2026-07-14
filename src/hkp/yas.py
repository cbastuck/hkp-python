from __future__ import annotations

"""YAS binary wire format — Python port.

Reference implementations:
  hkp-frontend/src/runtime/rest/Message.ts   (TypeScript, hand-rolled)
  hkp-rt/lib/src/types/message.cpp           (C++, yas library)

Frame layout (all integers little-endian):

  7 bytes   YAS header  b"yas" + 4 flag/version bytes (validated on 'yas' only)
  uint16    message purpose
  uint16    data type id (DataTypeId)
  uint64    sender length
  bytes     sender (ascii)
  ...       payload, type-dependent:

  FloatRingBuffer:  nested 7-byte YAS header
                    + type id            <- DIALECT: uint16 (C++ writer) or
                                            uint32 (TS writer); auto-detected
                                            on read via total-length check
                    + uint64 numSamples
                    + numSamples * 4 bytes of float32 samples
                    + uint32 id
                    + uint64 ts
  JSON / String:    nested 7-byte YAS header + uint64 byteLength + utf-8 bytes
  BinaryData:       raw bytes to end of frame (no nested header)
  Null/Undefined:   single ignored byte

On write this codec emits the uint16 ring-buffer dialect, which is what the
TypeScript deserializer expects — every runtime hop is mediated by the
frontend, so the frontend is our only YAS peer in practice.
"""

import json
import struct
from dataclasses import dataclass
from typing import Any

from .data import (
    BinaryData,
    DataTypeId,
    FloatRingBuffer,
    NullData,
    TextData,
    UndefinedData,
    get_data_type_id,
)


class MessagePurpose:
    NOTIFICATION = 0
    RESULT = 1
    RESULT_AWAITING_RESPONSE = 2
    RESULT_WITH_REQUEST_ID = 3


# Matches the header bytes the TypeScript writer emits ("yas" 0 0 1 7); the
# C++ yas library accepts these, and readers only validate the "yas" magic.
YAS_HEADER = b"yas0017"


@dataclass
class Message:
    purpose: int
    sender: str
    data: Any


class YasError(ValueError):
    pass


# ── Serialization ──────────────────────────────────────────────────────────────


def serialize_message(
    data: Any, sender: str = "", purpose: int = MessagePurpose.NOTIFICATION
) -> bytes:
    sender_bytes = sender.encode("ascii")
    header = (
        YAS_HEADER
        + struct.pack("<HHQ", purpose, get_data_type_id(data), len(sender_bytes))
        + sender_bytes
    )
    return header + _serialize_payload(data)


def _serialize_payload(data: Any) -> bytes:
    if isinstance(data, FloatRingBuffer):
        return (
            YAS_HEADER
            + struct.pack("<HQ", DataTypeId.FLOAT_RING_BUFFER, data.num_samples)
            + data.samples
            + struct.pack("<IQ", data.id, data.ts)
        )
    if isinstance(data, (dict, list)):
        payload = json.dumps(data).encode("utf-8")
        return YAS_HEADER + struct.pack("<Q", len(payload)) + payload
    if isinstance(data, (TextData, str)):
        text = data.text if isinstance(data, TextData) else data
        payload = text.encode("utf-8")
        return YAS_HEADER + struct.pack("<Q", len(payload)) + payload
    if isinstance(data, BinaryData):
        return data.data
    if isinstance(data, (NullData, UndefinedData)) or data is None:
        return b"\x00"
    raise YasError(f"serialize_message: unsupported data type: {type(data)!r}")


# ── Deserialization ────────────────────────────────────────────────────────────


def deserialize_message(buffer: bytes | bytearray | memoryview) -> Message:
    buf = bytes(buffer)
    offset = _parse_yas_header(buf, 0)

    if len(buf) < offset + 12:
        raise YasError("deserialize_message: truncated message header")
    purpose, data_type, sender_length = struct.unpack_from("<HHQ", buf, offset)
    offset += 12

    if len(buf) < offset + sender_length:
        raise YasError("deserialize_message: truncated sender")
    sender = buf[offset : offset + sender_length].decode("ascii", errors="replace")
    offset += sender_length

    return Message(purpose=purpose, sender=sender, data=_deserialize_payload(buf, offset, data_type))


def _deserialize_payload(buf: bytes, offset: int, data_type: int) -> Any:
    if data_type == DataTypeId.FLOAT_RING_BUFFER:
        return _deserialize_ring_buffer(buf, offset)
    if data_type == DataTypeId.JSON:
        return json.loads(_deserialize_sized_text(buf, offset))
    if data_type == DataTypeId.STRING:
        return TextData(_deserialize_sized_text(buf, offset))
    if data_type == DataTypeId.BINARY:
        return BinaryData(buf[offset:])
    if data_type == DataTypeId.NULL:
        return NullData()
    if data_type == DataTypeId.UNDEFINED:
        return UndefinedData()
    raise YasError(f"deserialize_message: unsupported data type id: {data_type}")


def _deserialize_sized_text(buf: bytes, offset: int) -> str:
    offset = _parse_yas_header(buf, offset)
    if len(buf) < offset + 8:
        raise YasError("deserialize_message: truncated text payload")
    (length,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    if len(buf) < offset + length:
        raise YasError("deserialize_message: truncated text payload body")
    return buf[offset : offset + length].decode("utf-8")


def _deserialize_ring_buffer(buf: bytes, offset: int) -> FloatRingBuffer:
    offset = _parse_yas_header(buf, offset)

    # The type-id field width differs between writers: the C++ runtime writes
    # uint16, the TypeScript frontend writes uint32. Detect which dialect fits
    # by checking that the implied sample count fills the frame exactly.
    for type_width, type_fmt in ((2, "<H"), (4, "<I")):
        pos = offset
        if len(buf) < pos + type_width + 8:
            continue
        (type_id,) = struct.unpack_from(type_fmt, buf, pos)
        if type_id != DataTypeId.FLOAT_RING_BUFFER:
            continue
        pos += type_width
        (num_samples,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        if len(buf) != pos + num_samples * 4 + 4 + 8:
            continue
        samples = buf[pos : pos + num_samples * 4]
        pos += num_samples * 4
        rb_id, ts = struct.unpack_from("<IQ", buf, pos)
        return FloatRingBuffer(samples=samples, id=rb_id, ts=ts)

    raise YasError("deserialize_message: malformed FloatRingBuffer payload")


def _parse_yas_header(buf: bytes, offset: int) -> int:
    if len(buf) < offset + 7 or buf[offset : offset + 3] != b"yas":
        raise YasError("Invalid YAS buffer header")
    return offset + 7


def is_yas_message(buffer: bytes | bytearray | memoryview) -> bool:
    buf = bytes(buffer[:3])
    return buf == b"yas"
