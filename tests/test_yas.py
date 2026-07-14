from __future__ import annotations

import json
import struct

import pytest

from hkp.data import BinaryData, DataTypeId, FloatRingBuffer, NullData, TextData
from hkp.yas import (
    Message,
    MessagePurpose,
    YasError,
    deserialize_message,
    is_yas_message,
    serialize_message,
)

YAS_HEADER = b"yas0017"


def _message_header(purpose: int, data_type: int, sender: str) -> bytes:
    sender_bytes = sender.encode("ascii")
    return YAS_HEADER + struct.pack("<HHQ", purpose, data_type, len(sender_bytes)) + sender_bytes


# ── Round-trips through our own writer/reader ──────────────────────────────────


def test_ring_buffer_round_trip() -> None:
    rb = FloatRingBuffer.from_floats([0.0, -1.5, 0.25, 3.0], id=42, ts=1234567890123)
    msg = deserialize_message(serialize_message(rb, sender="svc-1"))

    assert msg.sender == "svc-1"
    assert msg.purpose == MessagePurpose.NOTIFICATION
    assert isinstance(msg.data, FloatRingBuffer)
    assert msg.data.id == 42
    assert msg.data.ts == 1234567890123
    assert list(msg.data.to_floats()) == [0.0, -1.5, 0.25, 3.0]


def test_json_round_trip() -> None:
    payload = {"text": "hello wörld", "segments": [{"start": 0.0, "end": 1.5}]}
    msg = deserialize_message(serialize_message(payload, sender=""))
    assert msg.data == payload


def test_json_array_round_trip() -> None:
    msg = deserialize_message(serialize_message([1, 2, {"a": True}]))
    assert msg.data == [1, 2, {"a": True}]


def test_text_round_trip() -> None:
    msg = deserialize_message(serialize_message(TextData("plain text äöü")))
    assert isinstance(msg.data, TextData)
    assert msg.data.text == "plain text äöü"


def test_null_round_trip() -> None:
    msg = deserialize_message(serialize_message(NullData()))
    assert isinstance(msg.data, NullData)


def test_none_serializes_as_null() -> None:
    msg = deserialize_message(serialize_message(None))
    assert isinstance(msg.data, NullData)


def test_purpose_and_sender_survive() -> None:
    raw = serialize_message({"x": 1}, sender="req-99", purpose=MessagePurpose.RESULT_WITH_REQUEST_ID)
    msg = deserialize_message(raw)
    assert msg.purpose == MessagePurpose.RESULT_WITH_REQUEST_ID
    assert msg.sender == "req-99"


# ── Foreign-writer fixtures ────────────────────────────────────────────────────


def test_deserialize_typescript_ring_buffer_dialect() -> None:
    """Byte-exact replica of Message.ts serializeYasMessage + serializeYasRingBuffer.

    The TS writer stores the nested type id as uint32 (the C++ writer uses
    uint16); the codec must auto-detect this dialect.
    """
    samples = struct.pack("<3f", 0.5, -0.5, 1.0)
    payload = (
        YAS_HEADER
        + struct.pack("<IQ", DataTypeId.FLOAT_RING_BUFFER, 3)  # uint32 type id
        + samples
        + struct.pack("<IQ", 7, 111222333444)
    )
    raw = _message_header(MessagePurpose.NOTIFICATION, DataTypeId.FLOAT_RING_BUFFER, "req-1") + payload

    msg = deserialize_message(raw)
    assert msg.sender == "req-1"
    assert isinstance(msg.data, FloatRingBuffer)
    assert msg.data.num_samples == 3
    assert msg.data.id == 7
    assert msg.data.ts == 111222333444
    assert list(msg.data.to_floats()) == [0.5, -0.5, 1.0]


def test_deserialize_cpp_ring_buffer_dialect() -> None:
    """C++ FloatRingBuffer::serialise stores the nested type id as uint16."""
    samples = struct.pack("<2f", 2.0, 4.0)
    payload = (
        YAS_HEADER
        + struct.pack("<HQ", DataTypeId.FLOAT_RING_BUFFER, 2)  # uint16 type id
        + samples
        + struct.pack("<IQ", 3, 999)
    )
    raw = _message_header(MessagePurpose.RESULT, DataTypeId.FLOAT_RING_BUFFER, "") + payload

    msg = deserialize_message(raw)
    assert isinstance(msg.data, FloatRingBuffer)
    assert msg.data.num_samples == 2
    assert msg.data.id == 3
    assert list(msg.data.to_floats()) == [2.0, 4.0]


def test_our_ring_buffer_writer_uses_uint16_dialect() -> None:
    """The TS deserializer reads a uint16 type id — our writer must match it."""
    rb = FloatRingBuffer.from_floats([1.0], id=1, ts=2)
    raw = serialize_message(rb, sender="s")

    offset = 7 + 2 + 2 + 8 + 1  # header, purpose, dtype, sender len, sender
    offset += 7  # nested YAS header
    (type_id,) = struct.unpack_from("<H", raw, offset)
    assert type_id == DataTypeId.FLOAT_RING_BUFFER
    (num_samples,) = struct.unpack_from("<Q", raw, offset + 2)
    assert num_samples == 1


def test_deserialize_cpp_json_payload() -> None:
    """C++ buildFrame wraps JSON as nested-header + uint64 length + utf-8."""
    body = json.dumps({"ok": True}).encode("utf-8")
    payload = YAS_HEADER + struct.pack("<Q", len(body)) + body
    raw = _message_header(MessagePurpose.NOTIFICATION, DataTypeId.JSON, "svc") + payload
    assert deserialize_message(raw).data == {"ok": True}


def test_deserialize_binary_payload() -> None:
    raw = _message_header(MessagePurpose.NOTIFICATION, DataTypeId.BINARY, "") + b"\x01\x02\x03"
    msg = deserialize_message(raw)
    assert isinstance(msg.data, BinaryData)
    assert msg.data.data == b"\x01\x02\x03"


# ── Error handling ─────────────────────────────────────────────────────────────


def test_invalid_magic_rejected() -> None:
    with pytest.raises(YasError):
        deserialize_message(b"nope" + b"\x00" * 32)


def test_truncated_ring_buffer_rejected() -> None:
    rb = FloatRingBuffer.from_floats([1.0, 2.0, 3.0])
    raw = serialize_message(rb)
    with pytest.raises(YasError):
        deserialize_message(raw[:-5])


def test_is_yas_message() -> None:
    assert is_yas_message(serialize_message({"a": 1}))
    assert not is_yas_message(b'{"type":"processRuntime"}')
