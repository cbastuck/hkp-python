from __future__ import annotations

"""HKP shared data types for the Python runtime.

These mirror the cross-runtime data model:
  hkp-frontend/src/runtime/rest/Data.ts   (TypeScript)
  hkp-rt/lib/include/types/data.h         (C++)

Plain ``dict`` / ``list`` values represent the JSON data type, so existing
services keep working untouched. The classes below cover the non-JSON types
that travel over the YAS binary wire format.
"""

from array import array
from dataclasses import dataclass, field
from typing import Any
import sys
import time


class DataTypeId:
    """Wire-format type ids — in sync with data.h / Data.ts."""

    UNDEFINED = 0
    FLOAT_RING_BUFFER = 1
    JSON = 2
    BINARY = 3
    STRING = 4
    NULL = 5
    CONTROL_FLOW = 6
    CUSTOM = 7


@dataclass
class FloatRingBuffer:
    """Contiguous float32 samples with id + timestamp (the audio type)."""

    samples: bytes  # raw little-endian float32 samples
    id: int = 0
    ts: int = field(default_factory=lambda: int(time.time() * 1000))

    @classmethod
    def from_floats(cls, values: list[float] | array, id: int = 0, ts: int | None = None) -> "FloatRingBuffer":
        arr = values if isinstance(values, array) else array("f", values)
        if arr.typecode != "f":
            raise ValueError("FloatRingBuffer.from_floats requires float32 ('f') values")
        kwargs = {} if ts is None else {"ts": ts}
        return cls(samples=arr.tobytes(), id=id, **kwargs)

    def to_floats(self) -> array:
        arr = array("f")
        arr.frombytes(self.samples)
        return arr

    @property
    def num_samples(self) -> int:
        return len(self.samples) // 4


@dataclass
class TextData:
    text: str


@dataclass
class BinaryData:
    data: bytes


@dataclass
class ControlFlowData:
    """Early-return signal: skips the remaining services in the runtime and
    makes `result` the runtime's output (see the shared data-types table)."""

    result: Any


_BYTES = (bytes, bytearray, memoryview)


@dataclass
class MixedData:
    """JSON beside a payload of numbers or bytes — a model's scores, the
    tensor its layers add to.

    Inside one runtime the payload stays whatever the service that made it
    holds: a tensor, an array, bytes. Services hand it on untouched, so
    fourteen of them in a row copy nothing. It becomes bytes only where the
    value leaves — a coordinator connection, a captured call report — and then
    as little-endian float32 unless it already is bytes.

    Arriving from another runtime the same value is a dict with the bytes under
    ``binary`` (see ``binary_frame.py``); ``MixedData.of`` reads either.
    """

    meta: dict[str, Any]
    payload: Any

    @classmethod
    def of(cls, value: Any) -> "MixedData | None":
        """``value`` as mixed data, or None when it is not: one of these
        already, or the dict a coordinator connection delivers."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            carried = value.get("binary")
            if isinstance(carried, BinaryData):
                carried = carried.data
            if isinstance(carried, _BYTES):
                rest = {key: item for key, item in value.items() if key != "binary"}
                return cls(rest, bytes(carried))
        return None

    @property
    def size(self) -> int:
        """How many bytes the payload travels as, without making them."""
        payload = self.payload
        if isinstance(payload, memoryview):
            return payload.nbytes
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        if isinstance(payload, array):
            return len(payload) * 4
        count = getattr(payload, "numel", None)  # a torch tensor
        if callable(count):
            return int(count()) * 4
        size = getattr(payload, "size", None)  # a numpy array
        if isinstance(size, int):
            return size * 4
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """The payload as it travels."""
        payload = self.payload
        if isinstance(payload, _BYTES):
            return bytes(payload)
        if isinstance(payload, array):
            floats = payload if payload.typecode == "f" else array("f", payload)
            if sys.byteorder != "little":
                floats = array("f", floats)
                floats.byteswap()
            return floats.tobytes()
        if callable(getattr(payload, "detach", None)):  # a torch tensor
            payload = payload.detach().cpu().numpy()
        if callable(getattr(payload, "astype", None)):  # a numpy array
            return payload.astype("<f4").tobytes()
        raise TypeError(f"MixedData cannot carry a {type(payload).__name__}")


class NullData:
    """Signals stop / no output."""

    _instance: "NullData | None" = None

    def __new__(cls) -> "NullData":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NullData()"


class UndefinedData:
    """Uninitialized / not yet set."""

    _instance: "UndefinedData | None" = None

    def __new__(cls) -> "UndefinedData":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UndefinedData()"


def get_data_type_id(data: object) -> int:
    if isinstance(data, FloatRingBuffer):
        return DataTypeId.FLOAT_RING_BUFFER
    if isinstance(data, (dict, list)):
        return DataTypeId.JSON
    if isinstance(data, BinaryData):
        return DataTypeId.BINARY
    if isinstance(data, (TextData, str)):
        return DataTypeId.STRING
    if isinstance(data, NullData) or data is None:
        return DataTypeId.NULL
    if isinstance(data, UndefinedData):
        return DataTypeId.UNDEFINED
    raise TypeError(f"get_data_type_id: unsupported data type: {type(data)!r}")
