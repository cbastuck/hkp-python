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
