"""JSON beside a payload of numbers: a model's scores, the tensor its layers
add to. What is pinned: the payload stays what it was between services, turns
into float32 bytes only where it leaves, and reads the same whether it was made
here or arrived over a coordinator connection.
"""
from __future__ import annotations

import json
from array import array

import numpy as np
import pytest

from hkp.binary_frame import from_binary, to_binary
from hkp.data import BinaryData, MixedData
from hkp.server import _json_placeholder


def test_keeps_its_payload_as_it_was_given():
    tensor = np.arange(6, dtype=np.float64).reshape(2, 3)

    assert MixedData({"shape": [2, 3]}, tensor).payload is tensor


def test_travels_as_little_endian_float32_whatever_it_held():
    expected = np.arange(6, dtype="<f4").tobytes()

    assert MixedData({}, np.arange(6, dtype=np.float64).reshape(2, 3)).to_bytes() == expected
    assert MixedData({}, array("f", range(6))).to_bytes() == expected
    assert MixedData({}, array("d", range(6))).to_bytes() == expected
    assert MixedData({}, expected).to_bytes() == expected
    with pytest.raises(TypeError, match="cannot carry a str"):
        MixedData({}, "text").to_bytes()


def test_says_its_size_without_becoming_bytes():
    assert MixedData({}, np.zeros((2, 3), dtype=np.float64)).size == 24
    assert MixedData({}, array("f", range(5))).size == 20
    assert MixedData({}, b"\x00" * 7).size == 7


def test_crosses_a_coordinator_connection_and_reads_the_same_on_the_other_side():
    scores = MixedData({"tokens": [1, 2]}, np.array([0.5, -1.0], dtype=np.float32))

    shape, payload = to_binary(scores)
    arrived = MixedData.of(from_binary(shape, payload))

    assert shape == {"kind": "mixed", "json": {"tokens": [1, 2]}}
    assert arrived.meta == {"tokens": [1, 2]}
    assert np.frombuffer(arrived.payload, dtype="<f4").tolist() == [0.5, -1.0]


def test_reads_only_what_is_mixed_data():
    value = MixedData({}, b"")

    assert MixedData.of(value) is value
    assert MixedData.of({"n": 1, "binary": BinaryData(b"\x01")}).payload == b"\x01"
    assert MixedData.of({"n": 1}) is None
    assert MixedData.of([1, 2]) is None


def test_is_reported_by_its_json_side_and_its_size():
    report = {"output": MixedData({"tokens": [1]}, np.zeros(4, dtype=np.float32))}

    assert json.loads(json.dumps(report, default=_json_placeholder)) == {
        "output": {"tokens": [1], "binary": {"type": "BinaryData", "size": 16}}
    }
