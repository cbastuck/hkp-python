from __future__ import annotations

import io
import math
import wave
from array import array
from typing import Any

import pytest

from hkp.data import FloatRingBuffer
from hkp.services.audio_encode import AudioEncodeService
from hkp.types import ServiceConfiguration

"""Samples a board made, as a file a board can keep.

What is worth pinning is the boundary either side of the encoder: what arrives
is float samples with no sample rate of their own, and what leaves is bytes
somebody else will name.
"""


def _service(state: dict | None = None) -> AudioEncodeService:
    return AudioEncodeService(
        ServiceConfiguration(service_id="audio-encode", uuid="enc-1", state=state)
    )


def _notify(sink: list) -> Any:
    def notify(payload: Any, instance_id: str | None = None) -> None:
        sink.append(payload)

    return notify


def _tone(seconds: float, rate: int = 24000) -> FloatRingBuffer:
    count = int(seconds * rate)
    return FloatRingBuffer.from_floats(
        array("f", (0.4 * math.sin(2 * math.pi * 440 * i / rate) for i in range(count)))
    )


def test_wav_needs_nothing_installed_and_says_what_it_holds() -> None:
    service = _service({"format": "wav"})
    notes: list = []

    encoded = service.process(_tone(0.1), _notify(notes))

    assert isinstance(encoded, bytes)
    with wave.open(io.BytesIO(encoded), "rb") as file:
        assert file.getnchannels() == 1
        assert file.getsampwidth() == 2
        assert file.getframerate() == 24000
        assert file.getnframes() == 2400
    assert notes[-1]["lastSeconds"] == pytest.approx(0.1, abs=0.001)


def test_mp3_is_an_mp3() -> None:
    lameenc = pytest.importorskip("lameenc", reason='needs pip install "hkp-python[mp3]"')
    assert lameenc

    encoded = _service({"format": "mp3", "bitrate": 64}).process(_tone(0.5), _notify([]))

    assert isinstance(encoded, bytes)
    # A frame header, which is all an mp3 announces itself with.
    assert encoded[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    # Roughly 64 kbit for half a second, with room for the encoder's own frames.
    assert 2000 < len(encoded) < 8000


def test_the_sample_rate_is_configuration_not_something_read_off_the_input() -> None:
    # A FloatRingBuffer carries floats, not a rate: the same samples are a
    # different length of audio depending on what a board says they are.
    samples = _tone(1.0, rate=24000)

    slow = _service({"format": "wav", "sampleRate": 16000}).process(samples, _notify([]))

    with wave.open(io.BytesIO(slow), "rb") as file:
        assert file.getframerate() == 16000
        assert file.getnframes() == 24000


def test_loud_samples_are_clipped_rather_than_the_clip_quietened() -> None:
    too_loud = FloatRingBuffer.from_floats(array("f", [2.0, -2.0, 0.5]))

    encoded = _service({"format": "wav"}).process(too_loud, _notify([]))

    with wave.open(io.BytesIO(encoded), "rb") as file:
        frames = array("h")
        frames.frombytes(file.readframes(3))
    assert frames[0] == 32767
    assert frames[1] == -32768
    # The sample that was in range is untouched by the two that were not.
    assert frames[2] == pytest.approx(16383, abs=1)


def test_anything_that_is_not_audio_is_reported_rather_than_encoded() -> None:
    service = _service({"format": "wav"})
    notes: list = []

    answer = service.process({"text": "not audio"}, _notify(notes))

    assert "error" in answer
    assert service.get_state()["error"]


def test_nothing_in_nothing_out() -> None:
    assert _service().process(None, _notify([])) is None
