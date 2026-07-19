from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from hkp.data import FloatRingBuffer
from hkp.services.speech_to_text import INSTALL_HINT, SpeechToTextService
from hkp.types import ServiceConfiguration


def _make_service(state: dict | None = None) -> SpeechToTextService:
    return SpeechToTextService(
        ServiceConfiguration(service_id="speech-to-text", uuid="stt-1", state=state)
    )


def _collect_notify(sink: list) -> Any:
    def notify(payload: Any, instance_id: str | None = None) -> None:
        sink.append(payload)

    return notify


# ── Stub backend ───────────────────────────────────────────────────────────────


@dataclass
class _FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class _FakeInfo:
    language: str
    language_probability: float


class _FakeWhisperModel:
    created_with: dict | None = None
    last_audio = None

    def __init__(self, model_name: str, device: str = "auto", compute_type: str = "int8"):
        _FakeWhisperModel.created_with = {
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
        }

    def transcribe(self, audio, language=None):
        _FakeWhisperModel.last_audio = audio
        segments = [
            _FakeSegment(0.0, 1.2, " Hello "),
            _FakeSegment(1.2, 2.0, " world. "),
        ]
        return iter(segments), _FakeInfo("en", 0.987)


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    _FakeWhisperModel.created_with = None
    _FakeWhisperModel.last_audio = None
    return module


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_configure_and_state_defaults() -> None:
    svc = _make_service()
    state = svc.get_state()
    assert state["model"] == "small"
    assert state["language"] == "auto"
    assert state["status"] == "idle"

    svc.configure({"model": "tiny", "language": "de", "computeType": "float32"})
    state = svc.get_state()
    assert state["model"] == "tiny"
    assert state["language"] == "de"
    assert state["computeType"] == "float32"


def test_invalid_config_values_ignored() -> None:
    svc = _make_service()
    svc.configure({"model": "not-a-model", "device": "gpu9000"})
    state = svc.get_state()
    assert state["model"] == "small"
    assert state["device"] == "auto"


def test_transcribes_ring_buffer(fake_faster_whisper) -> None:
    svc = _make_service({"model": "tiny"})
    notifications: list = []
    rb = FloatRingBuffer.from_floats([0.0] * 16000)  # 1 second of silence

    result = svc.process(rb, _collect_notify(notifications))

    assert result["text"] == "Hello world."
    assert result["language"] == "en"
    assert result["durationMs"] == 1000
    assert result["segments"] == [
        {"start": 0.0, "end": 1.2, "text": "Hello"},
        {"start": 1.2, "end": 2.0, "text": "world."},
    ]
    assert _FakeWhisperModel.created_with == {
        "model": "tiny",
        "device": "auto",
        "compute_type": "int8",
    }
    statuses = [n["status"] for n in notifications if isinstance(n, dict) and "status" in n]
    assert statuses[-1] == "idle"
    assert any("loading" in s for s in statuses)


def test_resamples_configured_sample_rate_to_whisper_rate(fake_faster_whisper) -> None:
    svc = _make_service({"model": "tiny", "sampleRate": 24000})
    assert svc.get_state()["sampleRate"] == 24000
    rb = FloatRingBuffer.from_floats([0.0] * 24000)  # 1 second at 24 kHz

    result = svc.process(rb, _collect_notify([]))

    assert result["durationMs"] == 1000
    assert len(_FakeWhisperModel.last_audio) == 16000


def test_invalid_sample_rate_ignored() -> None:
    svc = _make_service()
    svc.configure({"sampleRate": -1})
    assert svc.get_state()["sampleRate"] == 16000


def test_model_reloads_after_reconfigure(fake_faster_whisper) -> None:
    svc = _make_service({"model": "tiny"})
    rb = FloatRingBuffer.from_floats([0.0] * 160)
    svc.process(rb, _collect_notify([]))

    svc.configure({"model": "base"})
    svc.process(rb, _collect_notify([]))
    assert _FakeWhisperModel.created_with["model"] == "base"


def test_missing_dependency_reports_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError
    svc = _make_service()
    result = svc.process(FloatRingBuffer.from_floats([0.0]), _collect_notify([]))
    assert result == {"error": INSTALL_HINT}
    assert svc.get_state()["status"] == "error"


def test_non_audio_input_yields_error() -> None:
    svc = _make_service()
    result = svc.process({"some": "json"}, _collect_notify([]))
    assert "error" in result


def test_none_input_passes_through() -> None:
    svc = _make_service()
    assert svc.process(None, _collect_notify([])) is None
