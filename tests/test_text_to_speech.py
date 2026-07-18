from __future__ import annotations

import sys
import types
from array import array
from pathlib import Path
from typing import Any

import pytest

from hkp.data import FloatRingBuffer, TextData
from hkp.services import text_to_speech
from hkp.services.text_to_speech import (
    INSTALL_HINT,
    OUTPUT_SAMPLE_RATE,
    TextToSpeechService,
)
from hkp.types import ServiceConfiguration

SAMPLES = array("f", [0.0, 0.25, -0.25, 1.0])


def _make_service(state: dict | None = None) -> TextToSpeechService:
    return TextToSpeechService(
        ServiceConfiguration(service_id="text-to-speech", uuid="tts-1", state=state)
    )


def _collect_notify(sink: list) -> Any:
    def notify(payload: Any, instance_id: str | None = None) -> None:
        sink.append(payload)

    return notify


# ── Stub backend ───────────────────────────────────────────────────────────────


class _FakeKokoro:
    instances: list["_FakeKokoro"] = []

    def __init__(self, model_path: str, voices_path: str) -> None:
        self.model_path = model_path
        self.voices_path = voices_path
        self.calls: list[dict] = []
        _FakeKokoro.instances.append(self)

    def get_voices(self) -> list[str]:
        return ["af_heart", "af_bella", "am_adam"]

    def create(self, text: str, voice: str, speed: float, lang: str):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        return SAMPLES, OUTPUT_SAMPLE_RATE


@pytest.fixture
def fake_kokoro(monkeypatch, tmp_path: Path) -> type[_FakeKokoro]:
    """Installs a stub kokoro_onnx module and pre-creates the model files in a
    tmp modelDir so nothing is downloaded."""
    module = types.ModuleType("kokoro_onnx")
    module.Kokoro = _FakeKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)
    _FakeKokoro.instances = []

    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"onnx")
    (tmp_path / "kokoro-v1.0.int8.onnx").write_bytes(b"onnx")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"voices")
    return _FakeKokoro


@pytest.fixture
def model_dir(tmp_path: Path) -> str:
    return str(tmp_path)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_configure_and_state_defaults() -> None:
    svc = _make_service()
    state = svc.get_state()
    assert state["model"] == "kokoro-v1.0"
    assert state["voice"] == "af_heart"
    assert state["speed"] == 1.0
    assert state["lang"] == "en-us"
    assert state["sampleRate"] == OUTPUT_SAMPLE_RATE
    assert state["status"] == "idle"
    assert "kokoro-v1.0-int8" in state["availableModels"]

    svc.configure({"model": "kokoro-v1.0-int8", "voice": "am_adam", "speed": 1.5, "lang": "en-gb"})
    state = svc.get_state()
    assert state["model"] == "kokoro-v1.0-int8"
    assert state["voice"] == "am_adam"
    assert state["speed"] == 1.5
    assert state["lang"] == "en-gb"


def test_invalid_config_values_ignored() -> None:
    svc = _make_service()
    svc.configure({"model": "gpt-tts", "voice": "", "speed": 9.0, "lang": "", "modelDir": ""})
    state = svc.get_state()
    assert state["model"] == "kokoro-v1.0"
    assert state["voice"] == "af_heart"
    assert state["speed"] == 1.0
    assert state["lang"] == "en-us"
    # bool must not sneak in as a numeric speed
    svc.configure({"speed": True})
    assert svc.get_state()["speed"] == 1.0


def test_missing_dependency_reports_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
    svc = _make_service()
    result = svc.process("hello", _collect_notify([]))
    assert result == {"error": INSTALL_HINT}
    assert svc.get_state()["status"] == "error"


def test_synthesizes_string_to_float_ring_buffer(fake_kokoro, model_dir) -> None:
    svc = _make_service({"modelDir": model_dir})
    notifications: list = []

    result = svc.process("hallo how are you", _collect_notify(notifications))

    assert isinstance(result, FloatRingBuffer)
    assert result.samples == SAMPLES.tobytes()

    call = fake_kokoro.instances[0].calls[0]
    assert call == {"text": "hallo how are you", "voice": "af_heart", "speed": 1.0, "lang": "en-us"}

    statuses = [n["status"] for n in notifications if isinstance(n, dict) and "status" in n]
    assert statuses == ["loading", "generating", "idle"]

    meta = next(n for n in notifications if isinstance(n, dict) and "sampleRate" in n)
    assert meta["sampleRate"] == OUTPUT_SAMPLE_RATE
    assert meta["samples"] == len(SAMPLES)
    assert meta["voice"] == "af_heart"
    assert meta["audioMs"] == int(len(SAMPLES) / OUTPUT_SAMPLE_RATE * 1000)


def test_accepts_text_data_and_json_shapes(fake_kokoro, model_dir) -> None:
    svc = _make_service({"modelDir": model_dir})
    notify = _collect_notify([])

    assert isinstance(svc.process(TextData("from text data"), notify), FloatRingBuffer)
    # Output of the text-generation service pipes in directly via its `text` key.
    assert isinstance(svc.process({"text": "from llm", "model": "x"}, notify), FloatRingBuffer)
    assert isinstance(svc.process({"prompt": "from prompt"}, notify), FloatRingBuffer)

    texts = [c["text"] for c in fake_kokoro.instances[0].calls]
    assert texts == ["from text data", "from llm", "from prompt"]


def test_unknown_voice_lists_available(fake_kokoro, model_dir) -> None:
    svc = _make_service({"modelDir": model_dir, "voice": "af_nope"})
    result = svc.process("hello", _collect_notify([]))
    assert "unknown voice 'af_nope'" in result["error"]
    assert "af_heart" in result["error"]


def test_downloads_missing_model_files(fake_kokoro, monkeypatch, tmp_path: Path) -> None:
    downloads: list = []

    def fake_download(url: str, destination: Path) -> None:
        downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stub")

    monkeypatch.setattr(text_to_speech, "_download", fake_download)
    empty_dir = tmp_path / "empty"
    svc = _make_service({"modelDir": str(empty_dir)})
    notifications: list = []

    result = svc.process("hello", _collect_notify(notifications))

    assert isinstance(result, FloatRingBuffer)
    assert [u.rsplit("/", 1)[1] for u in downloads] == ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
    statuses = [n["status"] for n in notifications if isinstance(n, dict) and "status" in n]
    assert statuses[:2] == ["downloading", "downloading"]

    # Second call: files exist, engine cached — no further downloads.
    svc.process("again", _collect_notify([]))
    assert len(downloads) == 2


def test_engine_reloads_when_model_changes(fake_kokoro, model_dir) -> None:
    svc = _make_service({"modelDir": model_dir})
    notify = _collect_notify([])
    svc.process("one", notify)
    svc.process("two", notify)
    assert len(fake_kokoro.instances) == 1

    svc.configure({"model": "kokoro-v1.0-int8"})
    svc.process("three", notify)
    assert len(fake_kokoro.instances) == 2
    assert fake_kokoro.instances[1].model_path.endswith("kokoro-v1.0.int8.onnx")


def test_unsupported_input_yields_error(fake_kokoro, model_dir) -> None:
    svc = _make_service({"modelDir": model_dir})
    assert "error" in svc.process(b"binary", _collect_notify([]))
    assert "error" in svc.process({"some": "json"}, _collect_notify([]))
    assert "error" in svc.process("   ", _collect_notify([]))


def test_none_input_passes_through() -> None:
    svc = _make_service()
    assert svc.process(None, _collect_notify([])) is None
