from __future__ import annotations

# Service Documentation
# Service ID: speech-to-text
# Service Name: Speech To Text
# Runtime: hkp-python
# Modes: transcribe (faster-whisper backend)
# Key Config: model (tiny|base|small|medium|large-v3|distil-large-v3),
#             language ("auto" or ISO code), computeType (int8|int8_float16|float16|float32),
#             device (auto|cpu|cuda)
# IO: in=FloatRingBuffer (16 kHz mono float32) -> out=JSON
#     { text, language, languageProbability, durationMs, segments: [{start, end, text}] }
# Arrays: n/a
# Binary: consumes FloatRingBuffer; other inputs yield an error JSON
# MixedData: not supported
#
# The ML dependencies are an optional extra:  pip install "hkp-python[asr]"
# The model is loaded lazily on first use (first call downloads it from
# Hugging Face); progress is reported through notify() so the UI can show
# loading/transcribing states.

from typing import Any

from ..data import FloatRingBuffer
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

SPEECH_TO_TEXT_DESCRIPTOR = ServiceRegistryEntry(
    service_id="speech-to-text",
    service_name="Speech To Text",
)

# The pcm mode of the browser Audio Input service emits at this rate; it is
# also what Whisper-family models expect.
EXPECTED_SAMPLE_RATE = 16000

_MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
_COMPUTE_TYPES = ["int8", "int8_float16", "float16", "float32"]
_DEVICES = ["auto", "cpu", "cuda"]

INSTALL_HINT = 'faster-whisper is not installed — run: pip install "hkp-python[asr]"'


class SpeechToTextService:
    service_id = SPEECH_TO_TEXT_DESCRIPTOR.service_id
    service_name = SPEECH_TO_TEXT_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._model_name = "small"
        self._language = "auto"
        self._compute_type = "int8"
        self._device = "auto"
        self._model: Any = None
        self._loaded_key: tuple[str, str, str] | None = None
        self._status = "idle"

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if config.get("model") in _MODELS:
            self._model_name = config["model"]
        if isinstance(config.get("language"), str) and config["language"]:
            self._language = config["language"]
        if config.get("computeType") in _COMPUTE_TYPES:
            self._compute_type = config["computeType"]
        if config.get("device") in _DEVICES:
            self._device = config["device"]

        # Drop the cached model if its parameters changed; it reloads lazily.
        if self._loaded_key is not None and self._loaded_key != self._model_key():
            self._model = None
            self._loaded_key = None
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "model": self._model_name,
            "language": self._language,
            "computeType": self._compute_type,
            "device": self._device,
            "status": self._status,
            "availableModels": _MODELS,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if input is None:
            return None
        if not isinstance(input, FloatRingBuffer):
            return self._error(notify, "speech-to-text expects FloatRingBuffer input (16 kHz mono float32)")

        try:
            model = self._ensure_model(notify)
        except ImportError:
            return self._error(notify, INSTALL_HINT)
        except Exception as exc:
            return self._error(notify, f"failed to load model '{self._model_name}': {exc}")

        import numpy as np

        audio = np.frombuffer(input.samples, dtype=np.float32)
        duration_ms = int(len(audio) / EXPECTED_SAMPLE_RATE * 1000)

        self._set_status(notify, "transcribing")
        try:
            language = None if self._language == "auto" else self._language
            segments_iter, info = model.transcribe(audio, language=language)
            segments = [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                for s in segments_iter
            ]
        except Exception as exc:
            return self._error(notify, f"transcription failed: {exc}")

        self._set_status(notify, "idle")
        result = {
            "text": " ".join(s["text"] for s in segments).strip(),
            "language": getattr(info, "language", self._language),
            "languageProbability": round(getattr(info, "language_probability", 0.0), 3),
            "durationMs": duration_ms,
            "segments": segments,
        }
        notify(result)
        return result

    def set_host(self, host: Any) -> None:
        pass

    def destroy(self) -> None:
        self._model = None
        self._loaded_key = None

    # ── Internals ──────────────────────────────────────────────────────────────

    def _model_key(self) -> tuple[str, str, str]:
        return (self._model_name, self._device, self._compute_type)

    def _ensure_model(self, notify: NotifyCallback) -> Any:
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        # Stable status values (idle|loading|transcribing|error) so facade
        # status-indicators can color-map them; the model name goes in detail.
        self._set_status(notify, "loading", detail=f"loading model '{self._model_name}'")
        self._model = WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )
        self._loaded_key = self._model_key()
        return self._model

    def _set_status(self, notify: NotifyCallback, status: str, detail: str | None = None) -> None:
        self._status = status
        payload: JsonRecord = {"status": status}
        if detail is not None:
            payload["detail"] = detail
        notify(payload)

    def _error(self, notify: NotifyCallback, message: str) -> JsonRecord:
        self._set_status(notify, "error")
        result = {"error": message}
        notify(result)
        return result
