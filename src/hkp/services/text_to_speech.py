from __future__ import annotations

# Service Documentation
# Service ID: text-to-speech
# Service Name: Text To Speech
# Runtime: hkp-python
# Modes: synthesize (Kokoro-82M via kokoro-onnx)
# Key Config: model (kokoro-v1.0|kokoro-v1.0-int8), voice (e.g. af_heart),
#             speed (0.5-2.0), lang (e.g. en-us), modelDir (model file cache)
# IO: in=String/TextData (the text) or JSON ({text} | {prompt})
#     -> out=FloatRingBuffer (24 kHz mono float32)
# Arrays: n/a
# Binary: emits FloatRingBuffer; non-text inputs yield an error JSON
# MixedData: not supported
#
# The ML dependencies are an optional extra:  pip install "hkp-python[tts]"
# Model files (~310 MB onnx + ~27 MB voices) are downloaded lazily on first
# use from the kokoro-onnx GitHub release; progress is reported through
# notify() so the UI can show downloading/loading/generating states.
#
# The `{text}` input shape is deliberate: the text-generation service's
# output pipes straight in, completing the local voice loop
# audio-input -> speech-to-text -> text-generation -> text-to-speech.

import time
import urllib.request
from array import array
from pathlib import Path
from typing import Any

from ..data import FloatRingBuffer, TextData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

TEXT_TO_SPEECH_DESCRIPTOR = ServiceRegistryEntry(
    service_id="text-to-speech",
    service_name="Text To Speech",
)

# Kokoro synthesizes at this rate; the browser Audio Output service plays
# FloatRingBuffer data at its configured sampleRate (default 24000).
OUTPUT_SAMPLE_RATE = 24000

_RELEASE_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
_MODELS = {
    "kokoro-v1.0": "kokoro-v1.0.onnx",
    "kokoro-v1.0-int8": "kokoro-v1.0.int8.onnx",
}
_VOICES_FILE = "voices-v1.0.bin"

DEFAULT_MODEL = "kokoro-v1.0"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "en-us"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "hkp-python" / "kokoro"

INSTALL_HINT = 'kokoro-onnx is not installed — run: pip install "hkp-python[tts]"'


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def _to_float32_bytes(samples: Any) -> bytes:
    """kokoro-onnx returns a numpy float32 array; fall back to array('f') for
    any float-sequence a stubbed backend may produce."""
    if hasattr(samples, "astype"):
        return samples.astype("float32").tobytes()
    return array("f", samples).tobytes()


class TextToSpeechService:
    service_id = TEXT_TO_SPEECH_DESCRIPTOR.service_id
    service_name = TEXT_TO_SPEECH_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._model_name = DEFAULT_MODEL
        self._voice = DEFAULT_VOICE
        self._speed = DEFAULT_SPEED
        self._lang = DEFAULT_LANG
        self._model_dir = str(DEFAULT_MODEL_DIR)
        self._engine: Any = None
        self._loaded_key: tuple[str, str] | None = None
        self._status = "idle"

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if config.get("model") in _MODELS:
            self._model_name = config["model"]
        if isinstance(config.get("voice"), str) and config["voice"]:
            self._voice = config["voice"]
        speed = config.get("speed")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool) and 0.5 <= speed <= 2.0:
            self._speed = float(speed)
        if isinstance(config.get("lang"), str) and config["lang"]:
            self._lang = config["lang"]
        if isinstance(config.get("modelDir"), str) and config["modelDir"]:
            self._model_dir = config["modelDir"]

        # Drop the cached engine if its parameters changed; it reloads lazily.
        if self._loaded_key is not None and self._loaded_key != self._engine_key():
            self._engine = None
            self._loaded_key = None
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "model": self._model_name,
            "voice": self._voice,
            "speed": self._speed,
            "lang": self._lang,
            "modelDir": self._model_dir,
            "sampleRate": OUTPUT_SAMPLE_RATE,
            "status": self._status,
            "availableModels": list(_MODELS),
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if input is None:
            return None

        text = self._to_text(input)
        if text is None:
            return self._error(
                notify,
                "text-to-speech expects String input or JSON with 'text' or 'prompt'",
            )

        try:
            engine = self._ensure_engine(notify)
        except ImportError:
            return self._error(notify, INSTALL_HINT)
        except Exception as exc:
            return self._error(notify, f"failed to load model '{self._model_name}': {exc}")

        voices = list(engine.get_voices())
        if voices and self._voice not in voices:
            return self._error(
                notify,
                f"unknown voice '{self._voice}' — available: {', '.join(voices[:12])}"
                + (", ..." if len(voices) > 12 else ""),
            )

        self._set_status(notify, "generating")
        started = time.monotonic()
        try:
            samples, sample_rate = engine.create(
                text, voice=self._voice, speed=self._speed, lang=self._lang
            )
        except Exception as exc:
            return self._error(notify, f"synthesis failed: {exc}")

        payload = _to_float32_bytes(samples)
        sample_count = len(payload) // 4

        self._set_status(notify, "idle")
        notify(
            {
                "voice": self._voice,
                "sampleRate": sample_rate,
                "samples": sample_count,
                "audioMs": int(sample_count / sample_rate * 1000) if sample_rate else 0,
                "generationMs": int((time.monotonic() - started) * 1000),
            }
        )
        return FloatRingBuffer(samples=payload)

    def set_host(self, host: Any) -> None:
        pass

    def destroy(self) -> None:
        self._engine = None
        self._loaded_key = None

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_text(input: Any) -> str | None:
        text: str | None = None
        if isinstance(input, str):
            text = input
        elif isinstance(input, TextData):
            text = input.text
        elif isinstance(input, dict):
            for key in ("text", "prompt"):
                if isinstance(input.get(key), str) and input[key]:
                    text = input[key]
                    break
        if text is None or not text.strip():
            return None
        return text.strip()

    def _engine_key(self) -> tuple[str, str]:
        return (self._model_name, self._model_dir)

    def _ensure_engine(self, notify: NotifyCallback) -> Any:
        if self._engine is not None:
            return self._engine

        from kokoro_onnx import Kokoro

        model_dir = Path(self._model_dir).expanduser()
        model_path = model_dir / _MODELS[self._model_name]
        voices_path = model_dir / _VOICES_FILE
        for path in (model_path, voices_path):
            if not path.is_file():
                # Stable status values (idle|downloading|loading|generating|error)
                # so facade status-indicators can color-map them.
                self._set_status(notify, "downloading", detail=f"downloading {path.name}")
                _download(f"{_RELEASE_BASE}/{path.name}", path)

        self._set_status(notify, "loading", detail=f"loading model '{self._model_name}'")
        self._engine = Kokoro(str(model_path), str(voices_path))
        self._loaded_key = self._engine_key()
        return self._engine

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
