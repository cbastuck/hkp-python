from __future__ import annotations

# Service Documentation
# Service ID: audio-encode
# Service Name: Audio Encode
# Runtime: hkp-python
# Modes: mp3 | wav — the format is configuration, not a mode
# Key Config: format, bitrate, quality, sampleRate, channels
# IO: in=FloatRingBuffer (float32 samples) -> out=the encoded bytes
# Arrays: n/a
# Binary: emits it; that is the whole service
# MixedData: not native in runtime
#
# The step between audio a board made and audio a board can keep. Services that
# produce sound produce samples — a FloatRingBuffer is what text-to-speech,
# a microphone and a synthesiser all hand on, and what the Audio Output service
# plays — but samples are not a file: nothing stores them, serves them or hands
# them to a player. This encodes them.
#
# It emits **bare bytes**, not bytes with a name or a content type, because what
# they are called belongs to whatever keeps them: `storage` reads the file
# extension, an endpoint reads what storage says. An encoder that also decided
# names would have to be told about both.
#
# `wav` needs nothing beyond the standard library and is always available.
# `mp3` is an optional extra — pip install "hkp-python[mp3]" — because a board
# that only ever plays audio back has no reason to carry an encoder; the two
# formats are the same service because choosing between them is a board's
# decision about size against fidelity, not a different job.
#
# The sample rate is configuration rather than something read from the input:
# a FloatRingBuffer is a generic float carrier and does not say how fast its
# samples were meant to be played. Kokoro synthesises at 24 kHz, which is the
# default here for the same reason it is the default in Audio Output.

import io
import struct
import wave
from array import array
from typing import Any

from ..data import FloatRingBuffer
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

AUDIO_ENCODE_DESCRIPTOR = ServiceRegistryEntry(
    service_id="audio-encode",
    service_name="Audio Encode",
    version="v1",
)

FORMATS = ("mp3", "wav")

DEFAULT_FORMAT = "mp3"
DEFAULT_BITRATE = 64
DEFAULT_QUALITY = 5
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1

INSTALL_HINT = 'lameenc is not installed — run: pip install "hkp-python[mp3]"'


def _to_int16(samples: array) -> bytes:
    """float32 in -1..1 as the signed 16-bit PCM both encoders take.

    Clipped rather than scaled to fit: a sample outside the range is a sample
    that was already too loud when it was made, and quietening the whole clip to
    accommodate one would change audio that was correct.
    """
    clipped = array(
        "h",
        (
            32767 if value >= 1.0 else -32768 if value <= -1.0 else int(value * 32767)
            for value in samples
        ),
    )
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        clipped.byteswap()
    return clipped.tobytes()


class AudioEncodeService:
    service_id = AUDIO_ENCODE_DESCRIPTOR.service_id
    service_name = AUDIO_ENCODE_DESCRIPTOR.service_name
    version = AUDIO_ENCODE_DESCRIPTOR.version
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._format = DEFAULT_FORMAT
        self._bitrate = DEFAULT_BITRATE
        self._quality = DEFAULT_QUALITY
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._channels = DEFAULT_CHANNELS
        self._last_bytes = 0
        self._last_seconds = 0.0
        self._error = ""

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if config.get("format") in FORMATS:
            self._format = config["format"]
        bitrate = config.get("bitrate")
        if isinstance(bitrate, int) and not isinstance(bitrate, bool) and 8 <= bitrate <= 320:
            self._bitrate = bitrate
        quality = config.get("quality")
        if isinstance(quality, int) and not isinstance(quality, bool) and 0 <= quality <= 9:
            self._quality = quality
        rate = config.get("sampleRate")
        if isinstance(rate, int) and not isinstance(rate, bool) and 8000 <= rate <= 192000:
            self._sample_rate = rate
        channels = config.get("channels")
        if channels in (1, 2):
            self._channels = channels
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "format": self._format,
            "bitrate": self._bitrate,
            "quality": self._quality,
            "sampleRate": self._sample_rate,
            "channels": self._channels,
            "lastBytes": self._last_bytes,
            "lastSeconds": round(self._last_seconds, 3),
            "availableFormats": list(FORMATS),
            "error": self._error,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if input is None:
            return None

        if not isinstance(input, FloatRingBuffer):
            return self._fail(
                notify, "audio-encode expects FloatRingBuffer input (float32 samples)"
            )

        samples = input.to_floats()
        if not len(samples):
            return self._fail(notify, "audio-encode was given no samples")

        pcm = _to_int16(samples)
        try:
            encoded = (
                self._encode_mp3(pcm) if self._format == "mp3" else self._encode_wav(pcm)
            )
        except ImportError:
            return self._fail(notify, INSTALL_HINT)
        except Exception as error:  # a bad rate or a broken encoder, not a bad board
            return self._fail(notify, f"audio-encode failed: {error}")

        self._error = ""
        self._last_bytes = len(encoded)
        self._last_seconds = len(samples) / (self._sample_rate * self._channels)
        notify(self.get_state())
        return encoded

    # ── Private ────────────────────────────────────────────────────────────────

    def _encode_mp3(self, pcm: bytes) -> bytes:
        try:
            import lameenc
        except ImportError as error:  # reported as the install hint above
            raise ImportError(INSTALL_HINT) from error

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self._bitrate)
        encoder.set_in_sample_rate(self._sample_rate)
        encoder.set_channels(self._channels)
        encoder.set_quality(self._quality)
        # No informational tag frame: what an episode is called belongs to the
        # store's path and to the feed, not to a header inside the bytes.
        encoder.silence()
        return bytes(encoder.encode(pcm)) + bytes(encoder.flush())

    def _encode_wav(self, pcm: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as file:
            file.setnchannels(self._channels)
            file.setsampwidth(2)
            file.setframerate(self._sample_rate)
            file.writeframes(pcm)
        return buffer.getvalue()

    def _fail(self, notify: NotifyCallback, message: str) -> JsonRecord:
        self._error = message
        notify(self.get_state())
        return {"error": message}
