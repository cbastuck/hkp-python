from __future__ import annotations

# Service Documentation
# Service ID: text-generation
# Service Name: Text Generation
# Runtime: hkp-python
# Modes: chat (OpenAI-compatible /v1/chat/completions backend)
# Key Config: serverUrl (base URL of an OpenAI-compatible server, e.g. llama-server),
#             model (optional model name passed through to the server),
#             systemPrompt, temperature, topP, topK, maxTokens, timeoutSec
# IO: in=String/TextData (the prompt) or JSON ({prompt} | {text} | {messages: [...]})
#     -> out=JSON { text, thinking?, model, durationMs,
#                   usage: { promptTokens, completionTokens } }
# Arrays: n/a
# Binary: not supported; non-text inputs yield an error JSON
# MixedData: not supported
#
# The service is a thin client — the model runs in a separate local server
# process that speaks the OpenAI chat-completions API (llama-server, Ollama,
# vLLM, LM Studio, ...). No extra Python dependencies are needed; requests go
# through the standard library. The `{text}` input shape is deliberate: the
# speech-to-text service's output pipes straight in.
#
# Reference backend: 1-bit Bonsai 27B via the PrismML llama.cpp fork —
# see the "Text generation backend" section in the hkp-python README.

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ..data import TextData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

TEXT_GENERATION_DESCRIPTOR = ServiceRegistryEntry(
    service_id="text-generation",
    service_name="Text Generation",
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8081"

# Recommended defaults for the Bonsai reference backend (model card values);
# they are sensible generic chat settings for other backends too.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_SEC = 300.0

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def server_hint(server_url: str) -> str:
    return (
        f"no OpenAI-compatible server reachable at {server_url} — start one, e.g.: "
        f"llama-server -m Bonsai-27B-Q1_0.gguf --port 8081 -ngl 99"
    )


def _post_json(url: str, payload: JsonRecord, timeout: float) -> JsonRecord:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class TextGenerationService:
    service_id = TEXT_GENERATION_DESCRIPTOR.service_id
    service_name = TEXT_GENERATION_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._server_url = DEFAULT_SERVER_URL
        self._model = ""
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        self._temperature = DEFAULT_TEMPERATURE
        self._top_p = DEFAULT_TOP_P
        self._top_k = DEFAULT_TOP_K
        self._max_tokens = DEFAULT_MAX_TOKENS
        self._timeout_sec = DEFAULT_TIMEOUT_SEC
        # None = server default; False makes thinking models answer directly,
        # which matters for interactive boards (thinking burns the whole token
        # budget invisibly before the first visible character).
        self._thinking: bool | None = None
        self._status = "idle"

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("serverUrl"), str) and config["serverUrl"]:
            self._server_url = config["serverUrl"].rstrip("/")
        if isinstance(config.get("model"), str):
            self._model = config["model"]
        if isinstance(config.get("systemPrompt"), str):
            self._system_prompt = config["systemPrompt"]
        if _is_number(config.get("temperature")):
            self._temperature = float(config["temperature"])
        if _is_number(config.get("topP")):
            self._top_p = float(config["topP"])
        if _is_number(config.get("topK")):
            self._top_k = int(config["topK"])
        if _is_number(config.get("maxTokens")) and config["maxTokens"] > 0:
            self._max_tokens = int(config["maxTokens"])
        if _is_number(config.get("timeoutSec")) and config["timeoutSec"] > 0:
            self._timeout_sec = float(config["timeoutSec"])
        if "thinking" in config and (isinstance(config["thinking"], bool) or config["thinking"] is None):
            self._thinking = config["thinking"]
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "serverUrl": self._server_url,
            "model": self._model,
            "systemPrompt": self._system_prompt,
            "temperature": self._temperature,
            "topP": self._top_p,
            "topK": self._top_k,
            "maxTokens": self._max_tokens,
            "timeoutSec": self._timeout_sec,
            "thinking": self._thinking,
            "status": self._status,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if input is None:
            return None

        messages = self._to_messages(input)
        if messages is None:
            return self._error(
                notify,
                "text-generation expects String input or JSON with 'prompt', 'text', or 'messages'",
            )

        payload: JsonRecord = {
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if self._model:
            payload["model"] = self._model
        if self._thinking is not None:
            # llama-server extension (Qwen-style chat templates); only sent when
            # explicitly configured so other backends never see the field.
            payload["chat_template_kwargs"] = {"enable_thinking": self._thinking}

        self._set_status(notify, "generating")
        started = time.monotonic()
        try:
            response = _post_json(
                f"{self._server_url}/v1/chat/completions", payload, self._timeout_sec
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            return self._error(notify, f"server returned HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, OSError):
            return self._error(notify, server_hint(self._server_url))

        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return self._error(notify, f"unexpected response shape: {str(response)[:200]}")

        text, thinking = self._split_thinking(message)
        usage = response.get("usage") or {}
        result: JsonRecord = {
            "text": text,
            "model": response.get("model", self._model),
            "durationMs": int((time.monotonic() - started) * 1000),
            "usage": {
                "promptTokens": usage.get("prompt_tokens", 0),
                "completionTokens": usage.get("completion_tokens", 0),
            },
        }
        if thinking:
            result["thinking"] = thinking

        self._set_status(notify, "idle")
        notify(result)
        return result

    def set_host(self, host: Any) -> None:
        pass

    def destroy(self) -> None:
        pass

    # ── Internals ──────────────────────────────────────────────────────────────

    def _to_messages(self, input: Any) -> list[JsonRecord] | None:
        prompt: str | None = None
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, TextData):
            prompt = input.text
        elif isinstance(input, dict):
            if isinstance(input.get("messages"), list):
                return self._with_system_prompt(input["messages"])
            for key in ("prompt", "text"):
                if isinstance(input.get(key), str) and input[key]:
                    prompt = input[key]
                    break

        if prompt is None or not prompt.strip():
            return None
        return self._with_system_prompt([{"role": "user", "content": prompt}])

    def _with_system_prompt(self, messages: list[JsonRecord]) -> list[JsonRecord]:
        if not self._system_prompt:
            return messages
        if any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            return messages
        return [{"role": "system", "content": self._system_prompt}, *messages]

    @staticmethod
    def _split_thinking(message: JsonRecord) -> tuple[str, str]:
        """Separate reasoning from the answer for thinking models like Bonsai.

        The reasoning arrives either as a separate `reasoning_content` field
        (llama-server default) or inline as <think>...</think> in the content,
        depending on the server's chat-template handling.
        """
        content = message.get("content") or ""
        thinking = message.get("reasoning_content") or ""
        if THINK_CLOSE in content:
            inline, _, content = content.partition(THINK_CLOSE)
            inline = inline.replace(THINK_OPEN, "", 1)
            thinking = f"{thinking}\n{inline}".strip() if thinking else inline.strip()
        return content.strip(), thinking.strip()

    def _set_status(self, notify: NotifyCallback, status: str) -> None:
        self._status = status
        notify({"status": status})

    def _error(self, notify: NotifyCallback, message: str) -> JsonRecord:
        self._set_status(notify, "error")
        result = {"error": message}
        notify(result)
        return result
