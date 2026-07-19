from __future__ import annotations

# Service Documentation
# Service ID: text-generation
# Service Name: Text Generation
# Runtime: hkp-python
# Modes: chat, via one of two backends selected by the `backend` state:
#        server (default) — OpenAI-compatible /v1/chat/completions client
#        local            — loads a GGUF in-process via llama-cpp-python
# Key Config: backend (server|local),
#             serverUrl (server backend: base URL of an OpenAI-compatible server),
#             model (optional model name passed through to the server),
#             modelPath (local backend: path to a .gguf file),
#             contextSize, gpuLayers (local backend),
#             systemPrompt, temperature, topP, topK, maxTokens, timeoutSec,
#             stream (default true — generate token by token and notify the
#             growing text as {streamText} for live chat-bot-style display;
#             the pipeline output is unaffected and still emitted once, when
#             generation finishes)
# IO: in=String/TextData (the prompt) or JSON ({prompt} | {text} | {messages: [...]})
#     -> out=JSON { text, thinking?, model, durationMs,
#                   usage: { promptTokens, completionTokens } }
# Arrays: n/a
# Binary: not supported; non-text inputs yield an error JSON
# MixedData: not supported
#
# Server backend: the service is a thin client — the model runs in a separate
# local server process that speaks the OpenAI chat-completions API
# (llama-server, Ollama, vLLM, LM Studio, ...). No extra Python dependencies
# are needed; requests go through the standard library. This is the only way
# to run quants that need a custom llama.cpp build, e.g. 1-bit Bonsai 27B via
# the PrismML fork — see "Text generation backends" in the hkp-python README.
#
# Local backend: standard GGUFs (Qwen, Llama, ...) load directly into this
# process via llama-cpp-python (optional extra: pip install "hkp-python[llm]").
# The model is loaded lazily on first use and reloaded when modelPath /
# contextSize / gpuLayers change; no external server is required.
#
# The `{text}` input shape is deliberate: the speech-to-text service's output
# pipes straight in.

import json
import os
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

_BACKENDS = ["server", "local"]
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_GPU_LAYERS = -1  # -1 = offload every layer (llama-cpp-python convention)

INSTALL_HINT = 'llama-cpp-python is not installed — run: pip install "hkp-python[llm]"'

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


def _post_json_stream(url: str, payload: JsonRecord, timeout: float) -> Any:
    """POST an OpenAI-style streaming request and yield the SSE chunk objects.

    The caller must set "stream": true in the payload; chunks arrive as
    `data: {...}` server-sent-event lines and the generator ends on the
    `data: [DONE]` sentinel.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _consume_stream(chunks: Any, notify: NotifyCallback) -> JsonRecord:
    """Accumulate streamed completion chunks into a response of the same shape
    as a non-streaming call, notifying the growing text as {streamText} so a
    service UI can render the output while it is being generated. The last
    notification carries {streamDone: true}."""
    content = ""
    reasoning = ""
    model = ""
    usage: JsonRecord = {}
    last_sent = 0.0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("model"), str) and chunk["model"]:
            model = chunk["model"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        try:
            delta = chunk["choices"][0].get("delta") or {}
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        thought = delta.get("reasoning_content") or ""
        if thought:
            reasoning += thought
        piece = delta.get("content") or ""
        if piece:
            content += piece
            now = time.monotonic()
            # ~20 updates/s is plenty for a live view and keeps the
            # notification socket calm at high token rates.
            if now - last_sent >= 0.05:
                last_sent = now
                notify({"streamText": content})
    notify({"streamText": content, "streamDone": True})
    message: JsonRecord = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    response: JsonRecord = {"choices": [{"message": message}]}
    if model:
        response["model"] = model
    if usage:
        response["usage"] = usage
    return response


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class LocalLlm:
    """Lazy-loading wrapper around an in-process llama-cpp-python model.

    Owns the local-backend config (modelPath, contextSize, gpuLayers) and the
    loaded-model cache; shared by text-generation and skill-router so both
    services invalidate and reload identically.
    """

    def __init__(self) -> None:
        self.model_path = ""
        self.context_size = DEFAULT_CONTEXT_SIZE
        self.gpu_layers = DEFAULT_GPU_LAYERS
        self._llama: Any = None
        self._loaded_key: tuple[str, int, int] | None = None

    def configure(self, config: JsonRecord) -> None:
        if isinstance(config.get("modelPath"), str):
            self.model_path = os.path.expanduser(config["modelPath"])
        if _is_number(config.get("contextSize")) and config["contextSize"] > 0:
            self.context_size = int(config["contextSize"])
        if _is_number(config.get("gpuLayers")):
            self.gpu_layers = int(config["gpuLayers"])

        # Drop the cached model if its parameters changed; it reloads lazily.
        if self._loaded_key is not None and self._loaded_key != self._key():
            self.release()

    def state(self) -> JsonRecord:
        return {
            "modelPath": self.model_path,
            "contextSize": self.context_size,
            "gpuLayers": self.gpu_layers,
        }

    def ensure(self, on_loading: Any) -> Any:
        """Return the loaded model, loading it first if needed.

        on_loading(detail) is called before a (slow) load so the service can
        surface a status notification. Raises ImportError when llama-cpp-python
        is missing and whatever the loader raises on a bad model file.
        """
        if self._llama is not None:
            return self._llama

        from llama_cpp import Llama

        on_loading(f"loading model '{os.path.basename(self.model_path)}'")
        self._llama = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_gpu_layers=self.gpu_layers,
            verbose=False,
        )
        self._loaded_key = self._key()
        return self._llama

    def release(self) -> None:
        self._llama = None
        self._loaded_key = None

    def _key(self) -> tuple[str, int, int]:
        return (self.model_path, self.context_size, self.gpu_layers)


class TextGenerationService:
    service_id = TEXT_GENERATION_DESCRIPTOR.service_id
    service_name = TEXT_GENERATION_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._backend = "server"
        self._server_url = DEFAULT_SERVER_URL
        self._model = ""
        self._local = LocalLlm()
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
        # Stream the completion token by token, notifying the growing text as
        # {streamText} for live UI display. The pipeline output is unaffected:
        # the full result is still emitted once, when generation finishes.
        self._stream = True
        self._status = "idle"

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if config.get("backend") in _BACKENDS:
            self._backend = config["backend"]
        if isinstance(config.get("serverUrl"), str) and config["serverUrl"]:
            self._server_url = config["serverUrl"].rstrip("/")
        if isinstance(config.get("model"), str):
            self._model = config["model"]
        self._local.configure(config)
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
        if isinstance(config.get("stream"), bool):
            self._stream = config["stream"]
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "backend": self._backend,
            "serverUrl": self._server_url,
            "model": self._model,
            **self._local.state(),
            "systemPrompt": self._system_prompt,
            "temperature": self._temperature,
            "topP": self._top_p,
            "topK": self._top_k,
            "maxTokens": self._max_tokens,
            "timeoutSec": self._timeout_sec,
            "thinking": self._thinking,
            "stream": self._stream,
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

        if self._backend == "local":
            return self._process_local(messages, notify)
        return self._process_server(messages, notify)

    def _process_server(self, messages: list[JsonRecord], notify: NotifyCallback) -> Any:
        payload: JsonRecord = {
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_tokens": self._max_tokens,
            "stream": self._stream,
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
            if self._stream:
                response = _consume_stream(
                    _post_json_stream(
                        f"{self._server_url}/v1/chat/completions",
                        payload,
                        self._timeout_sec,
                    ),
                    notify,
                )
            else:
                response = _post_json(
                    f"{self._server_url}/v1/chat/completions", payload, self._timeout_sec
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            return self._error(notify, f"server returned HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, OSError):
            return self._error(notify, server_hint(self._server_url))

        return self._emit_result(response, started, notify)

    def _process_local(self, messages: list[JsonRecord], notify: NotifyCallback) -> Any:
        if not self._local.model_path:
            return self._error(
                notify, "local backend needs a modelPath pointing to a .gguf file"
            )

        try:
            llama = self._local.ensure(
                lambda detail: self._set_status(notify, "loading", detail=detail)
            )
        except ImportError:
            return self._error(notify, INSTALL_HINT)
        except Exception as exc:
            return self._error(
                notify, f"failed to load model '{self._local.model_path}': {exc}"
            )

        self._set_status(notify, "generating")
        started = time.monotonic()
        try:
            if self._stream:
                chunks = llama.create_chat_completion(
                    messages=messages,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    top_k=self._top_k,
                    max_tokens=self._max_tokens,
                    stream=True,
                )
                response = _consume_stream(chunks, notify)
            else:
                response = llama.create_chat_completion(
                    messages=messages,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    top_k=self._top_k,
                    max_tokens=self._max_tokens,
                )
        except Exception as exc:
            return self._error(notify, f"generation failed: {exc}")

        return self._emit_result(response, started, notify)

    def _emit_result(self, response: JsonRecord, started: float, notify: NotifyCallback) -> Any:
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
        self._local.release()

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
