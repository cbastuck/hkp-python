from __future__ import annotations

# Service Documentation
# Service ID: skill-router
# Service Name: Skill Router
# Runtime: hkp-python
# Modes: route (one LLM call, strict JSON output), via one of two backends
#        selected by the `backend` state — server (default, OpenAI-compatible
#        HTTP) or local (in-process GGUF via llama-cpp-python), exactly as in
#        the text-generation service
# Key Config: skills (array of {action, board, payload}), backend,
#             serverUrl, model (server backend),
#             modelPath, contextSize, gpuLayers (local backend),
#             temperature, maxTokens, timeoutSec,
#             stream (default true — consume the model output token by token,
#             notifying the growing text as {streamText} so the service UI's
#             process view can show the routing decision being generated; the
#             pipeline output is unaffected and still emitted once, at the end)
# IO: in=String/TextData (the request) or JSON ({text} | {prompt})
#     -> out=JSON { board, payload } for the matched skill, or None (stop)
# Arrays: n/a
# Binary: not supported; non-text inputs yield an error JSON
# MixedData: not supported
#
# Matches free-form text (e.g. a voice transcript) against a configured set of
# skills and extracts the payload parameters, using the same local
# LLM backends as the text-generation service. On a match the
# routing decision is final: the result is early-returned (ControlFlowData),
# skipping any services after the router in the same runtime. The noMatch
# config decides the other branch: "stop" (default) ends the pipeline,
# "forward" passes the original input to the remaining services — e.g. a
# text-generation fallback that answers requests no skill covers.
# A skill looks like:
#
#   { "action": "send notification", "board": "send ntfy",
#     "payload": { "topic": "the ntfy topic", "message": "the message text" } }
#
# The payload template's keys are the parameter names; its values describe to
# the model what to extract. On a match the service emits
# { "board": <skill.board>, "payload": { ...extracted values... } } — a browser
# Board-Service in "input" board-source mode then plays that saved board with
# the payload. No match (or unparseable model output) returns None, which
# stops the pipeline. Backend errors return an error JSON so they stay visible
# on monitors.

import json
import time
import urllib.error
from typing import Any

from ..data import ControlFlowData, TextData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry
from .text_generation import (
    _BACKENDS,
    INSTALL_HINT,
    THINK_CLOSE,
    THINK_OPEN,
    LocalLlm,
    _consume_stream,
    _post_json,
    _post_json_stream,
    server_hint,
)

SKILL_ROUTER_DESCRIPTOR = ServiceRegistryEntry(
    service_id="skill-router",
    service_name="Skill Router",
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8081"
# Routing wants determinism, not creativity.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SEC = 120.0

SYSTEM_PROMPT = (
    "You route user requests to skills. You are given a list of skills, each "
    "with an \"action\" describing what it does and a \"payload\" object whose "
    "keys are parameter names and whose values describe what to extract from "
    "the request.\n"
    "Reply with ONLY a JSON object, no other text:\n"
    '- If one skill matches: {"action": "<the matched skill\'s action>", '
    '"payload": {<parameter name>: <extracted value>, ...}}\n'
    '- If no skill matches: {"action": null}\n'
    "Extract payload values verbatim from the request where possible. Omit "
    "parameters the request does not mention."
)


def _is_skill(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("action"), str)
        and bool(entry["action"])
        and isinstance(entry.get("board"), str)
        and bool(entry["board"])
        and isinstance(entry.get("payload"), dict)
    )


def _strip_thinking(message: JsonRecord) -> str:
    """Thinking models put reasoning in reasoning_content or inline
    <think> tags; routing only cares about the answer."""
    content = message.get("content") or ""
    if THINK_CLOSE in content:
        _, _, content = content.partition(THINK_CLOSE)
    return content.replace(THINK_OPEN, "").strip()


def _extract_json_object(text: str) -> JsonRecord | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class SkillRouterService:
    service_id = SKILL_ROUTER_DESCRIPTOR.service_id
    service_name = SKILL_ROUTER_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._backend = "server"
        self._server_url = DEFAULT_SERVER_URL
        self._model = ""
        self._local = LocalLlm()
        self._skills: list[JsonRecord] = []
        self._temperature = DEFAULT_TEMPERATURE
        self._max_tokens = DEFAULT_MAX_TOKENS
        self._timeout_sec = DEFAULT_TIMEOUT_SEC
        self._no_match: str = "stop"
        self._stream = True
        self._status = "idle"
        # The most recent LLM call — request text, composed messages, raw model
        # output, and the routing decision — inspectable in the UI to debug
        # prompt composition and the model's routing behavior.
        self._last_prompt: list[JsonRecord] = []
        self._last_request = ""
        self._last_output = ""
        self._last_match: JsonRecord | None = None

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
        if isinstance(config.get("skills"), list):
            self._skills = [s for s in config["skills"] if _is_skill(s)]
        if isinstance(config.get("temperature"), (int, float)) and not isinstance(
            config.get("temperature"), bool
        ):
            self._temperature = float(config["temperature"])
        if (
            isinstance(config.get("maxTokens"), (int, float))
            and not isinstance(config.get("maxTokens"), bool)
            and config["maxTokens"] > 0
        ):
            self._max_tokens = int(config["maxTokens"])
        if (
            isinstance(config.get("timeoutSec"), (int, float))
            and not isinstance(config.get("timeoutSec"), bool)
            and config["timeoutSec"] > 0
        ):
            self._timeout_sec = float(config["timeoutSec"])
        if config.get("noMatch") in ("stop", "forward"):
            self._no_match = config["noMatch"]
        if isinstance(config.get("stream"), bool):
            self._stream = config["stream"]
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "backend": self._backend,
            "serverUrl": self._server_url,
            "model": self._model,
            **self._local.state(),
            "skills": self._skills,
            "temperature": self._temperature,
            "maxTokens": self._max_tokens,
            "timeoutSec": self._timeout_sec,
            "noMatch": self._no_match,
            "stream": self._stream,
            "status": self._status,
            "lastRequest": self._last_request,
            "lastPrompt": self._last_prompt,
            "lastOutput": self._last_output,
            "lastMatch": self._last_match,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if input is None:
            return None

        text = self._to_text(input)
        if text is None:
            return self._error(
                notify, "skill-router expects String input or JSON with 'text' or 'prompt'"
            )
        if not self._skills:
            self._set_status(notify, "idle")
            notify({"matched": None, "reason": "no skills configured"})
            return input if self._no_match == "forward" else None

        # Routing output must not be eaten by an invisible thinking budget.
        # The server backend disables thinking via chat_template_kwargs below;
        # the local backend has no such switch, so use the Qwen-style prompt
        # marker instead (harmless for non-thinking models).
        system_prompt = SYSTEM_PROMPT + (" /no_think" if self._backend == "local" else "")
        messages: list[JsonRecord] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Skills:\n"
                + json.dumps(
                    [{"action": s["action"], "payload": s["payload"]} for s in self._skills]
                )
                + "\n\nRequest:\n"
                + text,
            },
        ]

        self._last_request = text
        self._last_prompt = messages
        notify({"lastRequest": text, "lastPrompt": self._last_prompt})

        if self._backend == "local":
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

            self._set_status(notify, "routing")
            started = time.monotonic()
            try:
                if self._stream:
                    chunks = llama.create_chat_completion(
                        messages=messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        stream=True,
                    )
                    response = _consume_stream(chunks, notify)
                else:
                    response = llama.create_chat_completion(
                        messages=messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
            except Exception as exc:
                return self._error(notify, f"routing failed: {exc}")
        else:
            payload: JsonRecord = {
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "stream": self._stream,
                # llama-server extension; harmless elsewhere.
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if self._model:
                payload["model"] = self._model

            self._set_status(notify, "routing")
            started = time.monotonic()
            try:
                if self._stream:
                    chunks = _post_json_stream(
                        f"{self._server_url}/v1/chat/completions", payload, self._timeout_sec
                    )
                    response = _consume_stream(chunks, notify)
                else:
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

        raw_output = _strip_thinking(message)
        self._last_output = raw_output
        notify({"lastOutput": raw_output})

        decision = _extract_json_object(raw_output)
        duration_ms = int((time.monotonic() - started) * 1000)
        self._set_status(notify, "idle")

        skill = self._find_skill(decision)
        if skill is None:
            # No match, unknown action, or unparseable output — all mean
            # "nothing to dispatch": stop the pipeline, or forward the
            # original input to the services after the router.
            self._last_match = {"matched": None, "durationMs": duration_ms}
            notify(dict(self._last_match))
            return input if self._no_match == "forward" else None

        extracted = decision.get("payload") if isinstance(decision, dict) else None
        extracted = extracted if isinstance(extracted, dict) else {}
        # Only parameters the skill declares; the model must not smuggle in
        # extra fields.
        result_payload = {k: extracted[k] for k in skill["payload"] if k in extracted}

        result = {"board": skill["board"], "payload": result_payload}
        self._last_match = {"matched": skill["action"], "durationMs": duration_ms, **result}
        notify(dict(self._last_match))
        # The routing decision is final — skip any services after the router
        # (e.g. a text-generation fallback) and emit the dispatch payload.
        return ControlFlowData(result)

    def set_host(self, host: Any) -> None:
        pass

    def destroy(self) -> None:
        self._local.release()

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

    def _find_skill(self, decision: JsonRecord | None) -> JsonRecord | None:
        if not isinstance(decision, dict) or not isinstance(decision.get("action"), str):
            return None
        action = decision["action"].strip().lower()
        for skill in self._skills:
            if skill["action"].strip().lower() == action:
                return skill
        return None

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
