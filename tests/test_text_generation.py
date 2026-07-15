from __future__ import annotations

import urllib.error
from typing import Any

import pytest

from hkp.data import TextData
from hkp.services import text_generation
from hkp.services.text_generation import TextGenerationService, server_hint
from hkp.types import ServiceConfiguration


def _make_service(state: dict | None = None) -> TextGenerationService:
    return TextGenerationService(
        ServiceConfiguration(service_id="text-generation", uuid="llm-1", state=state)
    )


def _collect_notify(sink: list) -> Any:
    def notify(payload: Any, instance_id: str | None = None) -> None:
        sink.append(payload)

    return notify


# ── Stub backend ───────────────────────────────────────────────────────────────


def _chat_response(content: str, **message_extra: Any) -> dict:
    return {
        "model": "Bonsai-27B-Q1_0",
        "choices": [{"message": {"role": "assistant", "content": content, **message_extra}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    }


@pytest.fixture
def fake_server(monkeypatch):
    calls: list = []

    def post_json(url: str, payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return _chat_response("The answer is 42.")

    monkeypatch.setattr(text_generation, "_post_json", post_json)
    return calls


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_configure_and_state_defaults() -> None:
    svc = _make_service()
    state = svc.get_state()
    assert state["serverUrl"] == "http://127.0.0.1:8081"
    assert state["temperature"] == 0.7
    assert state["topP"] == 0.95
    assert state["topK"] == 20
    assert state["status"] == "idle"

    svc.configure(
        {"serverUrl": "http://10.0.0.5:9000/", "systemPrompt": "Be terse", "maxTokens": 128}
    )
    state = svc.get_state()
    assert state["serverUrl"] == "http://10.0.0.5:9000"
    assert state["systemPrompt"] == "Be terse"
    assert state["maxTokens"] == 128


def test_invalid_config_values_ignored() -> None:
    svc = _make_service()
    svc.configure(
        {"serverUrl": "", "temperature": "hot", "topK": True, "maxTokens": -5, "timeoutSec": 0}
    )
    state = svc.get_state()
    assert state["serverUrl"] == "http://127.0.0.1:8081"
    assert state["temperature"] == 0.7
    assert state["topK"] == 20
    assert state["maxTokens"] == 512
    assert state["timeoutSec"] == 300.0


def test_generates_from_string_prompt(fake_server) -> None:
    svc = _make_service({"temperature": 0.2})
    notifications: list = []

    result = svc.process("What is six times seven?", _collect_notify(notifications))

    assert result["text"] == "The answer is 42."
    assert result["model"] == "Bonsai-27B-Q1_0"
    assert result["usage"] == {"promptTokens": 12, "completionTokens": 34}
    assert "durationMs" in result

    call = fake_server[0]
    assert call["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert call["payload"]["messages"] == [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is six times seven?"},
    ]
    assert call["payload"]["temperature"] == 0.2
    assert call["payload"]["stream"] is False

    statuses = [n["status"] for n in notifications if isinstance(n, dict) and "status" in n]
    assert statuses == ["generating", "idle"]


def test_accepts_text_data_and_json_shapes(fake_server) -> None:
    svc = _make_service()
    notify = _collect_notify([])

    assert svc.process(TextData("from text data"), notify)["text"]
    assert svc.process({"prompt": "from prompt key"}, notify)["text"]
    # Output of the speech-to-text service pipes in directly via its `text` key.
    assert svc.process({"text": "from transcription", "language": "en"}, notify)["text"]

    prompts = [c["payload"]["messages"][-1]["content"] for c in fake_server]
    assert prompts == ["from text data", "from prompt key", "from transcription"]


def test_messages_passthrough_prepends_system_prompt_once(fake_server) -> None:
    svc = _make_service()
    notify = _collect_notify([])

    svc.process({"messages": [{"role": "user", "content": "hi"}]}, notify)
    assert fake_server[0]["payload"]["messages"][0]["role"] == "system"

    explicit = [{"role": "system", "content": "custom"}, {"role": "user", "content": "hi"}]
    svc.process({"messages": explicit}, notify)
    assert fake_server[1]["payload"]["messages"] == explicit


def test_thinking_toggle_sent_only_when_configured(fake_server) -> None:
    svc = _make_service()
    notify = _collect_notify([])

    svc.process("hi", notify)
    assert "chat_template_kwargs" not in fake_server[0]["payload"]

    svc.configure({"thinking": False})
    svc.process("hi", notify)
    assert fake_server[1]["payload"]["chat_template_kwargs"] == {"enable_thinking": False}

    svc.configure({"thinking": None})
    svc.process("hi", notify)
    assert "chat_template_kwargs" not in fake_server[2]["payload"]


def test_thinking_is_split_from_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        text_generation,
        "_post_json",
        lambda url, payload, timeout: _chat_response("<think>6*7=42</think>It is 42."),
    )
    svc = _make_service()
    result = svc.process("compute", _collect_notify([]))
    assert result["text"] == "It is 42."
    assert result["thinking"] == "6*7=42"


def test_reasoning_content_field_is_used(monkeypatch) -> None:
    monkeypatch.setattr(
        text_generation,
        "_post_json",
        lambda url, payload, timeout: _chat_response("It is 42.", reasoning_content="6*7=42"),
    )
    svc = _make_service()
    result = svc.process("compute", _collect_notify([]))
    assert result["text"] == "It is 42."
    assert result["thinking"] == "6*7=42"


def test_unreachable_server_reports_hint(monkeypatch) -> None:
    def refuse(url: str, payload: dict, timeout: float) -> dict:
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr(text_generation, "_post_json", refuse)
    svc = _make_service()
    result = svc.process("hello", _collect_notify([]))
    assert result == {"error": server_hint("http://127.0.0.1:8081")}
    assert svc.get_state()["status"] == "error"


def test_http_error_reports_status_code(monkeypatch) -> None:
    import io

    def fail(url: str, payload: dict, timeout: float) -> dict:
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, io.BytesIO(b"loading"))

    monkeypatch.setattr(text_generation, "_post_json", fail)
    svc = _make_service()
    result = svc.process("hello", _collect_notify([]))
    assert "HTTP 503" in result["error"]
    assert "loading" in result["error"]


def test_unsupported_input_yields_error() -> None:
    svc = _make_service()
    assert "error" in svc.process(b"binary", _collect_notify([]))
    assert "error" in svc.process({"some": "json"}, _collect_notify([]))
    assert "error" in svc.process("   ", _collect_notify([]))


def test_none_input_passes_through() -> None:
    svc = _make_service()
    assert svc.process(None, _collect_notify([])) is None
