from __future__ import annotations

import urllib.error
from typing import Any

import aiohttp
import pytest

from hkp.data import ControlFlowData, TextData
from hkp.services import skill_router
from hkp.services.skill_router import SkillRouterService, server_hint
from hkp.types import ServiceConfiguration


def _unwrap(result: Any) -> Any:
    assert isinstance(result, ControlFlowData), result
    return result.result

SKILLS = [
    {
        "action": "send notification",
        "board": "send ntfy",
        "payload": {"topic": "the ntfy topic", "message": "the message text"},
    },
    {
        "action": "set timer",
        "board": "timer board",
        "payload": {"seconds": "duration in seconds"},
    },
]


def _make_service(state: dict | None = None) -> SkillRouterService:
    base = {"skills": SKILLS}
    if state:
        base.update(state)
    return SkillRouterService(
        ServiceConfiguration(service_id="skill-router", uuid="router-1", state=base)
    )


def _collect_notify(sink: list) -> Any:
    def notify(payload: Any, instance_id: str | None = None) -> None:
        sink.append(payload)

    return notify


def _chat_response(content: str, **message_extra: Any) -> dict:
    return {
        "model": "Bonsai-27B-Q1_0",
        "choices": [{"message": {"role": "assistant", "content": content, **message_extra}}],
    }


@pytest.fixture
def fake_server(monkeypatch):
    """Returns calls; set calls.response to control the model's reply."""
    calls: list = []
    responses = {"value": _chat_response('{"action": null}')}

    def post_json(url: str, payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return responses["value"]

    monkeypatch.setattr(skill_router, "_post_json", post_json)
    return calls, responses


# ── Config ─────────────────────────────────────────────────────────────────────


def test_configure_and_state_defaults() -> None:
    svc = _make_service()
    state = svc.get_state()
    assert state["serverUrl"] == "http://127.0.0.1:8081"
    assert state["temperature"] == 0.1
    assert state["skills"] == SKILLS
    assert state["status"] == "idle"


def test_malformed_skills_are_dropped() -> None:
    svc = _make_service(
        {
            "skills": [
                SKILLS[0],
                {"action": "no board", "payload": {}},
                {"action": "", "board": "b", "payload": {}},
                "not a dict",
                {"action": "no payload", "board": "b"},
            ]
        }
    )
    assert svc.get_state()["skills"] == [SKILLS[0]]


# ── Routing ────────────────────────────────────────────────────────────────────


def test_matches_skill_and_extracts_payload(fake_server) -> None:
    calls, responses = fake_server
    responses["value"] = _chat_response(
        '{"action": "send notification", "payload": {"topic": "X", "message": "hello"}}'
    )
    svc = _make_service()
    notifications: list = []

    result = svc.process(
        "send a notification to topic X and message hello", _collect_notify(notifications)
    )

    # Matches early-return so services after the router are skipped.
    assert _unwrap(result) == {
        "board": "send ntfy",
        "payload": {"topic": "X", "message": "hello"},
    }

    call = calls[0]
    assert call["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert call["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    user_msg = call["payload"]["messages"][1]["content"]
    assert "send notification" in user_msg
    assert "send a notification to topic X" in user_msg

    statuses = [n["status"] for n in notifications if isinstance(n, dict) and "status" in n]
    assert statuses == ["routing", "idle"]
    matched = next(n for n in notifications if isinstance(n, dict) and "matched" in n)
    assert matched["matched"] == "send notification"
    assert matched["board"] == "send ntfy"

    # The composed prompt is notified and inspectable in the service state.
    prompt_notification = next(
        n for n in notifications if isinstance(n, dict) and "lastPrompt" in n
    )
    assert prompt_notification["lastPrompt"] == call["payload"]["messages"]
    assert svc.get_state()["lastPrompt"] == call["payload"]["messages"]


def test_no_match_returns_none(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response('{"action": null}')
    svc = _make_service()
    notifications: list = []
    assert svc.process("what is the weather like", _collect_notify(notifications)) is None
    matched = next(n for n in notifications if isinstance(n, dict) and "matched" in n)
    assert matched["matched"] is None


def test_unknown_action_returns_none(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response('{"action": "made-up skill", "payload": {}}')
    svc = _make_service()
    assert svc.process("do something", _collect_notify([])) is None


def test_unparseable_output_returns_none(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response("I think the best skill would be...")
    svc = _make_service()
    assert svc.process("do something", _collect_notify([])) is None


def test_action_matching_is_case_insensitive_and_json_may_be_wrapped(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response(
        'Sure! {"action": "Send Notification", "payload": {"topic": "t", "message": "m"}} done'
    )
    svc = _make_service()
    result = svc.process("notify t with m", _collect_notify([]))
    assert _unwrap(result) == {"board": "send ntfy", "payload": {"topic": "t", "message": "m"}}


def test_inline_think_tags_are_stripped(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response(
        '<think>{"action": "set timer"} hmm no</think>{"action": "send notification", '
        '"payload": {"topic": "a", "message": "b"}}'
    )
    svc = _make_service()
    result = svc.process("notify a with b", _collect_notify([]))
    assert _unwrap(result)["board"] == "send ntfy"


def test_extra_payload_fields_are_filtered(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response(
        '{"action": "set timer", "payload": {"seconds": 30, "evil": "field"}}'
    )
    svc = _make_service()
    result = svc.process("timer for 30 seconds", _collect_notify([]))
    assert _unwrap(result) == {"board": "timer board", "payload": {"seconds": 30}}


def test_no_configured_skills_returns_none_without_calling_llm(fake_server) -> None:
    calls, _ = fake_server
    svc = _make_service({"skills": []})
    assert svc.process("send a notification", _collect_notify([])) is None
    assert calls == []


# ── noMatch: forward ───────────────────────────────────────────────────────────


def test_no_match_forward_passes_original_input_through(fake_server) -> None:
    _, responses = fake_server
    responses["value"] = _chat_response('{"action": null}')
    svc = _make_service({"noMatch": "forward"})
    # The stt output shape flows on unchanged so a text-generation fallback
    # can pick up its `text` key.
    original = {"text": "what is the weather like", "language": "en"}
    assert svc.process(original, _collect_notify([])) is original


def test_forward_without_configured_skills_passes_input_through(fake_server) -> None:
    calls, _ = fake_server
    svc = _make_service({"skills": [], "noMatch": "forward"})
    assert svc.process("hello", _collect_notify([])) == "hello"
    assert calls == []


def test_invalid_no_match_value_is_ignored() -> None:
    svc = _make_service({"noMatch": "explode"})
    assert svc.get_state()["noMatch"] == "stop"


# ── Pipeline integration (early return / forward through a real runtime) ───────


async def _process_via_server(monkeypatch, model_reply: str, no_match: str):
    monkeypatch.setattr(
        skill_router, "_post_json", lambda url, payload, timeout: _chat_response(model_reply)
    )
    from hkp.server import create_runtime_server

    server = create_runtime_server({"external_host": "127.0.0.1"})
    addr = await server.start(0, "127.0.0.1")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{addr['base_url']}/runtimes",
                json={
                    "id": "rt-1",
                    "name": "Python",
                    "services": [
                        {
                            "serviceId": "skill-router",
                            "uuid": "router",
                            "state": {"skills": SKILLS, "noMatch": no_match},
                        },
                        {
                            # Stand-in for the text-generation fallback: visibly
                            # transforms anything that reaches it.
                            "serviceId": "map",
                            "uuid": "fallback",
                            "state": {"mode": "replace", "template": {"fallback": True}},
                        },
                    ],
                },
            ) as resp:
                assert resp.status == 200
            async with session.post(
                f"{addr['base_url']}/runtimes/rt-1", json={"text": "notify t with m"}
            ) as resp:
                assert resp.status == 200
                return await resp.json()
    finally:
        await server.stop()


async def test_match_early_returns_past_downstream_services(monkeypatch) -> None:
    result = await _process_via_server(
        monkeypatch,
        '{"action": "send notification", "payload": {"topic": "t", "message": "m"}}',
        no_match="forward",
    )
    # The map service after the router must have been skipped.
    assert result == {"board": "send ntfy", "payload": {"topic": "t", "message": "m"}}


async def test_no_match_forward_reaches_downstream_services(monkeypatch) -> None:
    result = await _process_via_server(monkeypatch, '{"action": null}', no_match="forward")
    # The forwarded input flowed into the map service.
    assert result == {"fallback": True}


# ── Inputs / errors ────────────────────────────────────────────────────────────


def test_accepts_text_data_and_json_shapes(fake_server) -> None:
    calls, responses = fake_server
    responses["value"] = _chat_response('{"action": null}')
    svc = _make_service()
    notify = _collect_notify([])

    assert svc.process(TextData("from text data"), notify) is None
    # Output of the speech-to-text service pipes in via its `text` key.
    assert svc.process({"text": "from transcript"}, notify) is None
    assert svc.process({"prompt": "from prompt"}, notify) is None
    assert len(calls) == 3


def test_unreachable_server_reports_hint(monkeypatch) -> None:
    def refuse(url: str, payload: dict, timeout: float) -> dict:
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr(skill_router, "_post_json", refuse)
    svc = _make_service()
    result = svc.process("notify x", _collect_notify([]))
    assert result == {"error": server_hint("http://127.0.0.1:8081")}
    assert svc.get_state()["status"] == "error"


def test_unsupported_input_yields_error(fake_server) -> None:
    svc = _make_service()
    assert "error" in svc.process(b"binary", _collect_notify([]))
    assert "error" in svc.process({"some": "json"}, _collect_notify([]))


def test_none_input_passes_through() -> None:
    svc = _make_service()
    assert svc.process(None, _collect_notify([])) is None
