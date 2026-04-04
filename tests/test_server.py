"""Integration tests for hkp-python runtime server — ported from hkp-node/tests/server.test.ts."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.server import create_runtime_server
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR
from hkp.services.map_service import MAP_DESCRIPTOR
from hkp.services.monitor import MONITOR_DESCRIPTOR
from hkp.services.timer import TIMER_DESCRIPTOR
from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def server_info():
    """Start a fresh server for each test, stop it after."""
    server = create_runtime_server({"external_host": "127.0.0.1"})
    address = await server.start(0, "127.0.0.1")
    yield server, address
    await server.stop()


@pytest.fixture
def base_url(server_info):
    _, address = server_info
    return address["base_url"]


@pytest.fixture
def port(server_info):
    _, address = server_info
    return address["port"]


# ── Helper ────────────────────────────────────────────────────────────────────


async def _post_json(session: aiohttp.ClientSession, url: str, payload: Any) -> Any:
    async with session.post(url, json=payload) as resp:
        assert resp.status == 200, f"POST {url} returned {resp.status}: {await resp.text()}"
        return await resp.json()


async def _get_json(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as resp:
        assert resp.status == 200, f"GET {url} returned {resp.status}: {await resp.text()}"
        return await resp.json()


async def _delete(session: aiohttp.ClientSession, url: str) -> int:
    async with session.delete(url) as resp:
        return resp.status


async def create_runtime(
    session: aiohttp.ClientSession,
    port: int,
    services: list[dict] | None = None,
) -> dict:
    body = await _post_json(
        session,
        f"http://127.0.0.1:{port}/runtimes",
        {"id": "rt-1", "name": "Python Runtime", "services": services or []},
    )
    return body["runtimes"][0]


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_shape(server_info, base_url, port):
    server, address = server_info
    async with aiohttp.ClientSession() as session:
        body = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes",
            {"id": "rt-1", "name": "Python Runtime", "boardName": "Board A", "services": []},
        )

    # Registry entries omit version/capabilities keys when None
    assert body["registry"] == [
        {"serviceId": "monitor", "serviceName": "Monitor"},
        {"serviceId": "map", "serviceName": "Map", "version": "v1", "capabilities": []},
        {"serviceId": "sub-service", "serviceName": "SubService", "capabilities": ["subservices"]},
        {
            "serviceId": "http-server-subservices",
            "serviceName": "HttpServerSubservices",
            "capabilities": ["subservices"],
        },
        {"serviceId": "hookup.to/service/timer", "serviceName": "Timer"},
    ]

    assert len(body["runtimes"]) == 1
    rt = body["runtimes"][0]
    assert rt["id"] == "rt-1"
    assert rt["name"] == "Python Runtime"
    assert rt["boardName"] == "Board A"
    assert rt["services"] == []
    assert rt["inputs"] == []
    ws_base = base_url.replace("http://", "ws://")
    assert rt["outputUrl"] == f"{ws_base}/rt-1"


@pytest.mark.asyncio
async def test_add_configure_query_remove_service(server_info, port):
    async with aiohttp.ClientSession() as session:
        await create_runtime(session, port)

        # Add
        svc_state = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services",
            {
                "serviceId": MONITOR_DESCRIPTOR.service_id,
                "uuid": "svc-1",
                "state": {"logToConsole": True, "renderTextEditor": False},
            },
        )
        assert svc_state["logToConsole"] is True
        assert svc_state["renderTextEditor"] is False

        # Configure
        configured = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/svc-1",
            {"renderTextEditor": True, "fileLogPath": "/tmp/monitor.log"},
        )
        assert configured["renderTextEditor"] is True
        assert configured["fileLogPath"] == "/tmp/monitor.log"

        # Query
        queried = await _get_json(session, f"http://127.0.0.1:{port}/runtimes/rt-1/services/svc-1")
        assert queried["logToConsole"] is True
        assert queried["renderTextEditor"] is True

        # Property
        prop = await _get_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/svc-1/property/logToConsole",
        )
        assert prop is True

        # Remove
        after_delete = await (
            lambda: session.delete(f"http://127.0.0.1:{port}/runtimes/rt-1/services/svc-1")
        )()
        body = await after_delete.json()
        assert body["services"] == []


@pytest.mark.asyncio
async def test_websocket_process_and_notifications(server_info, port):
    async with aiohttp.ClientSession() as session:
        runtime = await create_runtime(
            session,
            port,
            services=[
                {
                    "serviceId": MONITOR_DESCRIPTOR.service_id,
                    "uuid": "svc-1",
                    "state": {"renderTextEditor": True},
                }
            ],
        )
        ws_url = runtime["outputUrl"]

    collected: list[dict] = []

    async def run_ws() -> list[dict]:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                await ws.send_str(json.dumps({"type": "readwrite", "id": "rt-1"}))
                await ws.send_str(
                    json.dumps(
                        {"type": "processRuntime", "params": {"hello": "world"}, "context": None}
                    )
                )

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        collected.append(data)

                        notifications = [m for m in collected if m.get("type") == "notification"]
                        has_start = any(
                            _safe_parse(m.get("value", "{}"), {}).get("__internal", {}).get("state")
                            == "call-process"
                            for m in notifications
                        )
                        has_finish = any(
                            _safe_parse(m.get("value", "{}"), {}).get("__internal", {}).get("state")
                            == "call-process-finished"
                            for m in notifications
                        )
                        has_payload = any(
                            _safe_parse(m.get("value", "{}"), {}).get("hello") == "world"
                            for m in notifications
                        )
                        has_result = any(m.get("type") == "result" for m in collected)

                        if has_start and has_finish and has_payload and has_result:
                            await ws.close()
                            return collected
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

        return collected

    messages = await asyncio.wait_for(run_ws(), timeout=5.0)

    notifications = [m for m in messages if m.get("type") == "notification"]
    payload_notif = next(
        (m for m in notifications if _safe_parse(m.get("value", "{}"), {}).get("hello") == "world"),
        None,
    )
    start_notif = next(
        (
            m
            for m in notifications
            if _safe_parse(m.get("value", "{}"), {}).get("__internal", {}).get("state")
            == "call-process"
        ),
        None,
    )
    finish_notif = next(
        (
            m
            for m in notifications
            if _safe_parse(m.get("value", "{}"), {}).get("__internal", {}).get("state")
            == "call-process-finished"
        ),
        None,
    )
    result = next((m for m in messages if m.get("type") == "result"), None)

    assert start_notif is not None
    assert finish_notif is not None
    assert payload_notif is not None
    assert _safe_parse(payload_notif["value"], {}) == {"hello": "world"}
    assert payload_notif["instanceId"] == "svc-1"
    assert result is not None
    assert result["data"] == {"hello": "world"}

    # Monitor state must not include message
    async with aiohttp.ClientSession() as session:
        svc_state = await _get_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1/services/svc-1"
        )
    assert "message" not in svc_state


@pytest.mark.asyncio
async def test_map_templates(server_info, port):
    async with aiohttp.ClientSession() as session:
        await create_runtime(
            session,
            port,
            services=[
                {
                    "serviceId": MAP_DESCRIPTOR.service_id,
                    "uuid": "map-1",
                    "state": {
                        "mode": "overwrite",
                        "template": {
                            "greeting": "hello",
                            "count=": "params.count + 1",
                            "meta.kind": "mapped",
                        },
                    },
                }
            ],
        )

        # Overwrite mode
        result = await _post_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1", {"count": 3, "preserved": True}
        )
        assert result == {"count": 4, "preserved": True, "greeting": "hello", "meta": {"kind": "mapped"}}

        # Switch to add mode — should not overwrite existing count
        await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/map-1",
            {"mode": "add", "template": {"count=": "42"}},
        )

        result2 = await _post_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1", {"count": 10}
        )
        assert result2["count"] == 10


@pytest.mark.asyncio
async def test_sub_service_pipeline(server_info, port):
    async with aiohttp.ClientSession() as session:
        await create_runtime(
            session,
            port,
            services=[
                {
                    "serviceId": SUB_SERVICE_DESCRIPTOR.service_id,
                    "uuid": "sub-1",
                    "state": {
                        "pipeline": [
                            {
                                "serviceId": MONITOR_DESCRIPTOR.service_id,
                                "instanceId": "sub-monitor-1",
                                "state": {"logToConsole": False},
                            }
                        ]
                    },
                }
            ],
        )

        # Append a map service
        after_append = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/sub-1",
            {
                "appendService": {
                    "serviceId": MAP_DESCRIPTOR.service_id,
                    "state": {"mode": "replace", "template": {"count=": "params.count + 1"}},
                }
            },
        )
        assert len(after_append["pipeline"]) == 2
        appended_map = next(
            e for e in after_append["pipeline"] if e["serviceId"] == MAP_DESCRIPTOR.service_id
        )
        assert isinstance(appended_map["instanceId"], str) and len(appended_map["instanceId"]) > 0

        result = await _post_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1", {"count": 3}
        )
        assert result == {"count": 4}

        # Configure the appended map
        after_configure = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/sub-1",
            {
                "configureService": {
                    "instanceId": appended_map["instanceId"],
                    "state": {"mode": "add", "template": {"tag": "ok"}},
                }
            },
        )
        map_entry = next(
            e for e in after_configure["pipeline"] if e["instanceId"] == appended_map["instanceId"]
        )
        assert map_entry["state"]["mode"] == "add"

        result2 = await _post_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1", {"count": 3}
        )
        assert result2 == {"tag": "ok", "count": 3}

        # Remove the monitor
        after_remove = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/sub-1",
            {"removeService": "sub-monitor-1"},
        )
        assert len(after_remove["pipeline"]) == 1
        assert after_remove["pipeline"][0]["serviceId"] == MAP_DESCRIPTOR.service_id


@pytest.mark.asyncio
async def test_http_subservice_injects_into_outer_runtime(server_info, port):
    async with aiohttp.ClientSession() as session:
        await create_runtime(
            session,
            port,
            services=[
                {
                    "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                    "uuid": "http-sub-1",
                    "state": {
                        "bypass": False,
                        "mode": "process_on_session",
                        "port": 0,
                        "pipeline": [
                            {
                                "serviceId": MAP_DESCRIPTOR.service_id,
                                "instanceId": "inner-map-1",
                                "state": {
                                    "mode": "replace",
                                    "template": {
                                        "source": "http",
                                        "path=": "params.path",
                                        "method=": "params.method",
                                    },
                                },
                            }
                        ],
                    },
                },
                {
                    "serviceId": MONITOR_DESCRIPTOR.service_id,
                    "uuid": "outer-monitor-1",
                },
            ],
        )

        # Configure bypass=False to ensure server is started; response should include the port
        configure_resp = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/http-sub-1",
            {"bypass": False},
        )
        inner_port = configure_resp["port"]
        assert isinstance(inner_port, int)
        assert inner_port > 0

        ws_url = f"ws://127.0.0.1:{port}/rt-1"

        # Connect WebSocket and then curl the inner server; verify lifecycle notifications
        lifecycle_done = asyncio.Event()
        lifecycle_error: list[Exception] = []

        async def watch_ws() -> None:
            try:
                async with aiohttp.ClientSession() as ws_session:
                    async with ws_session.ws_connect(ws_url) as ws:
                        saw_start = False
                        saw_finish = False
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if data.get("type") != "notification" or data.get("instanceId") != "http-sub-1":
                                    continue
                                payload = _safe_parse(data.get("value", "{}"), {})
                                if payload.get("__internal", {}).get("state") == "call-process":
                                    saw_start = True
                                if payload.get("__internal", {}).get("state") == "call-process-finished":
                                    saw_finish = True
                                if saw_start and saw_finish:
                                    await ws.close()
                                    lifecycle_done.set()
                                    return
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as exc:
                lifecycle_error.append(exc)
                lifecycle_done.set()

        ws_task = asyncio.create_task(watch_ws())
        await asyncio.sleep(0.1)  # let websocket connect

        # Hit the inner HTTP server
        async with aiohttp.ClientSession() as fetch_session:
            async with fetch_session.get(f"http://127.0.0.1:{inner_port}/hello") as resp:
                assert resp.status == 200

        await asyncio.wait_for(lifecycle_done.wait(), timeout=5.0)
        ws_task.cancel()

        assert not lifecycle_error, lifecycle_error[0]

        # Verify pipeline output
        async with aiohttp.ClientSession() as fetch_session:
            async with fetch_session.get(f"http://127.0.0.1:{inner_port}/hello") as resp:
                assert resp.status == 200
                payload = await resp.json()
        assert payload == {"source": "http", "path": "/hello", "method": "GET"}

        # Monitor must not expose message
        svc_state = await _get_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1/services/outer-monitor-1"
        )
        assert "message" not in svc_state


@pytest.mark.asyncio
async def test_timer_periodic_fires_via_websocket(server_info, port):
    """Periodic timer autonomously fires and its results reach WebSocket clients."""
    async with aiohttp.ClientSession() as session:
        runtime = await create_runtime(
            session,
            port,
            services=[
                {
                    "serviceId": TIMER_DESCRIPTOR.service_id,
                    "uuid": "timer-1",
                    "state": {
                        "periodic": True,
                        "periodicValue": 50,
                        "periodicUnit": "ms",
                    },
                }
            ],
        )
        ws_url = runtime["outputUrl"]

    tick_received = asyncio.Event()
    first_result: dict = {}

    async def watch() -> None:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as ws:
                # Start the timer
                async with aiohttp.ClientSession() as s:
                    await _post_json(
                        s,
                        f"http://127.0.0.1:{port}/runtimes/rt-1/services/timer-1",
                        {"start": True},
                    )
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "result" and isinstance(data.get("data"), dict):
                            first_result.update(data["data"])
                            tick_received.set()
                            await ws.close()
                            return
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    task = asyncio.create_task(watch())
    await asyncio.wait_for(tick_received.wait(), timeout=5.0)
    task.cancel()

    assert first_result.get("triggerCount") == 1

    # Stop and verify state
    async with aiohttp.ClientSession() as session:
        state = await _post_json(
            session,
            f"http://127.0.0.1:{port}/runtimes/rt-1/services/timer-1",
            {"stop": True},
        )
    assert state["running"] is False


@pytest.mark.asyncio
async def test_timer_until_auto_stops(server_info, port):
    """Timer with until.triggerCount stops itself after N ticks."""
    collected_results: list[dict] = []
    done = asyncio.Event()

    async with aiohttp.ClientSession() as session:
        runtime = await create_runtime(
            session,
            port,
            services=[
                {
                    "serviceId": TIMER_DESCRIPTOR.service_id,
                    "uuid": "timer-1",
                    "state": {
                        "periodic": True,
                        "periodicValue": 30,
                        "periodicUnit": "ms",
                        "until": {"triggerCount": 3},
                    },
                }
            ],
        )
        ws_url = runtime["outputUrl"]

    async def watch() -> None:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as ws:
                async with aiohttp.ClientSession() as s:
                    await _post_json(
                        s,
                        f"http://127.0.0.1:{port}/runtimes/rt-1/services/timer-1",
                        {"start": True},
                    )
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "result" and isinstance(data.get("data"), dict):
                            collected_results.append(data["data"])
                            if len(collected_results) >= 3:
                                done.set()
                                await ws.close()
                                return
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    task = asyncio.create_task(watch())
    await asyncio.wait_for(done.wait(), timeout=5.0)
    task.cancel()

    assert [r["triggerCount"] for r in collected_results] == [1, 2, 3]

    # Timer should have stopped itself
    await asyncio.sleep(0.15)  # wait longer than one interval to confirm no more ticks
    async with aiohttp.ClientSession() as session:
        state = await _get_json(
            session, f"http://127.0.0.1:{port}/runtimes/rt-1/services/timer-1"
        )
    assert state["running"] is False


# ── Utility ───────────────────────────────────────────────────────────────────


def _safe_parse(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default
