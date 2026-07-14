"""Binary (YAS) transport tests for the runtime server."""
from __future__ import annotations

import json

import aiohttp
import pytest
import pytest_asyncio

from hkp.data import FloatRingBuffer
from hkp.server import create_runtime_server
from hkp.yas import MessagePurpose, deserialize_message, serialize_message


@pytest_asyncio.fixture
async def server_info():
    server = create_runtime_server({"external_host": "127.0.0.1"})
    address = await server.start(0, "127.0.0.1")
    yield server, address
    await server.stop()


@pytest.fixture
def port(server_info):
    _, address = server_info
    return address["port"]


async def _create_monitor_runtime(session: aiohttp.ClientSession, port: int) -> None:
    async with session.post(
        f"http://127.0.0.1:{port}/runtimes",
        json={
            "id": "rt-1",
            "name": "Python Runtime",
            "services": [
                {"serviceId": "monitor", "uuid": "mon-1", "serviceName": "Monitor"}
            ],
        },
    ) as resp:
        assert resp.status == 200


@pytest.mark.asyncio
async def test_post_yas_ring_buffer_returns_yas_result(port: int) -> None:
    """Monitor passes input through, so a YAS POST comes back as a YAS result."""
    rb = FloatRingBuffer.from_floats([0.1, 0.2, 0.3], id=5, ts=1000)
    frame = serialize_message(rb, sender="test")

    async with aiohttp.ClientSession() as session:
        await _create_monitor_runtime(session, port)
        async with session.post(
            f"http://127.0.0.1:{port}/runtimes/rt-1",
            data=frame,
            headers={"Content-Type": "application/octet-stream"},
        ) as resp:
            assert resp.status == 200
            assert resp.content_type == "application/octet-stream"
            message = deserialize_message(await resp.read())

    assert message.purpose == MessagePurpose.RESULT
    assert isinstance(message.data, FloatRingBuffer)
    assert message.data.num_samples == 3
    assert message.data.id == 5


@pytest.mark.asyncio
async def test_websocket_binary_frame_is_processed(port: int) -> None:
    rb = FloatRingBuffer.from_floats([1.0, -1.0], id=9, ts=42)
    frame = serialize_message(rb, sender="")

    async with aiohttp.ClientSession() as session:
        await _create_monitor_runtime(session, port)
        async with session.ws_connect(f"http://127.0.0.1:{port}/rt-1") as ws:
            await ws.send_str(json.dumps({"type": "readwrite", "id": "rt-1"}))
            await ws.send_bytes(frame)

            # Skip notification frames until the binary result arrives
            for _ in range(20):
                msg = await ws.receive(timeout=5)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    result = deserialize_message(msg.data)
                    break
            else:
                pytest.fail("no binary result received")

    assert result.purpose == MessagePurpose.RESULT
    assert isinstance(result.data, FloatRingBuffer)
    assert list(result.data.to_floats()) == [1.0, -1.0]


@pytest.mark.asyncio
async def test_json_post_still_works(port: int) -> None:
    async with aiohttp.ClientSession() as session:
        await _create_monitor_runtime(session, port)
        async with session.post(
            f"http://127.0.0.1:{port}/runtimes/rt-1", json={"hello": "world"}
        ) as resp:
            assert resp.status == 200
            assert await resp.json() == {"hello": "world"}
