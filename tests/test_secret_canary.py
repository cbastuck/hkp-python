"""A secret's value, used and then looked for everywhere it must not be.

A board names a secret; the value reaches the runtime separately and is used
where the service needs it — here, a header an HTTP Client sends. It must come
back out nowhere else: not in what the runtime reports or saves, not in what it
notifies or logs, and not in an error. The upstream fails, so the error paths
run too. The same sweep is in hkp-node (``tests/secret-canary.test.ts``) and
hkp-rt (``tests/secret_canary.test.cpp``).
"""
from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import pytest
from aiohttp import web

from hkp.server import create_runtime_server
from hkp.services.http_client import HTTP_CLIENT_DESCRIPTOR
from hkp.services.monitor import MONITOR_DESCRIPTOR

CANARY = "canary-6f1d0e2a-secret-value"


@pytest.mark.asyncio
async def test_a_secret_leaves_the_runtime_only_where_its_service_uses_it(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    received: list[str] = []

    async def fail(request: web.Request) -> web.Response:
        received.append(json.dumps(dict(request.headers)))
        return web.Response(status=500, text="upstream failed")

    upstream = web.Application()
    upstream.router.add_post("/hook", fail)
    runner = web.AppRunner(upstream)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    server = create_runtime_server({"external_host": "127.0.0.1"})
    base_url = (await server.start(0, "127.0.0.1"))["base_url"]
    frames: list[str] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/runtimes",
                json={
                    "id": "rt",
                    "name": "Python",
                    "services": [
                        {
                            "serviceId": HTTP_CLIENT_DESCRIPTOR.service_id,
                            "uuid": "call",
                            "state": {
                                "url": f"http://127.0.0.1:{port}/hook",
                                "method": "POST",
                                "headers": {"Authorization": "Bearer {{secret.api}}"},
                            },
                        },
                        {"serviceId": MONITOR_DESCRIPTOR.service_id, "uuid": "watch"},
                    ],
                    "secrets": {"api": {"value": CANARY}},
                },
            ) as res:
                assert res.status == 200

            async with session.ws_connect(base_url.replace("http", "ws") + "/rt") as ws:
                await ws.send_str(json.dumps({"type": "readwrite", "id": "rt"}))

                async def collect() -> None:
                    async for message in ws:
                        frames.append(
                            message.data
                            if isinstance(message.data, str)
                            else bytes(message.data).decode("latin1")
                        )

                collecting = asyncio.create_task(collect())
                async with session.post(f"{base_url}/runtimes/rt", json={"go": True}) as res:
                    answer = await res.text()
                for _ in range(100):
                    if received:
                        break
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.1)
                collecting.cancel()

            # Used: it went where the board sent it.
            assert CANARY in "".join(received)

            surfaces = {"the run's answer": answer, "the socket": "\n".join(frames)}
            for where, path in {
                "the runtimes": "/runtimes",
                "the runtime": "/runtimes/rt",
                "its services": "/runtimes/rt/services",
                "the service using it": "/runtimes/rt/services/call",
            }.items():
                async with session.get(f"{base_url}{path}") as res:
                    surfaces[where] = await res.text()
    finally:
        await server.stop()
        await runner.cleanup()

    captured = capsys.readouterr()
    surfaces["the log"] = caplog.text
    surfaces["the console"] = captured.out + captured.err
    for where, text in surfaces.items():
        assert CANARY not in text, where
