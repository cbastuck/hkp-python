"""Handing a running runtime its values, over the wire, and inward.

The unit tests say what the vault does with them; this says a browser can
actually deliver them — the route exists, the verb is one the server's CORS
allowlist permits, and what comes back names aliases rather than values. It also
says a service inside a nested pipeline resolves against the runtime around it,
which is the difference between a credential service being usable anywhere and
being usable only at the top level.
"""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.secrets import read_secrets_payload, resolve_credential
from hkp.server import create_runtime_server
from hkp.services.monitor import MONITOR_DESCRIPTOR


@pytest_asyncio.fixture
async def server():
    started = create_runtime_server({"external_host": "127.0.0.1"})
    address = await started.start(0, "127.0.0.1")
    yield started, address["base_url"]
    await started.stop()


async def provision(base_url: str, **extra: Any) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes",
            json={
                "id": "rt-1",
                "name": "Python",
                "services": [
                    {"serviceId": MONITOR_DESCRIPTOR.service_id, "uuid": "mon-1"}
                ],
                **extra,
            },
        ) as res:
            assert res.status == 200


@pytest.mark.asyncio
async def test_takes_values_and_answers_with_the_aliases_it_holds(server) -> None:
    _, base_url = server
    await provision(base_url)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes/rt-1/secrets", json={"gmail.imap": {"value": "hunter2"}}
        ) as res:
            assert res.status == 200
            assert await res.json() == {"aliases": ["gmail.imap"]}


@pytest.mark.asyncio
async def test_merges_so_sending_one_entry_does_not_strip_the_others(server) -> None:
    _, base_url = server
    await provision(base_url, secrets={"smtp": {"value": "first"}})

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes/rt-1/secrets", json={"slack": {"value": "xoxb"}}
        ) as res:
            assert sorted((await res.json())["aliases"]) == ["slack", "smtp"]


@pytest.mark.asyncio
async def test_has_no_way_to_read_a_value_back_out(server) -> None:
    _, base_url = server
    await provision(base_url, secrets={"smtp": {"value": "hunter2"}})

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/runtimes/rt-1/secrets") as res:
            assert res.status in (404, 405)
        # Nor anywhere the runtime describes itself.
        async with session.get(f"{base_url}/runtimes") as res:
            assert "hunter2" not in await res.text()


@pytest.mark.asyncio
async def test_uses_a_verb_the_browser_is_allowed_to_send(server) -> None:
    # What the equivalent route first got wrong elsewhere: the route worked, and
    # the preflight refused it.
    _, base_url = server
    await provision(base_url)

    async with aiohttp.ClientSession() as session:
        async with session.options(
            f"{base_url}/runtimes/rt-1/secrets",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        ) as res:
            assert res.status in (200, 204)
            assert "POST" in res.headers.get("Access-Control-Allow-Methods", "")


@pytest.mark.asyncio
async def test_answers_404_for_a_runtime_that_does_not_exist(server) -> None:
    _, base_url = server

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/runtimes/nope/secrets", json={"a": {"value": "1"}}
        ) as res:
            assert res.status == 404


class TestInsideANestedPipeline:
    """Nothing provisions a nested runtime, so it asks the one around it."""

    def _nested(self, secrets: dict) -> Any:
        from hkp.runtime import HostedRuntime
        from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
        from hkp.types import RuntimeConfiguration, ServiceConfiguration

        resolved: dict[str, Any] = {}

        class Credentialed:
            service_id = "credentialed"
            service_name = "Credentialed"

            def __init__(self, config: ServiceConfiguration) -> None:
                self.uuid = config.uuid
                self._host: Any = None

            def set_host(self, host: Any) -> None:
                self._host = host

            def get_state(self) -> dict:
                return {}

            def configure(self, config: dict) -> dict:
                return {}

            def process(self, input: Any, _notify: Any) -> Any:
                out = resolve_credential(
                    self._host.secrets() if self._host else None,
                    "{{secret.api}}",
                    "api.example.com",
                )
                resolved["value"] = out.value
                resolved["problem"] = out.problem
                return input

        def create_service(config: ServiceConfiguration) -> Any:
            if config.service_id == SUB_SERVICE_DESCRIPTOR.service_id:
                return SubService(config, create_service)
            return Credentialed(config)

        runtime = HostedRuntime(
            RuntimeConfiguration(
                id="rt-1",
                name="Python",
                secrets=read_secrets_payload(secrets),
                services=[
                    ServiceConfiguration(
                        service_id=SUB_SERVICE_DESCRIPTOR.service_id,
                        uuid="nest",
                        state={
                            "pipeline": [
                                {"serviceId": "credentialed", "instanceId": "inner"}
                            ]
                        },
                    )
                ],
            ),
            create_service,
        )
        return runtime, resolved

    def test_resolves_against_the_runtime_around_it(self) -> None:
        runtime, resolved = self._nested({"api": {"value": "sk-1"}})

        runtime.process({}, lambda *_: None)

        assert resolved["problem"] == ""
        assert resolved["value"] == "sk-1"

    def test_sees_a_value_pushed_after_the_board_was_running(self) -> None:
        # Asked for on each use rather than copied down when the pipeline was
        # built.
        runtime, resolved = self._nested({})

        runtime.process({}, lambda *_: None)
        assert resolved["value"] is None

        runtime.set_secrets(read_secrets_payload({"api": {"value": "arrived"}}))
        runtime.process({}, lambda *_: None)

        assert resolved["value"] == "arrived"


class TestAPassRunsOffTheLoop:
    """A board drives a runtime over HTTP, and the pass runs on a worker thread.

    A service that starts work during that pass — a request, a timer — has no
    running loop to schedule it on, so it schedules through the runtime, which
    knows the server's. Every test that calls ``process`` from inside a loop
    misses this, which is how it went unnoticed: the service reported "no event
    loop" into a callback the runtime-wide path discards, so a board saw
    silence rather than an error.
    """

    @pytest.mark.asyncio
    async def test_a_request_started_from_a_worker_thread_is_still_sent(
        self, server
    ) -> None:
        from aiohttp import web

        seen: list[dict] = []

        async def sink(request: web.Request) -> web.Response:
            seen.append({k.lower(): v for k, v in request.headers.items()})
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", sink)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        _started, base_url = server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/runtimes",
                json={
                    "id": "rt-client",
                    "name": "Python",
                    "services": [
                        {
                            "serviceId": "http-client",
                            "uuid": "cli-1",
                            "state": {
                                "url": f"http://127.0.0.1:{port}/hook",
                                "method": "get",
                                "headers": {"custom": "{{secret.gmail.imap}}"},
                                "bypass": False,
                            },
                        }
                    ],
                    "secrets": {"gmail.imap": {"value": "hunter2"}},
                },
            ) as res:
                assert res.status == 200

            # The runtime-wide entry point, which is what a board uses.
            async with session.post(f"{base_url}/runtimes/rt-client", json={}) as res:
                assert res.status == 200

            for _ in range(40):
                if seen:
                    break
                await asyncio.sleep(0.05)

        await runner.cleanup()

        assert seen, "the request was never sent"
        # And the credential was resolved for it, never held in state.
        assert seen[0]["custom"] == "hunter2"
