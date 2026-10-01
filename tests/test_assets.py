"""Content a board declares once and names by reference.

A runtime is handed descriptors and resolves a reference when a service uses it.
What is pinned here: the store resolves each source and says why when it cannot,
an edit reaches the next use without anything being reconfigured, nested
pipelines see their host's assets, and an endpoint serves an asset named as its
response body. Mirrors hkp-node's ``tests/assets.test.ts``.
"""
from __future__ import annotations

import base64
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from hkp.assets import AssetStore, parse_asset_ref, read_assets_payload, referenced_assets
from hkp.runtime import HostedRuntime
from hkp.secrets import SecretVault, read_secrets_payload
from hkp.server import create_runtime_server
from hkp.services.asset import AssetService
from hkp.services.http_server import HTTP_SERVER_SUBSERVICES_DESCRIPTOR
from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
from hkp.types import RuntimeConfiguration, ServiceConfiguration


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TestReferences:
    def test_are_whole_values(self) -> None:
        assert parse_asset_ref("hkp-asset://player") == "player"
        assert parse_asset_ref("<script>hkp-asset://player</script>") is None
        assert parse_asset_ref("hkp-asset://") is None
        assert parse_asset_ref(42) is None

    def test_are_found_anywhere_a_string_mentions_one(self) -> None:
        state = {
            "body": "hkp-asset://page",
            "pipeline": [{"state": {"body=": "p == '/app.js' ? 'hkp-asset://app' : 'hkp-asset://page'"}}],
        }
        assert sorted(referenced_assets(state)) == ["app", "page"]

    def test_the_payload_keeps_descriptors_with_exactly_one_source(self) -> None:
        entries = read_assets_payload(
            {
                "page": {"mediaType": "text/html", "text": "<p>hi</p>"},
                "both": {"mediaType": "text/plain", "text": "a", "url": "https://x"},
                "gone": None,
            }
        )
        assert sorted(entries) == ["gone", "page"]
        assert entries["page"] == {"id": "page", "mediaType": "text/html", "text": "<p>hi</p>"}
        assert entries["gone"] is None


class TestStore:
    def test_resolves_inline_text_and_base64(self) -> None:
        store = AssetStore()
        store.replace(
            {
                "page": {"id": "page", "mediaType": "text/html", "text": "<p>hi</p>"},
                "logo": {"id": "logo", "mediaType": "image/png", "base64": base64.b64encode(b"\x01\x02").decode()},
            }
        )
        page = store.resolve("hkp-asset://page")
        assert page.problem == ""
        assert page.asset.content == b"<p>hi</p>"
        assert store.resolve("hkp-asset://logo").asset.content == b"\x01\x02"

    def test_says_why_an_asset_does_not_resolve(self) -> None:
        store = AssetStore()
        store.replace(
            {
                "pinned": {"id": "pinned", "mediaType": "text/plain", "text": "changed", "sha256": _sha("original")},
                "local": {"id": "local", "mediaType": "text/plain", "url": "file:///etc/passwd"},
                "ftp": {"id": "ftp", "mediaType": "text/plain", "url": "ftp://example.com/a"},
            }
        )
        assert "not known" in store.resolve("hkp-asset://missing").problem
        assert "sha256" in store.resolve("hkp-asset://pinned").problem
        assert "file:// sources cannot be read" in store.resolve("hkp-asset://local").problem
        assert "not supported" in store.resolve("hkp-asset://ftp").problem

    def test_serves_an_edit_on_the_next_use_and_tells_subscribers(self) -> None:
        store = AssetStore()
        store.replace({"page": {"id": "page", "mediaType": "text/html", "text": "v1"}})
        changed: list[str] = []
        store.subscribe("page", changed.append)

        assert store.resolve("hkp-asset://page").asset.content == b"v1"
        store.merge({"page": {"id": "page", "mediaType": "text/html", "text": "v2"}})
        assert store.resolve("hkp-asset://page").asset.content == b"v2"
        store.merge({"page": None})
        assert "not known" in store.resolve("hkp-asset://page").problem
        assert changed == ["page", "page"]


@pytest.fixture
def remote():
    state = {"body": "remote v1", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            state["requests"].append(dict(self.headers))
            etag = f'"{_sha(state["body"])}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.end_headers()
                return
            data = state["body"].encode()
            self.send_response(200)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}/a.txt"
    yield state
    server.shutdown()


class TestUrlSource:
    def test_fetches_and_revalidates_with_its_etag(self, remote) -> None:
        store = AssetStore()
        store.replace({"r": {"id": "r", "mediaType": "text/plain", "url": remote["url"]}})
        assert store.resolve("hkp-asset://r").asset.content == b"remote v1"
        assert store.resolve("hkp-asset://r").asset.content == b"remote v1"
        assert remote["requests"][1].get("If-None-Match")
        remote["body"] = "remote v2"
        assert store.resolve("hkp-asset://r").asset.content == b"remote v2"

    def test_does_not_ask_again_for_content_pinned_by_its_hash(self, remote) -> None:
        store = AssetStore()
        store.replace({"r": {"id": "r", "mediaType": "text/plain", "url": remote["url"], "sha256": _sha("remote v1")}})
        store.resolve("hkp-asset://r")
        store.resolve("hkp-asset://r")
        assert len(remote["requests"]) == 1

    def test_sends_a_header_secret_only_where_its_audience_allows(self, remote) -> None:
        vault = SecretVault()
        vault.replace(read_secrets_payload({"token": {"value": "s3cret", "audience": ["127.0.0.1"]}}))
        store = AssetStore(lambda: vault)
        store.replace(
            {"r": {"id": "r", "mediaType": "text/plain", "url": remote["url"], "headers": {"Authorization": "Bearer {{secret.token}}"}}}
        )
        store.resolve("hkp-asset://r")
        assert remote["requests"][0].get("Authorization") == "Bearer s3cret"

        vault.replace(read_secrets_payload({"token": {"value": "s3cret", "audience": ["elsewhere.example"]}}))
        assert "may not be sent" in store.resolve("hkp-asset://r").problem


class TestAssetService:
    def _runtime(self, assets: dict[str, Any], services: list[ServiceConfiguration]) -> HostedRuntime:
        def create_service(config: ServiceConfiguration) -> Any:
            if config.service_id == SUB_SERVICE_DESCRIPTOR.service_id:
                return SubService(config, create_service)
            return AssetService(config)

        return HostedRuntime(
            RuntimeConfiguration(id="rt", name="Rt", assets=assets, services=services),
            create_service,
        )

    def test_emits_text_as_a_body_and_bytes_otherwise(self) -> None:
        runtime = self._runtime(
            {
                "page": {"id": "page", "mediaType": "text/html", "text": "<p>hi</p>"},
                "logo": {"id": "logo", "mediaType": "image/png", "base64": base64.b64encode(b"\x09").decode()},
            },
            [ServiceConfiguration(service_id="asset", uuid="a", state={"asset": "hkp-asset://page"})],
        )
        assert runtime.process({}, lambda *_: None) == {
            "meta": {"status": 200, "contentType": "text/html", "asset": "page", "size": 9},
            "body": "<p>hi</p>",
        }
        out = runtime.process({"asset": "hkp-asset://logo"}, lambda *_: None)
        assert out["binary"] == b"\x09"

    def test_answers_an_unknown_asset_with_an_error(self) -> None:
        runtime = self._runtime(
            {}, [ServiceConfiguration(service_id="asset", uuid="a", state={"asset": "hkp-asset://gone"})]
        )
        out = runtime.process({}, lambda *_: None)
        assert out["meta"]["status"] == 404
        assert "not known" in out["body"]["error"]

    def test_a_nested_pipeline_resolves_against_its_host_including_later_pushes(self) -> None:
        runtime = self._runtime(
            {"page": {"id": "page", "mediaType": "text/plain", "text": "v1"}},
            [
                ServiceConfiguration(
                    service_id=SUB_SERVICE_DESCRIPTOR.service_id,
                    uuid="scope",
                    state={"pipeline": [{"serviceId": "asset", "instanceId": "inner", "state": {"asset": "hkp-asset://page"}}]},
                )
            ],
        )
        assert runtime.process({}, lambda *_: None)["body"] == "v1"
        runtime.set_assets(read_assets_payload({"page": {"mediaType": "text/plain", "text": "v2"}}))
        assert runtime.process({}, lambda *_: None)["body"] == "v2"


@pytest_asyncio.fixture
async def servers():
    started = []

    async def start():
        server = create_runtime_server({"external_host": "127.0.0.1"})
        address = await server.start(0, "127.0.0.1")
        started.append(server)
        return address["base_url"]

    yield start
    for server in started:
        await server.stop()


async def _serve(session, base_url: str, assets: dict[str, Any], template: dict[str, Any]) -> str:
    async with session.post(
        f"{base_url}/runtimes",
        json={
            "id": "rt-1",
            "name": "Python",
            "assets": assets,
            "services": [
                {
                    "serviceId": HTTP_SERVER_SUBSERVICES_DESCRIPTOR.service_id,
                    "uuid": "http-1",
                    "state": {
                        "bypass": False,
                        "onRequest": [
                            {"instanceId": "answer", "serviceId": "map", "state": {"mode": "replace", "template": template}}
                        ],
                    },
                }
            ],
        },
    ) as res:
        assert res.status == 200
    async with session.get(f"{base_url}/runtimes/rt-1/services/http-1") as res:
        return (await res.json())["__hkpMount"]


@pytest.mark.asyncio
async def test_an_endpoint_serves_its_asset_body_and_an_edit_on_the_next_request(servers) -> None:
    base_url = await servers()
    async with aiohttp.ClientSession() as session:
        mount = await _serve(
            session,
            base_url,
            {"player": {"mediaType": "text/html; charset=utf-8", "text": "<h1>v1</h1>"}},
            {"meta": {"status": 200}, "body": "hkp-asset://player"},
        )
        async with session.get(mount) as res:
            assert res.headers["Content-Type"] == "text/html; charset=utf-8"
            assert await res.text() == "<h1>v1</h1>"

        async with session.post(
            f"{base_url}/runtimes/rt-1/assets",
            json={"player": {"mediaType": "text/html; charset=utf-8", "text": "<h1>v2</h1>"}},
        ) as res:
            assert await res.json() == {"ids": ["player"]}

        async with session.get(mount) as res:
            assert await res.text() == "<h1>v2</h1>"

        async with session.get(f"{base_url}/runtimes/rt-1/assets/player") as res:
            assert await res.json() == {"ok": True, "mediaType": "text/html; charset=utf-8", "size": 11}


@pytest.mark.asyncio
async def test_an_endpoint_fails_loudly_for_an_asset_it_does_not_have(servers) -> None:
    base_url = await servers()
    async with aiohttp.ClientSession() as session:
        mount = await _serve(session, base_url, {}, {"meta": {"status": 200}, "body": "hkp-asset://missing"})
        async with session.get(mount) as res:
            assert res.status == 500
            assert '"missing" is not known' in (await res.json())["error"]
