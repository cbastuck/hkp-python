from __future__ import annotations

import sys
import types
import unittest

try:
    import aiohttp  # noqa: F401
except ModuleNotFoundError:
    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.TCPConnector = object
    aiohttp_stub.ClientSession = object
    aiohttp_stub.ClientTimeout = object
    aiohttp_stub.web = types.SimpleNamespace(
        Request=object,
        Response=object,
        StreamResponse=object,
        WebSocketResponse=object,
    )
    sys.modules["aiohttp"] = aiohttp_stub

from hkp.resources import (
    ResourceStore,
    format_resource_ref,
    parse_resource_ref,
    read_resource_binding,
    read_resources_payload,
    referenced_resources,
)
from hkp.runtime import HostedRuntime
from hkp.services.monitor import MonitorService
from hkp.services.sub_service import SubService
from hkp.types import RuntimeConfiguration, ServiceConfiguration


OLLAMA_BINDING = {
    "id": "llm",
    "kind": "ollama",
    "url": "http://127.0.0.1:11434",
}

PG_BINDING = {
    "id": "db",
    "kind": "postgres",
    "url": "postgres://127.0.0.1:5432/app",
    "user": "app",
    "password": "secret",
    "database": "app",
}


def _create_service(cfg: ServiceConfiguration):
    if cfg.service_id == "sub-service":
        return SubService(cfg, _create_service)
    return MonitorService(cfg)


class TestResources(unittest.TestCase):
    def test_resource_references_and_payloads(self) -> None:
        self.assertEqual(parse_resource_ref("hkp-resource://llm"), "llm")
        self.assertEqual(parse_resource_ref(format_resource_ref("db.main")), "db.main")
        self.assertIsNone(parse_resource_ref("http://hkp-resource://llm"))
        self.assertEqual(
            referenced_resources(
                {
                    "a": "hkp-resource://llm",
                    "nested": [{"b": "x ? 'hkp-resource://db' : 'hkp-resource://llm'"}],
                }
            ),
            ["llm", "db"],
        )

        self.assertEqual(read_resource_binding(OLLAMA_BINDING), OLLAMA_BINDING)
        self.assertIsNone(read_resource_binding({"id": "bad", "kind": "unknown"}))
        self.assertEqual(
            read_resources_payload(
                {
                    "llm": OLLAMA_BINDING,
                    "db": PG_BINDING,
                    "gone": None,
                    "invalid": {"kind": "ollama"},
                }
            ),
            {
                "llm": OLLAMA_BINDING,
                "db": PG_BINDING,
                "gone": None,
            },
        )

    def test_resource_store_resolution_and_subscriptions(self) -> None:
        store = ResourceStore()
        changed: list[str] = []
        store.subscribe("llm", changed.append)

        store.replace({"llm": OLLAMA_BINDING, "db": PG_BINDING})
        self.assertEqual(changed, ["llm"])

        res = store.resolve("hkp-resource://llm", "ollama")
        self.assertEqual(res.resource, OLLAMA_BINDING)
        self.assertEqual(res.problem, "")

        wrong_kind = store.resolve("hkp-resource://llm", "postgres")
        self.assertIsNone(wrong_kind.resource)
        self.assertEqual(
            wrong_kind.problem,
            'resource "llm" is of kind "ollama", not "postgres"',
        )

        missing = store.resolve("hkp-resource://missing")
        self.assertIsNone(missing.resource)
        self.assertEqual(missing.problem, 'resource "missing" is not bound')

        store.merge(
            {
                "llm": {
                    **OLLAMA_BINDING,
                    "unreachable": 'resource "llm" is bound to 127.0.0.1 on another machine',
                }
            }
        )
        self.assertEqual(changed, ["llm", "llm"])
        unreachable = store.resolve("hkp-resource://llm", "ollama")
        self.assertIsNone(unreachable.resource)
        self.assertIn("another machine", unreachable.problem)

        store.merge({"llm": None})
        self.assertEqual(store.ids(), ["db"])

    def test_hosted_runtime_resources_and_nested_delegation(self) -> None:
        runtime = HostedRuntime(
            RuntimeConfiguration(
                id="rt-1",
                name="Python",
                resources={"llm": OLLAMA_BINDING},
                services=[
                    ServiceConfiguration(
                        uuid="sub",
                        service_id="sub-service",
                        state={
                            "pipeline": [
                                {
                                    "serviceId": "monitor",
                                    "instanceId": "inner",
                                    "state": {},
                                }
                            ]
                        },
                    )
                ],
            ),
            _create_service,
        )

        self.assertEqual(runtime.resources().ids(), ["llm"])
        self.assertEqual(
            runtime.resources().resolve("hkp-resource://llm", "ollama").resource,
            OLLAMA_BINDING,
        )

        sub = runtime.get_service("sub")
        self.assertIsNotNone(sub)
        self.assertEqual(
            sub._pipeline.resources().resolve("hkp-resource://llm", "ollama").resource,
            OLLAMA_BINDING,
        )

        runtime.set_resources(
            read_resources_payload({"db": PG_BINDING, "llm": None})
        )
        self.assertEqual(runtime.resources().ids(), ["db"])
        self.assertEqual(
            sub._pipeline.resources().resolve("hkp-resource://db", "postgres").resource,
            PG_BINDING,
        )
        self.assertIsNone(runtime.resources().resolve("hkp-resource://llm").resource)


if __name__ == "__main__":
    unittest.main()
