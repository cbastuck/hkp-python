from __future__ import annotations

import uuid as _uuid_mod
from typing import Any

from ..runtime import HostedRuntime
from ..types import (
    JsonRecord,
    NotifyCallback,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
)

SUB_SERVICE_DESCRIPTOR = ServiceRegistryEntry(
    service_id="sub-service",
    service_name="SubService",
    capabilities=["subservices"],
)


class SubService:
    service_id = SUB_SERVICE_DESCRIPTOR.service_id
    service_name = SUB_SERVICE_DESCRIPTOR.service_name
    version: str | None = None
    capabilities = SUB_SERVICE_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        self.uuid = config.uuid
        self._bypass = False
        self._pipeline_config: list[ServiceConfiguration] = []
        self._pipeline: HostedRuntime | None = None
        self._create_service = create_service

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("bypass"), bool):
            self._bypass = config["bypass"]

        if isinstance(config.get("pipeline"), list):
            next_pipeline = _normalize_pipeline_array(config["pipeline"])
            if next_pipeline is None:
                raise ValueError("Invalid sub-service pipeline format")
            self._pipeline_config = next_pipeline
            self._rebuild()
            return self.get_state()

        if _is_json_record(config.get("appendService")):
            appended = _normalize_pipeline_entry(config["appendService"])
            if not appended:
                raise ValueError("Invalid appendService payload")
            self._sync_states()
            self._pipeline_config.append(appended)
            self._rebuild()
            return self.get_state()

        if isinstance(config.get("removeService"), str):
            self._sync_states()
            target = config["removeService"]
            self._pipeline_config = [e for e in self._pipeline_config if e.uuid != target]
            self._rebuild()
            return self.get_state()

        if _is_json_record(config.get("configureService")):
            payload = config["configureService"]
            if (
                isinstance(payload.get("instanceId"), str)
                and _is_json_record(payload.get("state"))
                and self._pipeline
            ):
                self._pipeline.configure_service(payload["instanceId"], payload["state"])
                self._sync_states()

        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "bypass": self._bypass,
            "pipeline": self._get_pipeline_state(),
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if self._bypass or not self._pipeline or not self._pipeline.list_services():
            return input
        return self._pipeline.process(
            input,
            lambda notification: notify(notification.payload, notification.instance_id),
        )

    def set_host(self, host: Any) -> None:
        pass

    def destroy(self) -> None:
        if self._pipeline:
            self._pipeline.destroy()
            self._pipeline = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._pipeline = HostedRuntime(
            _make_runtime_config(self.uuid, self.service_name, self._pipeline_config),
            self._create_service,
        )

    def _sync_states(self) -> None:
        if not self._pipeline:
            return
        by_id = {svc.uuid: svc.state for svc in self._pipeline.list_services()}
        self._pipeline_config = [
            ServiceConfiguration(
                service_id=entry.service_id,
                uuid=entry.uuid,
                name=entry.name,
                service_name=entry.service_name,
                state=by_id.get(entry.uuid, entry.state),
            )
            for entry in self._pipeline_config
        ]

    def _get_pipeline_state(self) -> list[dict[str, Any]]:
        if not self._pipeline:
            return [
                {
                    "serviceId": e.service_id,
                    "instanceId": e.uuid,
                    "state": e.state or {},
                }
                for e in self._pipeline_config
            ]
        return [
            {
                "serviceId": svc.service_id,
                "instanceId": svc.uuid,
                "state": svc.state,
            }
            for svc in self._pipeline.list_services()
        ]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_runtime_config(
    owner_uuid: str,
    service_name: str,
    pipeline_config: list[ServiceConfiguration],
) -> Any:
    from ..types import RuntimeConfiguration
    return RuntimeConfiguration(
        id=f"{owner_uuid}:sub-runtime",
        name=f"{service_name}-{owner_uuid}",
        board_name="",
        services=pipeline_config,
    )


def _is_json_record(value: Any) -> bool:
    return isinstance(value, dict)


def _normalize_pipeline_array(value: list[Any]) -> list[ServiceConfiguration] | None:
    result: list[ServiceConfiguration] = []
    for entry in value:
        normalized = _normalize_pipeline_entry(entry)
        if normalized is None:
            return None
        result.append(normalized)
    return result


def _normalize_pipeline_entry(value: Any) -> ServiceConfiguration | None:
    if not _is_json_record(value) or not isinstance(value.get("serviceId"), str):
        return None

    instance_id: str
    if isinstance(value.get("instanceId"), str) and value["instanceId"]:
        instance_id = value["instanceId"]
    elif isinstance(value.get("uuid"), str) and value["uuid"]:
        instance_id = value["uuid"]
    else:
        instance_id = str(_uuid_mod.uuid4())

    state = value.get("state")
    if state is not None and not _is_json_record(state):
        return None

    return ServiceConfiguration(
        service_id=value["serviceId"],
        uuid=instance_id,
        name=value.get("name") if isinstance(value.get("name"), str) else None,
        service_name=value.get("serviceName") if isinstance(value.get("serviceName"), str) else None,
        state=state,
    )
