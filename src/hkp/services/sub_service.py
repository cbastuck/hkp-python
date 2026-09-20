from __future__ import annotations

# Service Documentation
# Service ID: sub-service
# Service Name: SubService
# Runtime: hkp-python
# Modes: sub-pipeline execution
# Key Config: pipeline/subservices configuration
# IO: in=any -> out=pipeline result
# Arrays: service-defined, typically forwarded
# Binary: depends on nested services
# MixedData: not native in runtime

import uuid as _uuid_mod
from typing import Any, Callable

from ..address import join_address
from ..runtime import HostedRuntime, child_run
from ..types import (
    ProcessContext,
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
    SlotStore,
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
        #: Whether what this pipeline produced leaves this service.
        #:
        #: A scope that ends here rather than feeding the services after it:
        #: the two flows on one runtime that a Stopper between them used to
        #: mark by convention. False — and absent — is the pipeline a board
        #: already has, so every board that says nothing about it goes on
        #: passing its result along.
        self._stop_propagation = False
        #: What this scope keeps to itself: "own" | "inherit". One block in the
        #: board rather than a flat key, because a scope has more than one
        #: thing to say about what its children can see.
        self._scope_slots = "own"
        #: The cells a scope of its own holds values in. Owned here rather than
        #: left to the nested runtime so that rebuilding the pipeline — which a
        #: board does on every edit — does not drop what is held across it.
        self._slot_store = SlotStore()
        self._pipeline_config: list[ServiceConfiguration] = []
        self._pipeline: HostedRuntime | None = None
        self._release_pipeline_notifications: Callable[[], None] | None = None
        self._release_pipeline_logs: Callable[[], None] | None = None
        self._create_service = create_service
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("bypass"), bool):
            self._bypass = config["bypass"]
        # Read only when it is a boolean, so a board that never mentions it
        # keeps the default rather than having one written over it by silence.
        if isinstance(config.get("stopPropagation"), bool):
            self._stop_propagation = config["stopPropagation"]
        scope = config.get("scope")
        if isinstance(scope, dict) and scope.get("slots") in ("own", "inherit"):
            self._scope_slots = scope["slots"]

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
            # Reported even when false, like the bypass beside it: a saved
            # board then says outright what each scope does with its answer.
            "stopPropagation": self._stop_propagation,
            "scope": {"slots": self._scope_slots},
            "pipeline": self._get_pipeline_state(),
        }

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        if self._bypass or not self._pipeline or not self._pipeline.list_services():
            # A scope that passes nothing on passes nothing on when there is
            # nothing to run either: what leaves this service is the board
            # author's to say, and it does not become the input again because
            # the pipeline was empty.
            return None if self._stop_propagation else input
        # The callback is a no-op: the nested runtime fans these out to the
        # target registered in _rebuild. Forwarding them here as well would
        # deliver every one twice.
        #
        # The nested pipeline runs as a run of its own, descended from the one
        # calling it, so what happens inside stays attributable to this service
        # rather than blending into the pipeline around it.
        result = self._pipeline.process(
            input,
            lambda _n: None,
            child_run(self._host.current_context() if self._host else None),
        )
        return None if self._stop_propagation else result

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host
        # A pipeline built in the constructor was built before there was a host
        # to ask, so what the board records reaches it here rather than never.
        self._apply_log_settings()
        self._apply_secrets()
        self._apply_slots()
        self._apply_mounts()

    def find_nested(self, instance_id: str):
        """The nested service a scoped address names inside this one.

        What makes a sub-pipeline addressable from outside: without it the
        board can reach this service but nothing it contains, so a facade could
        drive a scope but not read what the scope is doing.
        """
        return self._pipeline.get_service(instance_id) if self._pipeline else None

    def process_nested(
        self, address: str, input: Any, context: ProcessContext | None = None
    ) -> Any:
        """Enter this service's pipeline at one of its services.

        The nested pipeline is a chain like any other, so this is ``process_at``
        one level down — what follows the named service inside this scope runs,
        and what precedes it does not.
        """
        if not self._pipeline:
            raise KeyError(address)
        return self._pipeline.process_at(address, input, lambda _n: None, context)

    def remount(self) -> None:
        """Pass the retry down; see HostedService.remount in the runtime."""
        if not self._pipeline:
            return
        for descriptor in self._pipeline.list_services():
            svc = self._pipeline.get_service(descriptor.uuid)
            remount = getattr(svc, "remount", None)
            if remount:
                remount()

    def _apply_slots(self) -> None:
        """Points the nested pipeline at the cells its values are held in.

        Read on each lookup rather than now, so that changing what a scope
        keeps to itself takes effect without rebuilding the pipeline.
        """
        if not self._pipeline:
            return
        self._pipeline.delegate_slots(
            lambda: (self._host.slots() if self._host else None)
            if self._scope_slots == "inherit"
            else self._slot_store
        )

    def _apply_mounts(self) -> None:
        """Lets the services inside claim an endpoint on the runtime outside.

        A nested runtime has no server, so without this an ``http-server``
        inside a scope published no address. The name a mount is derived from
        falls back to the **scoped address**, so two copies of one scope do not
        derive the same address and quietly take each other's callers; a board
        that named its mount keeps that name, and with it the address it had
        before being scoped.
        """
        if not self._pipeline:
            return
        self._pipeline.delegate_mounts(
            lambda service_uuid, handler, mount_name: (
                self._host.mount(
                    service_uuid,
                    handler,
                    mount_name or join_address(self.uuid, service_uuid),
                )
                if self._host
                else None
            )
        )

    def _apply_secrets(self) -> None:
        """Points the nested pipeline at the surrounding runtime's secrets.

        Nothing provisions a nested runtime, so its vault is always empty: a
        service inside the pipeline holds the same ``{{secret.…}}`` reference as
        one at the top level and would have nothing to resolve it against. The
        host is read on each lookup rather than now, both because a value may be
        pushed after the board is running and because a pipeline nested deeper
        reaches its own host the same way — so the chain composes to whichever
        runtime was actually given something.
        """
        if self._pipeline:
            self._pipeline.delegate_secrets(
                lambda: self._host.secrets() if self._host else None
            )

    def _apply_log_settings(self) -> None:
        """Hands the board's log settings to the nested pipeline, if any."""
        if not self._host or not self._pipeline:
            return
        settings = self._host.log_settings()
        self._pipeline.set_logging(settings["logging"])
        self._pipeline.set_log_data(settings["log_data"])
        self._pipeline.set_log_level(settings["log_level"])

    def destroy(self) -> None:
        self._release_notifications()
        if self._pipeline:
            self._pipeline.destroy()
            self._pipeline = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._release_notifications()
        # The pipeline being replaced is about to become unreachable; its
        # services keep running until told otherwise. State worth carrying over
        # has already been read into _pipeline_config by _sync_states.
        if self._pipeline:
            self._pipeline.destroy()

        self._pipeline = HostedRuntime(
            _make_runtime_config(
                self.uuid, self.service_name, self._pipeline_config
            ),
            self._create_service,
        )

        # A nested runtime has no notification targets of its own, so what its
        # services report — a Timer's tick, a Hold's counts — reaches nobody
        # unless the service hosting the pipeline carries it out to the board.
        # Services report through their host precisely because it is not always
        # a call they are answering: an autonomous emitter has no caller.
        # Carried out under a scoped address rather than the bare instanceId
        # the nested service reported: an instanceId is unique only inside its
        # own pipeline, so on its own it is a name, not an address. Prefixing
        # at each boundary is what makes the path a listener hears the path it
        # can dial.
        self._release_pipeline_notifications = self._pipeline.register_notification_target(
            lambda n: self._host.notify(
                n.payload, join_address(self.uuid, n.instance_id)
            )
            if self._host
            else None
        )

        # A nested pipeline's entries belong to the same board log as everything
        # else; only the runtime hosting this service can carry them there,
        # since a nested runtime has no route out of its own.
        self._release_pipeline_logs = self._pipeline.register_log_target(
            lambda entry: self._host.forward_log(entry) if self._host else None
        )

        self._apply_log_settings()
        self._apply_secrets()
        self._apply_slots()
        self._apply_mounts()

    def _release_notifications(self) -> None:
        if self._release_pipeline_notifications:
            self._release_pipeline_notifications()
            self._release_pipeline_notifications = None
        if self._release_pipeline_logs:
            self._release_pipeline_logs()
            self._release_pipeline_logs = None

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
