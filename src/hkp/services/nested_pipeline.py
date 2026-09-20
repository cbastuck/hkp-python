from __future__ import annotations

"""A pipeline of services hosted inside a service.

SubService owns one of these; Tracks owns one per track; an http endpoint owns
one per entry point. Everything a nested runtime cannot do for itself lives
here: it has no notification targets, no route to the board's log, no vault and
no cells, so the service hosting it has to carry all of them in. Getting any of
them wrong is quiet rather than loud, which is why it is one implementation and
not one per host.
"""

from typing import Any, Callable

from ..address import join_address
from ..runtime import HostedRuntime, child_run
from ..types import (
    JsonRecord,
    RuntimeHost,
    ServiceConfiguration,
    ServiceCreator,
    SlotStore,
)
from .sub_service import (
    _make_runtime_config,
    _normalize_pipeline_array,
    _normalize_pipeline_entry,
)


class NestedPipeline:
    """One nested pipeline, with everything a nested runtime cannot do itself.

    It has no notification target, no route to the board's log, no vault and no
    cells of its own, so the service hosting it carries all of them in. Tracks
    owns one of these per track and one for the reducer; an http endpoint owns
    one per entry point.
    """

    def __init__(
        self,
        label: str,
        create_service: ServiceCreator,
        owner: str = "SubService",
        owner_uuid: str = "",
    ) -> None:
        """
        :param owner:      What the nested runtime is called in logs.
        :param owner_uuid: The uuid a service inside this pipeline is addressed
                           under. Separate from ``owner``, which is a name for
                           a person reading a log rather than one for dialling.
        """
        self._label = label
        self._owner = owner
        self._owner_uuid = owner_uuid
        self._create_service = create_service
        #: See share_slots. None means the surrounding runtime's cells.
        self._shared: SlotStore | None = None
        self._config: list[ServiceConfiguration] = []
        self._runtime: HostedRuntime | None = None
        self._release_notifications: Callable[[], None] | None = None
        self._release_logs: Callable[[], None] | None = None
        self._host: RuntimeHost | None = None

    def attach(self, host: RuntimeHost) -> None:
        self._host = host
        self._apply_log_settings()
        self._apply_secrets()
        self._apply_slots()
        self._apply_mounts()

    def share_slots(self, store: SlotStore) -> None:
        """Hold values here rather than in the surrounding runtime's cells.

        Set by a service with more than one pipeline that has to hold something
        between them.
        """
        self._shared = store
        self._apply_slots()

    def find(self, instance_id: str) -> Any | None:
        """One of the services here, by the name it carries — for an address."""
        return self._runtime.get_service(instance_id) if self._runtime else None

    def is_empty(self) -> bool:
        return not self._runtime or not self._runtime.list_services()

    def set_pipeline(self, value: Any) -> None:
        if not isinstance(value, list):
            raise ValueError(f"Invalid pipeline format for '{self._label}'")
        nxt = _normalize_pipeline_array(value)
        if nxt is None:
            raise ValueError(f"Invalid pipeline format for '{self._label}'")
        # A pipeline it already is, is not a change. Rebuilding destroys what is
        # running inside it, and the commonest configure a service gets is the
        # board handing back the state it just read.
        if self._matches_live(nxt):
            return
        self._config = nxt
        self._rebuild()

    def append(self, entry: Any) -> None:
        appended = _normalize_pipeline_entry(entry)
        if not appended:
            raise ValueError(f"Invalid appendService payload for '{self._label}'")
        self._sync_states()
        self._config.append(appended)
        self._rebuild()

    def remove(self, uuid: str) -> None:
        self._sync_states()
        self._config = [e for e in self._config if e.uuid != uuid]
        self._rebuild()

    def configure_service(self, instance_id: str, state: JsonRecord) -> None:
        if not self._runtime:
            return
        self._runtime.configure_service(instance_id, state)
        self._sync_states()

    def process(self, input: Any, parent: Any) -> Any:
        if not self._runtime or self.is_empty():
            return input
        # The callback is a no-op: the nested runtime already fans notifications
        # out to the target registered in _rebuild.
        return self._runtime.process(input, lambda _n: None, child_run(parent))

    def state(self) -> list[dict[str, Any]]:
        if not self._runtime:
            return [
                {"serviceId": e.service_id, "instanceId": e.uuid, "state": e.state or {}}
                for e in self._config
            ]
        return [
            {"serviceId": svc.service_id, "instanceId": svc.uuid, "state": svc.state}
            for svc in self._runtime.list_services()
        ]

    def services(self) -> list[Any]:
        return self._runtime.list_services() if self._runtime else []

    def destroy(self) -> None:
        self._release()
        if self._runtime:
            self._runtime.destroy()
            self._runtime = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _matches_live(self, nxt: list[ServiceConfiguration]) -> bool:
        if not self._runtime:
            return False
        live = self.state()
        if len(live) != len(nxt):
            return False
        return all(
            current["serviceId"] == entry.service_id
            and current["instanceId"] == entry.uuid
            and current["state"] == (entry.state or {})
            for current, entry in zip(live, nxt)
        )

    def _rebuild(self) -> None:
        self._release()
        if self._runtime:
            self._runtime.destroy()

        self._runtime = HostedRuntime(
            _make_runtime_config(self._label, self._owner, self._config),
            self._create_service,
        )
        # Under a scoped address, not the bare instanceId: an instanceId is
        # unique only inside its own pipeline, so each boundary prefixes its
        # owner on the way out. See address.py.
        self._release_notifications = self._runtime.register_notification_target(
            lambda n: self._host.notify(
                n.payload, join_address(self._owner_uuid, n.instance_id)
            )
            if self._host
            else None
        )
        self._release_logs = self._runtime.register_log_target(
            lambda entry: self._host.forward_log(entry) if self._host else None
        )
        self._apply_log_settings()
        self._apply_secrets()
        self._apply_slots()
        self._apply_mounts()

    def _release(self) -> None:
        if self._release_notifications:
            self._release_notifications()
            self._release_notifications = None
        if self._release_logs:
            self._release_logs()
            self._release_logs = None

    def _apply_secrets(self) -> None:
        if self._runtime:
            self._runtime.delegate_secrets(
                lambda: self._host.secrets() if self._host else None
            )

    def _apply_slots(self) -> None:
        """Points the nested runtime at the cells its values are held in.

        A store the owning service provided is what two of its pipelines share;
        with none, the surrounding runtime's is used, so a slot named inside a
        pipeline means what the same name means outside it. Read on each lookup,
        for the same reason secrets are: the pipeline is attached before the
        host is necessarily able to answer.
        """
        if not self._runtime:
            return
        self._runtime.delegate_slots(
            lambda: self._shared
            if self._shared is not None
            else (self._host.slots() if self._host else None)
        )

    def _apply_mounts(self) -> None:
        """Lets the services inside claim an endpoint on the runtime outside.

        Same reason as slots and secrets: a nested runtime has no server of its
        own. The name falls back to the scoped address, so two containers
        holding a pipeline that names no mount do not derive one address
        between them.
        """
        if not self._runtime:
            return
        self._runtime.delegate_mounts(
            lambda service_uuid, handler, mount_name: (
                self._host.mount(
                    service_uuid,
                    handler,
                    mount_name or join_address(self._owner_uuid, service_uuid),
                )
                if self._host
                else None
            )
        )

    def _apply_log_settings(self) -> None:
        if not self._host or not self._runtime:
            return
        settings = self._host.log_settings()
        self._runtime.set_logging(settings["logging"])
        self._runtime.set_log_data(settings["log_data"])
        self._runtime.set_log_level(settings["log_level"])

    def _sync_states(self) -> None:
        if not self._runtime:
            return
        by_id = {svc.uuid: svc.state for svc in self._runtime.list_services()}
        self._config = [
            ServiceConfiguration(
                service_id=e.service_id,
                uuid=e.uuid,
                name=e.name,
                service_name=e.service_name,
                state=by_id.get(e.uuid, e.state),
            )
            for e in self._config
        ]
