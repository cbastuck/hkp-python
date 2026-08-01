from __future__ import annotations

# Service Documentation
# Service ID: stopper
# Service Name: Stopper
# Runtime: hkp-python
# Modes: none
# Key Config: bypass
# IO: in=anything -> out=None (nothing is forwarded)
#
# Returns None on every call, which the runtime reads as "nothing to pass on":
# the services after it are not called, and the runtime emits no result, so the
# next runtime in the chain is not driven either.
#
# That last part is the reason to reach for it on a board with several runtimes.
# Runtimes are chained — the result of one becomes the input of the next — so a
# runtime whose work is a side effect rather than a value should end here,
# instead of feeding whatever it happened to produce into the next runtime.
# Mirrors hkp-node's and hkp-rt's `stopper`, and the browser runtime's
# `hookup.to/service/stopper`.

from typing import Any

from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

STOPPER_DESCRIPTOR = ServiceRegistryEntry(
    service_id="stopper",
    service_name="Stopper",
    version="v1",
    capabilities=[],
)


class StopperService:
    service_id = STOPPER_DESCRIPTOR.service_id
    service_name = STOPPER_DESCRIPTOR.service_name
    version = STOPPER_DESCRIPTOR.version
    capabilities = STOPPER_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._bypass = False

        if config.state:
            self.configure(config.state)

    def get_state(self) -> JsonRecord:
        return {"bypass": self._bypass}

    def configure(self, config: JsonRecord) -> JsonRecord:
        # Bypass is the only setting worth having: it turns the dead end back
        # into a passthrough, so a chain can be opened up without moving
        # services around.
        if isinstance(config.get("bypass"), bool):
            self._bypass = config["bypass"]
        return self.get_state()

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        return input if self._bypass else None
