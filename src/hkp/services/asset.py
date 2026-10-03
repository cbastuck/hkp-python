from __future__ import annotations

# Service Documentation
# Service ID: asset
# Service Name: Asset
# Runtime: hkp-python
# Modes: none
# Key Config: asset (an `hkp-asset://<id>` reference)
# IO: in=anything, or a reference naming another asset -> out={meta, body}
#     for text, {meta, binary} otherwise
# Arrays: not primary
# Binary: an asset that is not text leaves as bytes
#
# Puts one of the board's assets into the pipeline, for any service that has no
# way of its own to take one. A service that serves, plays or loads content
# resolves a reference itself; everything else composes with this.
#
# The asset is resolved on every pass, from the runtime's asset store, so editing
# it changes what the next pass carries without anything being reconfigured.
#
# The input can name the asset: a bare `hkp-asset://…` string, or an object whose
# `asset` field is one, takes the place of the configured reference for that
# pass. Any other input only triggers the pass.
#
# The answer is shaped like an HTTP response (`meta` with `status` and
# `contentType`, beside `body` or `binary`), so an endpoint can hand it straight
# back. An asset that does not resolve is answered with an error status and
# reported, never passed on as its reference.
#
# Mirrors hkp-node's `asset`.

from typing import Any

from ..assets import is_text_media_type, parse_asset_ref
from ..types import JsonRecord, NotifyCallback, RuntimeHost, ServiceConfiguration, ServiceRegistryEntry

ASSET_DESCRIPTOR = ServiceRegistryEntry(
    service_id="asset",
    service_name="Asset",
    version="v1",
    capabilities=[],
)


def _requested_reference(value: Any) -> str | None:
    """The reference an input names, when it names one."""
    if isinstance(value, str) and parse_asset_ref(value.strip()):
        return value.strip()
    if isinstance(value, dict):
        named = value.get("asset")
        if isinstance(named, str) and parse_asset_ref(named.strip()):
            return named.strip()
    return None


class AssetService:
    service_id = ASSET_DESCRIPTOR.service_id
    service_name = ASSET_DESCRIPTOR.service_name
    version = ASSET_DESCRIPTOR.version
    capabilities = ASSET_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._host: RuntimeHost | None = None
        #: The reference emitted when the input names none.
        self._asset = ""
        self._media_type = ""
        self._size = 0
        self._error = ""
        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def get_state(self) -> JsonRecord:
        return {
            "asset": self._asset,
            "mediaType": self._media_type,
            "size": self._size,
            "error": self._error,
        }

    def configure(self, config: JsonRecord) -> JsonRecord:
        asset = config.get("asset")
        if isinstance(asset, str):
            self._asset = asset.strip()
        return self.get_state()

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        reference = _requested_reference(input) or self._asset
        if not reference:
            return self._fail(notify, 400, "no asset is configured")
        if not parse_asset_ref(reference):
            return self._fail(notify, 400, f"{reference!r} is not an asset reference")
        store = self._host.assets() if self._host and hasattr(self._host, "assets") else None
        if store is None:
            return self._fail(notify, 500, "this runtime has no assets")

        resolution = store.resolve(reference)
        asset = resolution.asset
        if asset is None:
            return self._fail(notify, 404, resolution.problem)

        self._error = ""
        self._media_type = asset.media_type
        self._size = len(asset.content)
        notify(self.get_state())

        meta = {
            "status": 200,
            "contentType": asset.media_type,
            "asset": asset.id,
            "size": len(asset.content),
        }
        if is_text_media_type(asset.media_type):
            return {"meta": meta, "body": asset.content.decode("utf-8", errors="replace")}
        return {"meta": meta, "binary": asset.content}

    def _fail(self, notify: NotifyCallback, status: int, message: str) -> JsonRecord:
        self._error = message
        notify(self.get_state())
        return {"meta": {"status": status, "contentType": "application/json"}, "body": {"error": message}}
