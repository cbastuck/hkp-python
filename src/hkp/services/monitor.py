from __future__ import annotations

# Service Documentation
# Service ID: monitor
# Service Name: Monitor
# Runtime: hkp-python
# Modes: observe
# Key Config: runtime-specific observe/log settings
# IO: in=any -> out=identity
# Arrays: pass-through
# Binary: inspect/log support depends on runtime UI/logging
# MixedData: not native in runtime

import json
from typing import Any

from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

MONITOR_DESCRIPTOR = ServiceRegistryEntry(
    service_id="monitor",
    service_name="Monitor",
)


class MonitorService:
    service_id = MONITOR_DESCRIPTOR.service_id
    service_name = MONITOR_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._log_to_console = False
        #: Also record what passes through into the board's log.
        #:
        #: A probe is usually dropped in while something is being worked out and
        #: taken out again afterwards, so it is already sitting where a board
        #: author found the flow worth watching. Turning this on keeps that
        #: judgement after the author stops watching, without a second service.
        self._log_to_board = False
        self._host: Any = None
        self._file_log_path = ""
        self._render_text_editor = True
        # message is intentionally not persisted via get_state()
        self._message = ""

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("logToConsole"), bool):
            self._log_to_console = config["logToConsole"]
        if isinstance(config.get("logToBoard"), bool):
            self._log_to_board = config["logToBoard"]
        if isinstance(config.get("fileLogPath"), str):
            self._file_log_path = config["fileLogPath"]
        if isinstance(config.get("renderTextEditor"), bool):
            self._render_text_editor = config["renderTextEditor"]
        if isinstance(config.get("message"), str):
            self._message = config["message"]
        return self.get_state()

    def get_state(self) -> JsonRecord:
        # message is excluded so it is not persisted to board saves
        return {
            "logToConsole": self._log_to_console,
            "logToBoard": self._log_to_board,
            "fileLogPath": self._file_log_path,
            "renderTextEditor": self._render_text_editor,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        self._message = _format_message(input)
        if self._log_to_board and self._host:
            # The payload rides in ``data``, which the runtime drops unless the
            # board asked for it — so a probe left switched on cannot quietly
            # widen what the log holds.
            self._host.log("info", "monitor", input)
        if self._log_to_console:
            print(f"[MONITOR] {input}")
        notify(input)
        return input

    def set_host(self, host: Any) -> None:
        self._host = host

    def destroy(self) -> None:
        pass


def _format_message(input: Any) -> str:
    if input is None:
        return "null"
    if isinstance(input, (dict, list)):
        try:
            return json.dumps(input, indent=2)
        except Exception:
            return str(input)
    return str(input)
