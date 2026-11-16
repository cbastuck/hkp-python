from __future__ import annotations

# Service Documentation
# Service ID: loop
# Service Name: Loop
# Runtime: hkp-python
# Modes: run (rounds until done), step (one round per call)
# Key Config: until, maxRounds, mode, pipeline, bypass; commands stop, reset
# IO: in=the value to start from, or nothing to carry on -> out=the last round's output
# Arrays: passed to the nested pipeline as one value
# Binary: passed to the nested pipeline untouched
# MixedData: passed to the nested pipeline untouched

"""Repetition as a service: a pipeline run again on its own output.

Iterator runs one pipeline over many items. Loop runs one pipeline over one
value, many times: what a round returns is what the next round is given, until
a condition on that value holds or a limit of rounds is reached. Generating
text is the case it was built for — each round appends one token — but nothing
here knows about tokens.

**A value starts a loop, nothing carries one on.** A call with a value begins
at round one with it. A call with nothing continues from the value the loop
holds, which is how a Step button works: it calls the service and passes
nothing.

**Every round is reported** as ``{ round, done, reason, output }``, so what a
loop is building can be watched while it runs.

**Stop is a configure**, ``{ "stop": true }``. It takes effect between rounds:
the round in progress finishes, and the loop keeps its value, so it can be
carried on afterwards.

**A round that fails ends the call**, and the loop keeps the value it had
before that round.
"""

import threading
from typing import Any

from ..runtime import child_run
from ..types import (
    JsonRecord,
    NotifyCallback,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
)
from .expression import compile_expression, is_truthy
from .sub_service import SubService

LOOP_DESCRIPTOR = ServiceRegistryEntry(
    service_id="loop",
    service_name="Loop",
    version="v1",
    capabilities=["subservices"],
)

DEFAULT_MAX_ROUNDS = 100

#: Why a loop is not running, as its state and its reports say it.
REASON_UNTIL = "until"
REASON_MAX_ROUNDS = "maxRounds"
REASON_ENDED = "ended"
REASON_STOPPED = "stopped"


class LoopService(SubService):
    service_id = LOOP_DESCRIPTOR.service_id
    service_name = LOOP_DESCRIPTOR.service_name
    version = LOOP_DESCRIPTOR.version
    capabilities = LOOP_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        # Before the base constructor, which configures from the board's state
        # and so reaches the override below.
        self._until_source = ""
        self._until = None
        self._max_rounds = DEFAULT_MAX_ROUNDS
        self._mode = "run"
        #: What the next round is given. None: nothing to carry on from.
        self._held: Any = None
        self._round = 0
        self._done = False
        self._reason = ""
        self._running = False
        #: Set from configure, on the server's loop, while a call runs on the
        #: worker thread: the one thing the two share.
        self._stop = threading.Event()
        super().__init__(config, create_service)

    def configure(self, config: JsonRecord) -> JsonRecord:
        until = config.get("until")
        if isinstance(until, str):
            compiled = compile_expression(until) if until.strip() else None
            problem = getattr(compiled, "parse_error", None)
            if problem:
                raise ValueError(f"loop: {problem}")
            self._until_source = until
            self._until = compiled
        limit = config.get("maxRounds")
        if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit >= 0:
            self._max_rounds = int(limit)
        if config.get("mode") in ("run", "step"):
            self._mode = config["mode"]
        if config.get("stop") is True:
            self._stop.set()
        if config.get("reset") is True:
            self._start(None)
        # The pipeline, bypass and the editing commands belong to the base.
        super().configure(config)
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            **super().get_state(),
            "until": self._until_source,
            "maxRounds": self._max_rounds,
            "mode": self._mode,
            "round": self._round,
            "running": self._running,
            "done": self._done,
            "reason": self._reason,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        if self._bypass or not self._pipeline or not self._pipeline.list_services():
            # Nothing to repeat: what an empty SubService does.
            return None if self._stop_propagation else input

        if input is not None:
            self._start(input)
        if self._held is None or self._done:
            # Nothing to carry on from, or nothing left to do.
            return None

        parent = self._host.current_context() if self._host else None
        self._stop.clear()
        self._reason = ""
        self._running = True
        ran = False
        try:
            while True:
                if self._stop.is_set():
                    self._reason = REASON_STOPPED
                    break
                if self._max_rounds and self._round >= self._max_rounds:
                    # Only reached when the limit was lowered under a loop
                    # already past it.
                    self._finish(REASON_MAX_ROUNDS)
                    break

                # A run of its own per round, descended from the run that
                # reached the loop, so a log answers "what did round 7 do".
                output = self._pipeline.process(
                    self._held, lambda _n: None, child_run(parent)
                )
                if output is None:
                    # The pipeline stopped: there is nothing to give the next
                    # round, so this is where the loop ends.
                    self._finish(REASON_ENDED)
                    break

                self._held = output
                self._round += 1
                ran = True
                if self._until is not None and is_truthy(self._until(output)):
                    self._finish(REASON_UNTIL)
                elif self._max_rounds and self._round >= self._max_rounds:
                    self._finish(REASON_MAX_ROUNDS)
                notify(self._report(output), None)
                if self._done or self._mode == "step":
                    break
        finally:
            self._running = False

        if not ran:
            notify(self._report(None), None)
        if self._stop_propagation or not ran:
            return None
        return self._held

    # ── Private ──────────────────────────────────────────────────────────────

    def _start(self, value: Any) -> None:
        self._held = value
        self._round = 0
        self._done = False
        self._reason = ""

    def _finish(self, reason: str) -> None:
        self._done = True
        self._reason = reason

    def _report(self, output: Any) -> JsonRecord:
        report: JsonRecord = {
            "round": self._round,
            "done": self._done,
            "reason": self._reason,
        }
        if output is not None:
            report["output"] = output
        return report
