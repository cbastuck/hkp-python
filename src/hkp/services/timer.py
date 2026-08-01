from __future__ import annotations

# Service Documentation
# Service ID: hookup.to/service/timer
# Service Name: Timer
# Runtime: hkp-python
# Modes: periodic | oneShot
# Key Config: service-specific timer fields (period, units, triggers)
# IO: in=any -> out=tick payload or delayed payload
# Arrays: treated as generic input
# Binary: pass-through when payload-based
# MixedData: not native in runtime

import asyncio
from typing import Any

from ..types import (
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    ServiceConfiguration,
    ServiceRegistryEntry,
)

TIMER_DESCRIPTOR = ServiceRegistryEntry(
    service_id="timer",
    service_name="Timer",
)

#: The id this service used to answer to, from before it matched hkp-node and
#: hkp-rt. Kept as a creation-time alias so boards saved against it still load;
#: it is not advertised in the registry.
TIMER_LEGACY_SERVICE_ID = "hookup.to/service/timer"

_UNIT_MS: dict[str, float] = {
    "ms": 1,
    "s":  1_000,
    "m":  60_000,
    "h":  3_600_000,
    "d":  86_400_000,
}


def _duration_ms(value: float, unit: str) -> float:
    return value * _UNIT_MS.get(unit, 1_000)


class TimerService:
    service_id = TIMER_DESCRIPTOR.service_id
    service_name = TIMER_DESCRIPTOR.service_name
    version: str | None = None
    capabilities: list[str] | None = None

    def __init__(
        self,
        config: ServiceConfiguration,
        min_interval_ms: float = 0,
    ) -> None:
        # Lower bound on the periodic interval. On a shared host a very short
        # period is a way for one tenant to spend everyone's CPU, so the server
        # supplies a floor and ticks are clamped to it.
        self._min_interval_ms = min_interval_ms

        self.uuid = config.uuid
        self._periodic = False
        self._periodic_value = 1
        self._periodic_unit = "s"
        self._one_shot_delay = 0
        self._one_shot_delay_unit = "ms"
        self._running = False
        self._counter = 0
        self._condition_until_trigger_count: int | None = None
        self._task: asyncio.Task | None = None
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def get_state(self) -> JsonRecord:
        return {
            "periodic": self._periodic,
            "periodicValue": self._periodic_value,
            "periodicUnit": self._periodic_unit,
            "oneShotDelay": self._one_shot_delay,
            "oneShotDelayUnit": self._one_shot_delay_unit,
            "running": self._running,
            "counter": self._counter,
        }

    def configure(self, config: JsonRecord) -> JsonRecord:
        periodic_value = config.get("periodicValue")
        periodic_unit = config.get("periodicUnit")
        periodic = config.get("periodic")
        one_shot_delay = config.get("oneShotDelay")
        one_shot_delay_unit = config.get("oneShotDelayUnit")
        immediate = config.get("immediate")
        counter = config.get("counter")
        until = config.get("until")
        running = config.get("running")
        stop = config.get("stop")
        start = config.get("start")
        restart = config.get("restart")

        do_stop: bool = bool(stop or restart or (self._running and running is not None and not running))
        do_start: bool = bool(start or restart)

        def silent_restart() -> None:
            nonlocal do_start
            self._cancel_task()
            do_start = True

        if periodic_value is not None:
            self._periodic_value = periodic_value
            self._notify({"periodicValue": periodic_value})
            if not do_start and running:
                do_start = True
            elif self._running:
                silent_restart()

        if periodic_unit is not None:
            self._periodic_unit = periodic_unit
            self._notify({"periodicUnit": periodic_unit})
            if not do_start and running:
                do_start = True
            elif self._running:
                silent_restart()

        if periodic is not None:
            self._periodic = periodic
            self._notify({"periodic": periodic})
            if not do_start and running:
                do_start = True

        if one_shot_delay is not None:
            self._one_shot_delay = one_shot_delay
            self._notify({"oneShotDelay": one_shot_delay, "periodic": False})

        if counter is not None:
            self._counter = counter

        if until is not None and isinstance(until, dict):
            trigger_count = until.get("triggerCount")
            if trigger_count is not None:
                self._condition_until_trigger_count = int(trigger_count)

        if one_shot_delay_unit is not None:
            self._one_shot_delay_unit = one_shot_delay_unit
            self._notify({"oneShotDelayUnit": one_shot_delay_unit})

        if do_stop:
            self._clear_timer()

        if do_start:
            self._schedule(immediate=bool(immediate))

        self._running = self._task is not None and not self._task.done()
        self._notify({"running": self._running})
        return self.get_state()

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        # Periodic timers drive themselves — passthrough.
        # One-shot: schedule a delayed fire and return input immediately.
        if not self._periodic:
            delay_s = _duration_ms(self._one_shot_delay, self._one_shot_delay_unit) / 1000
            self._run_soon(self._tick_with_input(input, delay_s))
        return input

    def destroy(self) -> None:
        self._clear_timer()

    # ── Private ──────────────────────────────────────────────────────────────

    def _schedule(self, immediate: bool = False) -> None:
        if self._periodic:
            self._cancel_task()
            interval_s = (
                max(
                    _duration_ms(self._periodic_value, self._periodic_unit),
                    self._min_interval_ms,
                )
                / 1000
            )
            first_delay_s = 0.001 if immediate else interval_s
            self._run_soon(self._periodic_loop(first_delay_s, interval_s))
        else:
            if self._task:
                self._cancel_task()
            delay_s = 0.001 if immediate else _duration_ms(self._one_shot_delay, self._one_shot_delay_unit) / 1000
            self._run_soon(self._oneshot_fire(delay_s))

    def _run_soon(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(coro)
        except RuntimeError:
            pass  # no event loop (e.g. in tests without asyncio)

    async def _periodic_loop(self, first_delay_s: float, interval_s: float) -> None:
        await asyncio.sleep(first_delay_s)
        while True:
            self._tick()
            if self._task is None or self._task.cancelled():
                break
            await asyncio.sleep(interval_s)

    async def _oneshot_fire(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        self._tick()
        self._task = None
        self._running = False
        self._notify({"running": False})

    async def _tick_with_input(self, input: Any, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        trigger_count = self._counter + 1
        self._counter = trigger_count
        self._notify({"counter": trigger_count})
        if self._host:
            merged = {**(input if isinstance(input, dict) else {}), "triggerCount": trigger_count}
            result = self._host.process_from(
                self.uuid,
                merged,
                # No-op: the runtime already fans these out to its notification
                # targets. Re-notifying through the host would deliver each twice.
                lambda _n: None,
            )
            self._host.emit_result(result)

    def _tick(self) -> None:
        if (
            self._condition_until_trigger_count is not None
            and self._counter >= self._condition_until_trigger_count
        ):
            self._clear_timer()
            return
        trigger_count = self._counter + 1
        self._counter = trigger_count
        self._notify({"counter": trigger_count})
        if self._host:
            result = self._host.process_from(
                self.uuid,
                {"triggerCount": trigger_count},
                # No-op: the runtime already fans these out to its notification
                # targets. Re-notifying through the host would deliver each twice.
                lambda _n: None,
            )
            self._host.emit_result(result)

    def _clear_timer(self) -> None:
        self._cancel_task()
        self._counter = 0
        self._running = False
        self._notify({"running": False, "count": self._counter})

    def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def _notify(self, payload: Any, instance_id: str | None = None) -> None:
        if self._host:
            self._host.notify(payload, instance_id or self.uuid)
