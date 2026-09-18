from __future__ import annotations

# Service Documentation
# Service ID: join
# Service Name: Join
# Runtime: hkp-python
# Modes: overwrite | add
# Key Config: as, mode, pipeline, bypass
# IO: in=any -> out=the input and the nested pipeline's result, together
# Arrays: the input is one value; iterate outside this service
# Binary: passed to the nested pipeline untouched
# MixedData: not native in runtime
#
# Splitting off a piece of work and getting the answer back **beside** what you
# already had, rather than instead of it.
#
# Most services replace their input: text-to-speech answers with audio, not with
# the request and the audio. That is right for the service and wrong for the
# pipeline around it, because whatever the input was carrying — which episode
# this is, where it is to be kept — is gone by the time the answer arrives, and
# the service that has to file the answer no longer knows where it belongs.
#
#     input ──┬──────────────────────────────► merged output
#             └── nested pipeline ── result ──┘
#
# `as` is the safe way to use it: naming a key puts the result there, where
# nothing the nested pipeline produces can collide with what the input carried.
# Merging at the top level is for the cases where the two shapes are known to be
# disjoint.
#
# A nested pipeline that **stops** — returns None — stops this one too. The
# merge is the reason the Join is there, so continuing without it would hand the
# services downstream something that looks like a successful merge and is not.
#
# Mirrors hkp-node's `join`, down to the state it keeps, so one board reads the
# same whichever runtime hosts it.

from typing import Any

from ..types import (
    JsonRecord,
    NotifyCallback,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
)
from .sub_service import SubService

JOIN_DESCRIPTOR = ServiceRegistryEntry(
    service_id="join",
    service_name="Join",
    version="v1",
    # Holds a pipeline, so the board's UI must let anyone look inside it.
    capabilities=["subservices"],
)

MODES = ("overwrite", "add")


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


class JoinService(SubService):
    service_id = JOIN_DESCRIPTOR.service_id
    service_name = JOIN_DESCRIPTOR.service_name
    version = JOIN_DESCRIPTOR.version
    capabilities = JOIN_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        self._as = ""
        self._mode = "overwrite"
        # Set before the base constructor, which configures — unlike the two
        # TypeScript runtimes, a Python attribute assigned here is not undone by
        # a later declaration, so this is the whole of it.
        super().__init__(config, create_service)

    def configure(self, config: JsonRecord) -> JsonRecord:
        self._settle(config)
        # The pipeline, bypass and the editing commands belong to the base.
        super().configure(config)
        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {**super().get_state(), "as": self._as, "mode": self._mode}

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        result = super().process(input, notify)

        # An empty or bypassed Join is the base's pass-through, and there is
        # nothing to merge with itself.
        if result is input:
            return input

        if result is None:
            return None

        if self._as:
            if _is_record(input):
                return {**input, self._as: result}
            return {"input": input, self._as: result}

        if not _is_record(input) or not _is_record(result):
            # Nothing to merge into, or nothing to merge: a scalar or a list on
            # either side has no fields to combine, and dropping one silently
            # would be worse than saying which one a board gets.
            return result if _is_record(result) else input

        return {**result, **input} if self._mode == "add" else {**input, **result}

    # ── Private ────────────────────────────────────────────────────────────────

    def _settle(self, config: JsonRecord) -> None:
        if isinstance(config.get("as"), str):
            self._as = config["as"]
        if config.get("mode") in MODES:
            self._mode = config["mode"]
