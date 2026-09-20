from __future__ import annotations

# Service Documentation
# Service ID: tracks
# Service Name: Tracks
# Runtime: hkp-python
# Modes: serial | parallel
# Key Config: tracks, reduce, run, bypass
# IO: in=any -> out=what the reducer made of the tracks' answers
# Arrays: the answers are an array, one element per track, in declaration order
# Binary: handed to every track untouched
# MixedData: not native in runtime

"""Several pipelines over one input, and one answer out.

``iterator`` runs one pipeline over many items. This is the other half of that
pair: many pipelines over one item. A board that has to do two unrelated things
with the same value — write it to a table *and* tell somebody about it, ask
three services and compare — has otherwise to fake it by putting those things in
a row and teaching each of them to pass its input through. That works and says
nothing: nothing in the board records that they are siblings rather than a
sequence, in which order they may run, or which of their answers matters.

    input ──┬── track ── answer ──┬── reduce ── output
            ├── track ── answer ──┤
            └── track ── answer ──┘

**Every track is given the same input.** They do not feed each other, which is
the whole point: a track can be read on its own.

**The answers are a list with one element per track, nulls included.** A track
that stopped — declined, found nothing, failed — leaves a hole rather than being
left out, so position still names the track that produced it.

**``reduce`` is a pipeline, not a list of strategies.** It is given
``{"input": …, "results": [...]}``: the value the tracks were run on, and what
they answered. Carrying on as though the tracks were side effects is then
``{"=": "params.input"}``. With no reducer the answers travel on as they are,
and a reducer returning ``None`` stops the pipeline as any service does.

**``run`` is recorded and honoured where a runtime can.** This one cannot: the
python runtime processes synchronously, so tracks here always run one at a time.
A board says the same thing in either runtime and gets the same answers; only
the wall clock differs.
"""

from typing import Any, Callable

from ..types import (
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    ServiceConfiguration,
    ServiceCreator,
    ServiceRegistryEntry,
)
from .nested_pipeline import NestedPipeline
from .sub_service import _is_json_record

TRACKS_DESCRIPTOR = ServiceRegistryEntry(
    service_id="tracks",
    service_name="Tracks",
    version="v1",
    capabilities=["subservices"],
)

RUN_MODES = ("serial", "parallel")

#: What the reducer answers to when an edit names a branch. A track may not take it.
REDUCE = "reduce"


class _Track:
    def __init__(self, name: str, bypass: bool, pipeline: NestedPipeline) -> None:
        self.name = name
        self.bypass = bypass
        self.pipeline = pipeline


class TracksService:
    service_id = TRACKS_DESCRIPTOR.service_id
    service_name = TRACKS_DESCRIPTOR.service_name
    version = TRACKS_DESCRIPTOR.version
    capabilities = TRACKS_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, create_service: ServiceCreator) -> None:
        self.uuid = config.uuid
        self._create_service = create_service
        self._host: RuntimeHost | None = None
        self._bypass = False
        self._run = "serial"
        self._tracks: list[_Track] = []
        self._reduce = NestedPipeline(f"{self.uuid}:{REDUCE}", create_service, "Tracks")
        self._last_error = ""

        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host
        self._reduce.attach(host)
        for track in self._tracks:
            track.pipeline.attach(host)

    def get_state(self) -> JsonRecord:
        return {
            "bypass": self._bypass,
            "run": self._run,
            "tracks": [
                {"name": t.name, "bypass": t.bypass, "pipeline": t.pipeline.state()}
                for t in self._tracks
            ],
            "reduce": self._reduce.state(),
            "error": self._last_error,
        }

    def configure(self, config: JsonRecord) -> JsonRecord:
        # An edit naming a track is about one pipeline under this service rather
        # than about the service; without it there is no way to say which.
        if isinstance(config.get("track"), str):
            self._configure_track(config["track"], config)
            return self.get_state()

        if isinstance(config.get("bypass"), bool):
            self._bypass = config["bypass"]
        if isinstance(config.get("run"), str) and config["run"] in RUN_MODES:
            self._run = config["run"]
        if isinstance(config.get("tracks"), list):
            self._set_tracks(config["tracks"])
        if isinstance(config.get("reduce"), list):
            self._reduce.set_pipeline(config["reduce"])

        return self.get_state()

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        if self._bypass or not self._tracks:
            return input

        parent = self._host.current_context() if self._host else None
        results = [self._run_track(track, input, parent) for track in self._tracks]

        if self._reduce.is_empty():
            return results
        return self._reduce.process({"input": input, "results": results}, parent)

    def destroy(self) -> None:
        self._reduce.destroy()
        for track in self._tracks:
            track.pipeline.destroy()
        self._tracks = []
        self._host = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _run_track(self, track: _Track, input: Any, parent: Any) -> Any:
        """One track's answer, or None where it has none.

        A track that raises is reported and leaves a hole rather than taking the
        others down with it: they were given the same input and have nothing to
        do with each other, which is exactly why they are tracks.
        """
        if track.bypass or track.pipeline.is_empty():
            return None
        try:
            return track.pipeline.process(input, parent)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            self._last_error = f"track '{track.name}' failed: {error}"
            if self._host:
                self._host.log("error", "service.failed", {"message": self._last_error})
            return None

    def _set_tracks(self, value: list[Any]) -> None:
        previous = {track.name: track for track in self._tracks}
        nxt: list[_Track] = []

        for index, entry in enumerate(value):
            if not _is_json_record(entry):
                raise ValueError("every track is an object with a name and a pipeline")
            # A name is what a reducer, a log and a panel call this track.
            # Unnamed, it is called by the position it was declared in.
            name = entry["name"] if isinstance(entry.get("name"), str) and entry.get("name") else str(index)
            if name == REDUCE:
                raise ValueError(f"'{REDUCE}' is the reducer's name and cannot be a track's")
            if any(track.name == name for track in nxt):
                raise ValueError(f"two tracks are both called '{name}'")

            # A track already here keeps its pipeline: rebuilding it would
            # destroy what is running inside it because a track beside it was
            # edited.
            existing = previous.pop(name, None)
            pipeline = existing.pipeline if existing else NestedPipeline(f"{self.uuid}:{name}", self._create_service, "Tracks")
            if isinstance(entry.get("pipeline"), list):
                pipeline.set_pipeline(entry["pipeline"])
            if existing is None and self._host:
                pipeline.attach(self._host)

            nxt.append(_Track(name, entry.get("bypass") is True, pipeline))

        # Whatever is left was removed, and holds services nothing will reach.
        for dropped in previous.values():
            dropped.pipeline.destroy()
        self._tracks = nxt

    def _configure_track(self, name: str, config: JsonRecord) -> None:
        track = next((t for t in self._tracks if t.name == name), None)
        pipeline = self._reduce if name == REDUCE else (track.pipeline if track else None)
        if pipeline is None:
            raise ValueError(f"no track called '{name}'")

        if isinstance(config.get("bypass"), bool) and track:
            track.bypass = config["bypass"]
        if isinstance(config.get("pipeline"), list):
            pipeline.set_pipeline(config["pipeline"])
        if _is_json_record(config.get("appendService")):
            pipeline.append(config["appendService"])
        if isinstance(config.get("removeService"), str):
            pipeline.remove(config["removeService"])
        if _is_json_record(config.get("configureService")):
            payload = config["configureService"]
            if isinstance(payload.get("instanceId"), str) and _is_json_record(payload.get("state")):
                pipeline.configure_service(payload["instanceId"], payload["state"])
