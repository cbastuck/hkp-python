"""Resources a runtime was given, and the one way to their bindings.

A board declares what it needs from outside Readymade — an Ollama server, a
Postgres database, an S3 bucket — by kind (``ollama``, ``postgres``, ``s3``),
and service state names one by reference, ``hkp-resource://<id>``, as a whole
field. ``get_state`` echoes the reference, and saving a board writes back what
was configured: addresses and credentials never enter service state.

The bindings arrive with the runtime's create payload, or on
``POST /runtimes/<id>/resources`` — and again whenever a binding changes, which
is the point: a service resolves its reference at the moment it uses it, so the
next use gets the new binding without anything being reconfigured.

The format matches ``hkp-frontend/src/runtime/board/resources.ts``,
``hkp-node/src/resources.ts`` and ``hkp-rt/lib/include/resources.h``: a board
written against one runtime has to open against another.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable


RESOURCE_SCHEME = "hkp-resource://"
RESOURCE_KINDS = ("ollama", "postgres", "s3")

_ID = r"[A-Za-z0-9_.\-]+"
_ID_PATTERN = re.compile(rf"^{_ID}$")
_WHOLE_REFERENCE = re.compile(rf"^hkp-resource://({_ID})$")
_ANY_REFERENCE = re.compile(rf"hkp-resource://({_ID})")

ResourceBinding = dict[str, Any]


@dataclass
class ResourceResolution:
    #: The resolved binding (without ``unreachable``), or None when there is none.
    resource: dict[str, Any] | None
    #: Why there is none, or "" when there is.
    problem: str


def is_resource_kind(value: Any) -> bool:
    return isinstance(value, str) and value in RESOURCE_KINDS


def parse_resource_ref(value: Any) -> str | None:
    """The id a whole-value reference names, or None for anything else."""
    if not isinstance(value, str):
        return None
    match = _WHOLE_REFERENCE.match(value)
    return match.group(1) if match else None


def format_resource_ref(resource_id: str) -> str:
    return f"{RESOURCE_SCHEME}{resource_id}"


def referenced_resources(value: Any) -> list[str]:
    """Every resource id a value mentions, however deeply it is nested."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for resource_id in _ANY_REFERENCE.findall(node):
                if resource_id not in found:
                    found.append(resource_id)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    return found


def read_resource_binding(
    value: Any, fallback_id: str | None = None
) -> ResourceBinding | None:
    """One binding off the wire, or None when it is not one."""
    if not isinstance(value, dict):
        return None
    resource_id = value.get("id") if isinstance(value.get("id"), str) else fallback_id
    kind = value.get("kind")
    if not resource_id or not _ID_PATTERN.match(resource_id) or not is_resource_kind(kind):
        return None

    raw_unreachable = value.get("unreachable")
    unreachable = (
        raw_unreachable.strip()
        if isinstance(raw_unreachable, str) and raw_unreachable.strip()
        else None
    )

    if kind == "ollama":
        url = value.get("url").strip() if isinstance(value.get("url"), str) else ""
        if not url and not unreachable:
            return None
        binding: ResourceBinding = {"id": resource_id, "kind": "ollama", "url": url}
        if unreachable:
            binding["unreachable"] = unreachable
        return binding

    if kind == "postgres":
        url = value.get("url").strip() if isinstance(value.get("url"), str) else ""
        if not url and not unreachable:
            return None
        binding = {"id": resource_id, "kind": "postgres", "url": url}
        for key in ("user", "password", "database"):
            if isinstance(value.get(key), str):
                binding[key] = value[key]
        if unreachable:
            binding["unreachable"] = unreachable
        return binding

    if kind == "s3":
        endpoint = (
            value.get("endpoint").strip()
            if isinstance(value.get("endpoint"), str)
            else ""
        )
        if not endpoint and not unreachable:
            return None
        binding = {"id": resource_id, "kind": "s3", "endpoint": endpoint}
        for key in ("region", "bucket", "accessKey", "secretKey"):
            if isinstance(value.get(key), str):
                binding[key] = value[key]
        if unreachable:
            binding["unreachable"] = unreachable
        return binding

    return None


def read_resources_payload(value: Any) -> dict[str, ResourceBinding | None]:
    """Reads a resources payload off the wire: a map of id to binding, where
    ``None`` removes one, or a list of bindings."""
    entries: dict[str, ResourceBinding | None] = {}
    if isinstance(value, list):
        for item in value:
            binding = read_resource_binding(item)
            if binding:
                entries[binding["id"]] = binding
        return entries
    if not isinstance(value, dict):
        return entries
    for resource_id, item in value.items():
        if item is None:
            if _ID_PATTERN.match(resource_id):
                entries[resource_id] = None
            continue
        binding = read_resource_binding(item, resource_id)
        if binding:
            entries[resource_id] = {**binding, "id": resource_id}
    return entries


def _version(binding: ResourceBinding | None) -> str:
    return json.dumps(binding, sort_keys=True)


class ResourceStore:
    """The resource bindings one runtime was given."""

    def __init__(self) -> None:
        self._bindings: dict[str, ResourceBinding] = {}
        self._listeners: dict[str, set[Callable[[str], None]]] = {}
        self._lock = threading.RLock()

    def replace(self, entries: dict[str, ResourceBinding]) -> None:
        """Replaces everything held."""
        with self._lock:
            previous = self._bindings
            self._bindings = dict(entries)
            touched = set(previous) | set(self._bindings)
        for resource_id in touched:
            if _version(previous.get(resource_id)) != _version(
                self._bindings.get(resource_id)
            ):
                self._changed(resource_id)

    def merge(self, entries: dict[str, ResourceBinding | None]) -> None:
        """Adds, replaces or removes (``None``) individual bindings."""
        for resource_id, entry in entries.items():
            with self._lock:
                previous = self._bindings.get(resource_id)
                if entry is None:
                    self._bindings.pop(resource_id, None)
                else:
                    self._bindings[resource_id] = entry
            if _version(previous) != _version(entry):
                self._changed(resource_id)

    def ids(self) -> list[str]:
        """The ids held, for saying what a runtime knows about."""
        with self._lock:
            return list(self._bindings)

    def binding(self, resource_id: str) -> ResourceBinding | None:
        with self._lock:
            return self._bindings.get(resource_id)

    def subscribe(
        self, resource_id: str, listener: Callable[[str], None]
    ) -> Callable[[], None]:
        """Calls ``listener`` with the id whenever the binding changes or goes away."""
        with self._lock:
            self._listeners.setdefault(resource_id, set()).add(listener)

        def unsubscribe() -> None:
            with self._lock:
                listeners = self._listeners.get(resource_id)
                if listeners is not None:
                    listeners.discard(listener)
                    if not listeners:
                        self._listeners.pop(resource_id, None)

        return unsubscribe

    def resolve(
        self, reference: str, expected_kind: str | None = None
    ) -> ResourceResolution:
        """Resolves a ``hkp-resource://<id>`` reference for one use, or answers
        with a sentence saying why there is none."""
        resource_id = parse_resource_ref(reference)
        if not resource_id:
            return ResourceResolution(
                None, f"{json.dumps(reference)} is not a resource reference"
            )
        with self._lock:
            binding = self._bindings.get(resource_id)
        if binding is None:
            return ResourceResolution(None, f'resource "{resource_id}" is not bound')
        if expected_kind and binding.get("kind") != expected_kind:
            return ResourceResolution(
                None,
                f'resource "{resource_id}" is of kind "{binding.get("kind")}", not "{expected_kind}"',
            )
        unreachable = binding.get("unreachable")
        if isinstance(unreachable, str) and unreachable:
            return ResourceResolution(None, unreachable)
        resolved = {k: v for k, v in binding.items() if k != "unreachable"}
        return ResourceResolution(resolved, "")

    def _changed(self, resource_id: str) -> None:
        with self._lock:
            listeners = list(self._listeners.get(resource_id, ()))
        for listener in listeners:
            try:
                listener(resource_id)
            except Exception as error:  # noqa: BLE001
                print(f'[resources] listener for "{resource_id}" failed: {error}')
