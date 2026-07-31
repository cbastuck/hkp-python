from __future__ import annotations

# Service Documentation
# Service ID: map
# Service Name: Map
# Runtime: hkp-python
# Modes: replace | add | overwrite | sensingMode
# Key Config: template, mode, arrayMode, sensingMode
# IO: in=object|array|scalar -> out=mapped payload
# Arrays: maps each element (arrayMode "single" maps the array as a whole)
# Binary: not intended for raw binary
# MixedData: not native in runtime
#
# The template dialect matches the browser, Node and C++ runtimes so all of them
# share one UI: a key ending in "=" is a dynamic term whose value is an
# expression evaluated against `params`, a plain key is a static value, a dot in
# a key nests the result, and a lone "=" key produces a scalar instead of an
# object. Templates that nest objects or arrays keep their shape and are
# evaluated recursively.

import copy
import sys
from typing import Any, Callable

from ..types import (
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    ServiceConfiguration,
    ServiceRegistryEntry,
)
from .expression import compile_expression

MAP_DESCRIPTOR = ServiceRegistryEntry(
    service_id="map",
    service_name="Map",
    version="v1",
    capabilities=[],
)


class MapService:
    service_id = MAP_DESCRIPTOR.service_id
    service_name = MAP_DESCRIPTOR.service_name
    version = MAP_DESCRIPTOR.version
    capabilities = MAP_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._mode: str = "replace"
        self._array_mode: str = "array"
        self._template: Any = {}
        self._sensing_mode = False
        self._terms: dict[str, Callable[[Any], Any]] = {}  # keys that end with "=" → expressions
        self._properties: dict[str, Any] = {}  # plain value keys
        # A template that nests objects or arrays keeps its authored shape and is
        # evaluated node by node, rather than being flattened into dotted keys.
        self._structured: Any = None
        self._host: RuntimeHost | None = None

        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("template"), (dict, list)):
            self._update_template(config["template"])

        if config.get("mode") in ("replace", "add", "overwrite"):
            self._mode = config["mode"]
            self._notify({"mode": self._mode})

        if config.get("arrayMode") in ("array", "single"):
            self._array_mode = config["arrayMode"]
            self._notify({"arrayMode": self._array_mode})

        if isinstance(config.get("sensingMode"), bool):
            self._update_sensing_mode(config["sensingMode"])

        if isinstance(config.get("command"), dict):
            self._run_command(config["command"])

        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "mode": self._mode,
            "arrayMode": self._array_mode,
            "template": copy.deepcopy(self._template),
            "sensingMode": self._sensing_mode,
        }

    def process(self, input: Any, _notify: NotifyCallback | None = None) -> Any:
        if self._sensing_mode:
            self._update_template(
                _flatten(input) if isinstance(input, (dict, list)) else {"value": input}
            )
            self._update_sensing_mode(False)
            return None

        if self._array_mode != "single" and isinstance(input, list):
            return [self._mapper(entry) for entry in input]

        if self._structured is None and not self._terms and not self._properties:
            return {} if self._mode == "replace" else input

        return self._mapper(input)

    def destroy(self) -> None:
        pass

    # ── Private ────────────────────────────────────────────────────────────────

    def _mapper(self, input: Any) -> Any:
        try:
            if self._structured is not None:
                return self._merge_with_input(self._evaluate_node(self._structured, input), input)

            term_keys = list(self._terms.keys())

            # A lone "=" key maps to a scalar rather than to an object.
            if len(term_keys) == 1 and term_keys[0] == "":
                return self._terms[""](input)

            input_record = input if isinstance(input, dict) else {}

            if self._mode == "replace":
                initial: dict[str, Any] = copy.deepcopy(self._properties)
            elif self._mode == "overwrite":
                # template wins: start from input then overwrite with properties
                initial = {**input_record, **copy.deepcopy(self._properties)}
            else:
                # add: input wins: start from properties then overwrite with input
                initial = {**copy.deepcopy(self._properties), **input_record}

            result = initial
            for key in term_keys:
                value = self._terms[key](input)

                if "." in key:
                    _merge_at_path(result, value, key)
                elif self._mode == "add" and key in input_record:
                    result[key] = input_record[key]
                else:
                    result[key] = value

            return result
        except Exception as exc:
            print(f"MapService.process error: {exc}", file=sys.stderr)
            return input

    def _merge_with_input(self, mapped: Any, input: Any) -> Any:
        """Merging only applies when both sides are objects; a template that
        produced an array or a scalar replaces the input whatever the mode."""
        if self._mode == "replace" or not isinstance(mapped, dict) or not isinstance(input, dict):
            return mapped

        if self._mode == "overwrite":
            return {**input, **mapped}
        return {**mapped, **input}

    def _evaluate_node(self, node: Any, params: Any) -> Any:
        kind = node[0]
        if kind == "value":
            return copy.deepcopy(node[1])

        if kind == "expression":
            return node[1](params)

        if kind == "array":
            return [self._evaluate_node(item, params) for item in node[1]]

        entries = node[1]
        if len(entries) == 1:
            key, dynamic, child = entries[0]
            if dynamic and key == "" and child[0] == "expression":
                return self._evaluate_node(child, params)

        result: dict[str, Any] = {}
        for key, _dynamic, child in entries:
            value = self._evaluate_node(child, params)
            if "." in key:
                _merge_at_path(result, value, key)
            else:
                result[key] = value
        return result

    def _update_template(self, template: Any) -> None:
        self._structured = None
        self._terms = {}
        self._properties = {}

        if _is_structured(template):
            self._template = copy.deepcopy(template)
            self._structured = _compile_template(template)
            self._notify({"template": copy.deepcopy(self._template)})
            return

        flat = template if isinstance(template, dict) else {}
        self._template = _flatten(flat)  # stored flat for persistence

        for key, value in flat.items():
            if key.endswith("="):
                self._terms[key[:-1]] = compile_expression(value)
            elif "." in key:
                _merge_at_path(self._properties, value, key)
            else:
                self._properties[key] = value

        self._notify({"template": copy.deepcopy(self._template)})

    def _update_sensing_mode(self, is_active: bool) -> None:
        self._sensing_mode = is_active
        self._notify({"sensingMode": is_active})

    def _run_command(self, command: JsonRecord) -> None:
        if command.get("action") != "inject":
            return

        output = self.process(command.get("params", {}))
        if output is None or self._host is None:
            return

        # Push the injected result through the rest of the pipeline, the way an
        # autonomously emitting service does — the runtime fans the notifications
        # out to its own targets, so none are re-sent here.
        result = self._host.process_from(self.uuid, output, lambda n: None)
        self._host.emit_result(result)

    def _notify(self, payload: JsonRecord) -> None:
        if self._host:
            self._host.notify(payload, self.uuid)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_structured(template: Any) -> bool:
    """An array, or an object with an object/array somewhere in it, keeps its
    shape; anything else is a flat key/value template."""
    if isinstance(template, list):
        return True
    if not isinstance(template, dict):
        return False
    return any(isinstance(value, (dict, list)) for value in template.values())


def _compile_template(template: Any) -> Any:
    """Compiles a template into ("value" | "expression" | "array" | "object", payload)."""
    if isinstance(template, list):
        return ("array", [_compile_template(item) for item in template])

    if isinstance(template, dict):
        entries = []
        for key, value in template.items():
            dynamic = key.endswith("=")
            entries.append(
                (
                    key[:-1] if dynamic else key,
                    dynamic,
                    ("expression", compile_expression(value))
                    if dynamic
                    else _compile_template(value),
                )
            )
        return ("object", entries)

    return ("value", template)


def _flatten(value: Any, prefix: str = "", target: dict[str, Any] | None = None) -> dict[str, Any]:
    if target is None:
        target = {}

    entries = (
        enumerate(value) if isinstance(value, list) else value.items() if isinstance(value, dict) else []
    )
    for key, entry in entries:
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(entry, (dict, list)) and len(entry) > 0:
            _flatten(entry, path, target)
        else:
            target[path] = entry
    return target


def _merge_at_path(destination: dict[str, Any], value: Any, path: str) -> dict[str, Any]:
    segments = path.split(".")
    branch = destination
    for i, segment in enumerate(segments):
        if i == len(segments) - 1:
            branch[segment] = value
        else:
            if not isinstance(branch.get(segment), dict):
                branch[segment] = {}
            branch = branch[segment]
    return destination
