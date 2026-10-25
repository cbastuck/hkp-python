from __future__ import annotations

# Service Documentation
# Service ID: map
# Service Name: Map
# Runtime: hkp-python
# Modes: replace | add | overwrite | sensingMode
# Key Config: template, mode, arrayMode, sensingMode, __hkpMount (an address
#             this template mentions)
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
#
# What a pass says about itself is notified when it changes, and is part of the
# state while there is something to say, so a panel opened later shows it;
# configuring it does nothing. The same in every runtime (browser Map.ts,
# hkp-node map.ts, hkp-rt map.h):
#
# - `error`: a term that does not parse — said when the template is configured —
#   or that fails to evaluate. A failing pass stops the flow (None), since a
#   mapping that could not be made has nothing to pass on, and is logged, since
#   nobody may be watching. With an array input, one failing element fails the
#   pass.
# - `warnings`: a term whose value is empty because what it read was not there
#   — `title: params.titel not found`. The flow goes on: an absent field is
#   often legitimate, and a typo is only visible this way.
#
# A lone "=" is the value of its term, whatever the mode; a None one has nothing
# to pass on.

import copy
from typing import Any, Callable

from ..types import (
    JsonRecord,
    NotifyCallback,
    RuntimeHost,
    ServiceConfiguration,
    ServiceRegistryEntry,
)
from ..mount import MOUNT_FIELD, is_mount_reference
from .expression import compile_expression

MAP_DESCRIPTOR = ServiceRegistryEntry(
    service_id="map",
    service_name="Map",
    version="v1",
    capabilities=[],
)


class _TermError(Exception):
    """A term that failed, named by where it sits in the template."""


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
        self._terms: dict[str, Callable[..., Any]] = {}  # keys that end with "=" → expressions
        self._properties: dict[str, Any] = {}  # plain value keys
        # A template that nests objects or arrays keeps its authored shape and is
        # evaluated node by node, rather than being flattened into dotted keys.
        self._structured: Any = None
        self._host: RuntimeHost | None = None
        # What was last notified about the template, so only a change is.
        self._reported_error = ""
        self._reported_warnings: list[str] = []
        # The address a mount reference in this template resolves to. A board
        # that has to *mention* an endpoint — an enclosure URL in a feed, a link
        # somebody is handed — cannot write one down: it is assigned when the
        # board loads. Reserved name: the coordinator writes it, nobody authors
        # it.
        self._mount = ""

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

        if isinstance(config.get(MOUNT_FIELD), str):
            self._mount = config[MOUNT_FIELD]
            self._notify({MOUNT_FIELD: self._mount})

        return self.get_state()

    def get_state(self) -> JsonRecord:
        state: JsonRecord = {
            "mode": self._mode,
            "arrayMode": self._array_mode,
            "template": copy.deepcopy(self._template),
            "sensingMode": self._sensing_mode,
            MOUNT_FIELD: self._mount,
        }
        if self._reported_error:
            state["error"] = self._reported_error
        if self._reported_warnings:
            state["warnings"] = list(self._reported_warnings)
        return state

    def process(self, input: Any, _notify: NotifyCallback | None = None) -> Any:
        if self._sensing_mode:
            self._update_template(
                _sensed(_flatten(input) if isinstance(input, (dict, list)) else {"value": input})
            )
            self._update_sensing_mode(False)
            return None

        per_element = self._array_mode != "single" and isinstance(input, list)
        if not per_element and self._structured is None and not self._terms and not self._properties:
            return {} if self._mode == "replace" else input

        warnings: dict[str, None] = {}  # ordered, without repeats
        try:
            mapped = (
                [self._mapper(entry, warnings) for entry in input]
                if per_element
                else self._mapper(input, warnings)
            )
        except _TermError as error:
            self._report(str(error), list(warnings))
            if self._host:
                self._host.log("error", "service.failed", {"message": str(error)})
            return None
        self._report("", list(warnings))
        return self._addressed(mapped)

    def destroy(self) -> None:
        pass

    # ── Private ────────────────────────────────────────────────────────────────

    def _addressed(self, value: Any) -> Any:
        """Replaces a mount reference in what was produced with its address.

        Applied to the output rather than to the template, so it covers a
        reference written as a static value and one an expression produced, and
        so a template still reads as what the board author wrote.

        An unresolved reference is left as it stands, the same choice a unit
        parameter makes: the owner has not published yet, and a value saying
        ``hkp-mount://…`` is visibly not an address where an empty string would
        look like a field nobody filled in.
        """
        if not self._mount or is_mount_reference(self._mount):
            return value
        if isinstance(value, str):
            return self._mount if is_mount_reference(value) else value
        if isinstance(value, list):
            return [self._addressed(item) for item in value]
        if isinstance(value, dict):
            return {key: self._addressed(item) for key, item in value.items()}
        return value

    def _mapper(self, input: Any, warnings: dict[str, None]) -> Any:
        if self._structured is not None:
            mapped = self._evaluate_node(self._structured, input, warnings, "")
            return mapped if _is_lone_term(self._structured) else self._merge_with_input(mapped, input)

        term_keys = list(self._terms.keys())

        # A lone "=" key maps to a scalar rather than to an object.
        if len(term_keys) == 1 and term_keys[0] == "":
            return self._evaluate_term(self._terms[""], input, warnings, _path_in("", ""))

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
            value = self._evaluate_term(self._terms[key], input, warnings, _path_in("", key))

            if "." in key:
                _merge_at_path(result, value, key)
            elif self._mode == "add" and key in input_record:
                result[key] = input_record[key]
            else:
                result[key] = value

        return result

    def _evaluate_term(
        self, term: Callable[..., Any], params: Any, warnings: dict[str, None], label: str
    ) -> Any:
        """One term's value. A failure is named by ``label``, where the term sits
        in the template; an empty value whose reads found nothing is a warning."""
        missing: list[str] = []
        try:
            value = term(params, missing.append)
        except Exception as exc:
            raise _TermError(f"{label}: {exc}") from exc
        if value is None and missing:
            warnings[f"{label}: {', '.join(missing)} not found"] = None
        return value

    def _report(self, error: str, warnings: list[str]) -> None:
        """Notifies what a pass or a template said about itself, when it changed."""
        if error != self._reported_error:
            self._notify({"error": error})
        if warnings != self._reported_warnings:
            self._notify({"warnings": warnings})
        self._reported_error = error
        self._reported_warnings = warnings

    def _merge_with_input(self, mapped: Any, input: Any) -> Any:
        """Merging only applies when both sides are objects; a template that
        produced an array or a scalar replaces the input whatever the mode."""
        if self._mode == "replace" or not isinstance(mapped, dict) or not isinstance(input, dict):
            return mapped

        if self._mode == "overwrite":
            return {**input, **mapped}
        return {**mapped, **input}

    def _evaluate_node(
        self, node: Any, params: Any, warnings: dict[str, None], label: str
    ) -> Any:
        kind = node[0]
        if kind == "value":
            return copy.deepcopy(node[1])

        if kind == "expression":
            return self._evaluate_term(node[1], params, warnings, label)

        if kind == "array":
            return [
                self._evaluate_node(item, params, warnings, _path_in(label, str(index)))
                for index, item in enumerate(node[1])
            ]

        entries = node[1]
        if _is_lone_term(node):
            return self._evaluate_node(entries[0][2], params, warnings, _path_in(label, ""))

        result: dict[str, Any] = {}
        for key, _dynamic, child in entries:
            value = self._evaluate_node(child, params, warnings, _path_in(label, key))
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
            self._report(_first_parse_error(self._structured, ""), [])
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
        broken = next(
            (
                f"{_path_in('', key)}: {term.parse_error}"
                for key, term in self._terms.items()
                if getattr(term, "parse_error", None)
            ),
            "",
        )
        self._report(broken, [])

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


def _sensed(flat: dict[str, Any]) -> dict[str, Any]:
    """A sensed template: the shape of what arrived, and nothing that runs — as
    in the other runtimes. Sensing builds a template out of input, which arrives
    from wherever the board is fed, and a key ending in "=" is an expression: a
    field named ``pwned=`` would be compiled as code on the next pass. The suffix
    is dropped rather than the field, so the field is still sensed."""
    return {key.rstrip("="): value for key, value in flat.items()}


def _is_lone_term(node: Any) -> bool:
    """An object of a lone "=" key, which stands for its term's value."""
    if node[0] != "object" or len(node[1]) != 1:
        return False
    key, dynamic, _child = node[1][0]
    return dynamic and key == ""


def _path_in(label: str, key: str) -> str:
    """Where a term sits in a template: ``items.0.title``. A lone "=" adds
    nothing to the path, since it stands for the value it sits in; at the top it
    is named "=". The one rule both configuring and a pass name a term by, so
    the same failure is said in the same words."""
    if not key:
        return label or "="
    return f"{label}.{key}" if label else key


def _first_parse_error(node: Any, label: str) -> str:
    """The first term in a structured template that does not parse, named."""
    kind = node[0]
    if kind == "expression":
        parse_error = getattr(node[1], "parse_error", None)
        return f"{label}: {parse_error}" if parse_error else ""
    if kind == "array":
        for index, item in enumerate(node[1]):
            error = _first_parse_error(item, _path_in(label, str(index)))
            if error:
                return error
    if kind == "object":
        for key, _dynamic, child in node[1]:
            error = _first_parse_error(child, _path_in(label, key))
            if error:
                return error
    return ""


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
