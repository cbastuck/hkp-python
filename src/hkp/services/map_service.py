from __future__ import annotations

import copy
from typing import Any

from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

MAP_DESCRIPTOR = ServiceRegistryEntry(
    service_id="map",
    service_name="Map",
    version="v1",
    capabilities=[],
)

# Builtins available inside map expression evaluation
_EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "abs": abs,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "round": round,
        "str": str,
        "sum": sum,
        "type": type,
    }
}


class _ParamsProxy:
    """Wraps a dict so map expressions can use attribute-style access (params.key)."""

    def __init__(self, data: Any) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if not isinstance(data, dict):
            raise AttributeError(name)
        val = data.get(name)
        if isinstance(val, dict):
            return _ParamsProxy(val)
        return val

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_data"))


class MapService:
    service_id = MAP_DESCRIPTOR.service_id
    service_name = MAP_DESCRIPTOR.service_name
    version = MAP_DESCRIPTOR.version
    capabilities = MAP_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._mode: str = "replace"
        self._template: JsonRecord = {}
        self._sensing_mode = False
        self._terms: dict[str, Any] = {}      # keys that end with "=" → expressions
        self._properties: dict[str, Any] = {}  # plain value keys

        if config.state:
            self.configure(config.state)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("template"), dict):
            self._update_template(config["template"])

        if config.get("mode") in ("replace", "add", "overwrite"):
            self._mode = config["mode"]

        if isinstance(config.get("sensingMode"), bool):
            self._sensing_mode = config["sensingMode"]

        return self.get_state()

    def get_state(self) -> JsonRecord:
        return {
            "mode": self._mode,
            "template": dict(self._template),
            "sensingMode": self._sensing_mode,
        }

    def process(self, input: Any, _notify: NotifyCallback) -> Any:
        if self._sensing_mode:
            if isinstance(input, dict):
                self._update_template(input)
            else:
                self._update_template({"value": input})
            self._sensing_mode = False
            return None

        if isinstance(input, list):
            return [self._mapper(entry) for entry in input]

        if not self._terms and not self._properties:
            return {} if self._mode == "replace" else input

        return self._mapper(input)

    def set_host(self, host: Any) -> None:
        pass

    def destroy(self) -> None:
        pass

    # ── Private ────────────────────────────────────────────────────────────────

    def _mapper(self, input: Any) -> Any:
        try:
            term_keys = list(self._terms.keys())

            # Single anonymous expression — evaluate against the whole input
            if len(term_keys) == 1 and term_keys[0] == "":
                return _evaluate_expression(self._terms[""], input)

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
                expression = self._terms[key]
                value = _evaluate_expression(expression, input)

                if "." in key:
                    _merge_at_path(result, value, key)
                elif self._mode == "add" and key in input_record:
                    result[key] = input_record[key]
                else:
                    result[key] = value

            return result
        except Exception as exc:
            import sys
            print(f"MapService.process error: {exc}", file=sys.stderr)
            return input

    def _update_template(self, template: dict[str, Any]) -> None:
        self._template = _flatten_object(template)
        self._properties = {}
        self._terms = {}

        for key, value in template.items():
            if key.endswith("="):
                self._terms[key[:-1]] = value
            elif "." in key:
                _merge_at_path(self._properties, value, key)
            else:
                self._properties[key] = value


# ── Helpers ────────────────────────────────────────────────────────────────────


def _flatten_object(
    value: dict[str, Any],
    prefix: str = "",
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target is None:
        target = {}
    for key, entry in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(entry, dict) and entry:
            _flatten_object(entry, path, target)
        else:
            target[path] = entry
    return target


def _evaluate_expression(expression: Any, params: Any) -> Any:
    if not isinstance(expression, str):
        return expression
    proxy = _ParamsProxy(params) if isinstance(params, dict) else params
    try:
        return eval(expression, _EVAL_GLOBALS, {"params": proxy})  # noqa: S307
    except Exception:
        return expression


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
