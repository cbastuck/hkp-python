from __future__ import annotations

# Service Documentation
# Service ID: attention
# Service Name: Attention
# Runtime: hkp-python
# Modes: causal | window | full — self-attention over the stream
# Key Config: model (hkp-asset://…, the weights), config (hkp-asset://…),
#             layer, mask (causal | window | full), window, temperature,
#             disabledHeads, head
# IO: in=MixedData stream -> out=MixedData stream, with what attention found added
# Arrays: n/a
# Binary: not supported on its own
# MixedData: its input and output

"""Attention — the one place where tokens look at each other.

Everything else in the model works on each token alone. Here every token asks
a question of the tokens before it (its *query*), each of those offers what it
is about (its *key*), and how well the two match decides how much of the
earlier token's content (its *value*) the asking token takes. A model does
this several times side by side — its **heads** — each with its own idea of
what is worth asking.

What can be changed on pretrained weights:

  mask           ``causal`` (default: tokens before it), ``window`` (the last
                 ``window`` tokens), or ``full`` (every token, including what
                 follows)
  temperature    divides the attention scores before softmax: below 1 sharpens
                 focus onto the best match, above 1 flattens it
  disabledHeads  heads switched off (e.g. ``[0, 3]`` or ``"0, 3"``), contributing
                 nothing to the stream

What the heads gathered is **added** to the stream, not put in its place. The
stream that leaves is the one that came in, plus what this layer found.

``layer`` says which of the model's layers this is, and so which weights it
uses. Two of these with the same ``layer`` run that layer twice.

Each call reports the pattern of one head, ``head``; choosing another reports
it from the last call without running again. Every head's pattern goes on in
the stream's record of its layers (``local``, never sent), where `unembed`
reads the one a board asks it to show.
"""

import math
import re
from typing import Any

from ..data import MixedData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry
from .llm_model import ModelService, Problem, layers_of, length, matrix, number, stream_of, torch

ATTENTION_DESCRIPTOR = ServiceRegistryEntry(service_id="attention", service_name="Attention")

MASKS = ("causal", "window", "full")


def _parse_disabled_heads(value: Any) -> list[int] | None:
    """Reads disabled head indices from a list of ints or a string like '0, 3'."""
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            if isinstance(item, bool):
                return None
            if isinstance(item, (int, float)) and int(item) >= 0:
                out.append(int(item))
            elif isinstance(item, str) and item.strip().isdigit():
                out.append(int(item.strip()))
            else:
                return None
        return sorted(set(out))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parts = [p for p in re.split(r"[\s,]+", stripped) if p]
        if all(p.isdigit() for p in parts):
            return sorted({int(p) for p in parts})
        return None
    return None


def _pieces_of(meta: JsonRecord, count: int) -> list[str]:
    raw = meta.get("pieces")
    if isinstance(raw, list) and len(raw) == count:
        return [str(p) for p in raw]
    tokens = meta.get("tokens")
    if isinstance(tokens, list) and len(tokens) == count:
        return [str(t) for t in tokens]
    return [str(i) for i in range(count)]


class LayerService(ModelService):
    """A service that is one step of one of the model's layers."""

    uses_weights = True

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self._layer = 0
        super().__init__(config)

    def configure(self, config: JsonRecord) -> JsonRecord:
        layer = number(config, "layer")
        if layer is not None:
            self._layer = int(layer)
        return super().configure(config)

    def get_state(self) -> JsonRecord:
        return {**super().get_state(), "layer": self._layer}

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        try:
            lib = torch()
            given = stream_of(input)
            if given is None:
                return self.fail(notify, "expected the stream, as token-embed hands it on")
            meta, stream = given
            weights = self.weights()
            if stream.shape[1] != weights.width:
                raise Problem(
                    f"a stream {stream.shape[1]} wide is not this model's, which is {weights.width} wide"
                )
            if not 0 <= self._layer < weights.layers:
                raise Problem(f"this model has no layer {self._layer}: it has {weights.layers}, counted from 0")
            pieces = _pieces_of(meta, int(stream.shape[0]))
            with lib.no_grad():
                added, said, kept = self.add(lib, weights, stream)
                out = stream + added
                token_sizes = [round(size, 3) for size in out.norm(dim=-1).tolist()]
        except Problem as problem:
            return self.fail(notify, str(problem))

        added_size = length(added[-1])
        notify(
            {
                "layer": self._layer,
                **said,
                "added": added_size,
                "size": token_sizes[-1],
                "pieces": pieces,
                "tokenSizes": token_sizes,
            },
            None,
        )
        layers = self.record(
            layers_of(input),
            {"size": token_sizes[-1], "vector": out[-1], "pieces": pieces, **kept},
            added_size,
        )
        local = MixedData.of(input).local  # stream_of found it, so it is one
        return MixedData(meta, out, {**local, "layers": layers})

    def add(self, lib: Any, weights: Any, stream: Any) -> tuple[Any, JsonRecord, JsonRecord]:
        """What this step adds to the stream, what it says about it, and what
        it keeps in the stream's record of its layers."""
        raise NotImplementedError

    def record(self, layers: list[JsonRecord], entry: JsonRecord, added: float) -> list[JsonRecord]:
        """The stream's record of its layers with this step in it."""
        raise NotImplementedError


class AttentionService(LayerService):
    service_id = ATTENTION_DESCRIPTOR.service_id
    service_name = ATTENTION_DESCRIPTOR.service_name
    version = ATTENTION_DESCRIPTOR.version
    capabilities = ATTENTION_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self._mask = "causal"
        self._window = 4
        self._temperature = 1.0
        self._disabled_heads: list[int] = []
        self._head = 0
        #: The last call's report and every head's pattern, so choosing
        #: another head shows it without running again.
        self._last_said: JsonRecord | None = None
        self._last_patterns: Any = None
        super().__init__(config)

    def configure(self, config: JsonRecord) -> JsonRecord:
        mask = config.get("mask")
        if isinstance(mask, str) and mask:
            if mask not in MASKS:
                raise ValueError(f"attention: unknown mask '{mask}'")
            self._mask = mask
        window = number(config, "window", least=1.0)
        if window is not None:
            self._window = max(1, int(window))
        temp = number(config, "temperature", least=0.01)
        if temp is not None:
            self._temperature = temp
        if "disabledHeads" in config:
            parsed = _parse_disabled_heads(config["disabledHeads"])
            if parsed is not None:
                self._disabled_heads = parsed
        head = number(config, "head", least=0.0)
        head_changed = head is not None and int(head) != self._head
        if head is not None:
            self._head = int(head)
        state = super().configure(config)
        if head_changed:
            self._say_head()
        return state

    def get_state(self) -> JsonRecord:
        return {
            **super().get_state(),
            "mask": self._mask,
            "window": self._window,
            "temperature": self._temperature,
            "disabledHeads": list(self._disabled_heads),
            "head": self._head,
        }

    def add(self, lib: Any, weights: Any, stream: Any) -> tuple[Any, JsonRecord, JsonRecord]:
        count, width = stream.shape
        heads = weights.heads
        each = width // heads

        read = weights.norm(stream, f"h.{self._layer}.ln_1")
        # One projection makes all three: the question, the offer, the content.
        projected = read @ weights.layer(self._layer, "attn.c_attn.weight") + weights.layer(
            self._layer, "attn.c_attn.bias"
        )
        query, key, value = (
            part.reshape(count, heads, each).transpose(0, 1) for part in projected.split(width, dim=-1)
        )
        # [heads, asking token, token asked]
        match = query @ key.transpose(-1, -2) / math.sqrt(each)
        if self._temperature != 1.0:
            match = match / self._temperature

        if self._mask == "causal":
            ahead = lib.triu(lib.ones(count, count, dtype=lib.bool), diagonal=1)
            match = match.masked_fill(ahead, float("-inf"))
        elif self._mask == "window":
            rows = lib.arange(count).unsqueeze(1)
            cols = lib.arange(count).unsqueeze(0)
            outside = (cols > rows) | ((rows - cols) >= max(1, self._window))
            match = match.masked_fill(outside, float("-inf"))
        # "full" masks nothing: every token may look at every token, including
        # those after it.

        pattern = lib.softmax(match, dim=-1)
        head_out = pattern @ value

        disabled = [h for h in self._disabled_heads if 0 <= h < heads]
        if disabled:
            pattern = pattern.clone()
            head_out = head_out.clone()
            pattern[disabled] = 0.0
            head_out[disabled] = 0.0

        head_sizes = [round(size, 3) for size in head_out[:, -1].norm(dim=-1).tolist()]
        gathered = head_out.transpose(0, 1).reshape(count, width)
        added = gathered @ weights.layer(self._layer, "attn.c_proj.weight") + weights.layer(
            self._layer, "attn.c_proj.bias"
        )

        setting: JsonRecord = {
            "heads": heads,
            "disabledHeads": disabled,
            "mask": self._mask,
            "window": self._window,
            "temperature": self._temperature,
        }
        chosen = min(max(self._head, 0), heads - 1)
        said: JsonRecord = {
            **setting,
            "head": chosen,
            "headSizes": head_sizes,
            "pattern": matrix(pattern[chosen]),
        }
        self._last_said, self._last_patterns = said, pattern
        return added, said, {**setting, "patterns": pattern}

    def record(self, layers: list[JsonRecord], entry: JsonRecord, added: float) -> list[JsonRecord]:
        return [*layers, {"layer": self._layer, **entry, "attentionAdded": added}]

    def _say_head(self) -> None:
        """Reports the chosen head from the last call, when there was one."""
        said, patterns = self._last_said, self._last_patterns
        if said is None or patterns is None or self._host is None:
            return
        chosen = min(max(self._head, 0), int(patterns.shape[0]) - 1)
        self._last_said = {**said, "head": chosen, "pattern": matrix(patterns[chosen])}
        try:
            self._host.notify(self._last_said, self.uuid)
        except Exception:  # noqa: BLE001 — nobody watching is not a failure
            pass
