from __future__ import annotations

# Service Documentation
# Service ID: attention
# Service Name: Attention
# Runtime: hkp-python
# Modes: one — causal self-attention, as GPT-2 has it
# Key Config: model (hkp-asset://…, the weights), config (hkp-asset://…),
#             layer (which of the model's layers this is)
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

A token may only look backwards: what follows it has not been written yet.

What the heads gathered is **added** to the stream, not put in its place. The
stream that leaves is the one that came in, plus what this layer found.

``layer`` says which of the model's layers this is, and so which weights it
uses. Two of these with the same ``layer`` run that layer twice.
"""

import math
from typing import Any

from ..data import MixedData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry
from .llm_model import ModelService, Problem, length, number, stream_of, torch

ATTENTION_DESCRIPTOR = ServiceRegistryEntry(service_id="attention", service_name="Attention")


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
            with lib.no_grad():
                added, report = self.add(lib, weights, stream)
                out = stream + added
        except Problem as problem:
            return self.fail(notify, str(problem))
        notify({"layer": self._layer, **report, "added": length(added[-1]), "size": length(out[-1])}, None)
        return MixedData(meta, out)

    def add(self, lib: Any, weights: Any, stream: Any) -> tuple[Any, JsonRecord]:
        """What this step adds to the stream, and what it says about it."""
        raise NotImplementedError


class AttentionService(LayerService):
    service_id = ATTENTION_DESCRIPTOR.service_id
    service_name = ATTENTION_DESCRIPTOR.service_name
    version = ATTENTION_DESCRIPTOR.version
    capabilities = ATTENTION_DESCRIPTOR.capabilities

    def add(self, lib: Any, weights: Any, stream: Any) -> tuple[Any, JsonRecord]:
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
        ahead = lib.triu(lib.ones(count, count, dtype=lib.bool), diagonal=1)
        pattern = lib.softmax(match.masked_fill(ahead, float("-inf")), dim=-1)
        gathered = (pattern @ value).transpose(0, 1).reshape(count, width)
        added = gathered @ weights.layer(self._layer, "attn.c_proj.weight") + weights.layer(
            self._layer, "attn.c_proj.bias"
        )
        return added, {"heads": heads}
