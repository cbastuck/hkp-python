from __future__ import annotations

# Service Documentation
# Service ID: mlp
# Service Name: MLP
# Runtime: hkp-python
# Modes: one — the feed-forward half of a GPT-2 layer
# Key Config: model (hkp-asset://…, the weights), config (hkp-asset://…),
#             layer (which of the model's layers this is)
# IO: in=MixedData stream -> out=MixedData stream, with what it computed added
# Arrays: n/a
# Binary: not supported on its own
# MixedData: its input and output

"""MLP — what each token works out on its own.

Attention moves information between tokens; this is where something is done
with it. Each token's vector is spread out to four times its width, put
through a curve that lets some of it pass and holds the rest back, and brought
back to size. No token sees another here.

Most of a model's weights are in these, and much of what it "knows" — that
Paris goes with France — is read out of the stream by one of them.

Like attention, the result is **added** to the stream.
"""

from typing import Any

from ..types import JsonRecord, ServiceRegistryEntry
from .attention import LayerService

MLP_DESCRIPTOR = ServiceRegistryEntry(service_id="mlp", service_name="MLP")


class MlpService(LayerService):
    service_id = MLP_DESCRIPTOR.service_id
    service_name = MLP_DESCRIPTOR.service_name
    version = MLP_DESCRIPTOR.version
    capabilities = MLP_DESCRIPTOR.capabilities

    def add(self, lib: Any, weights: Any, stream: Any) -> tuple[Any, JsonRecord]:
        read = weights.norm(stream, f"h.{self._layer}.ln_2")
        wide = read @ weights.layer(self._layer, "mlp.c_fc.weight") + weights.layer(
            self._layer, "mlp.c_fc.bias"
        )
        # GPT-2's curve: GELU in the form its authors wrote it.
        passed = lib.nn.functional.gelu(wide, approximate="tanh")
        added = passed @ weights.layer(self._layer, "mlp.c_proj.weight") + weights.layer(
            self._layer, "mlp.c_proj.bias"
        )
        return added, {}
