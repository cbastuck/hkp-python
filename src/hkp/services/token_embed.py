from __future__ import annotations

# Service Documentation
# Service ID: token-embed
# Service Name: Token Embed
# Runtime: hkp-python
# Modes: one — tokens to the stream the model's layers work on
# Key Config: model (hkp-asset://…, the weights), config (hkp-asset://…)
# IO: in=JSON { tokens, … }
#     -> out=MixedData: float32 [tokens, width]; meta = the input with `shape`
# Arrays: n/a
# Binary: not supported as input
# MixedData: its output

"""Token Embed — where the model's forward pass begins.

A token id means nothing to arithmetic. The model keeps a vector for every
entry of its vocabulary and a vector for every position a token can stand at;
this looks both up and adds them, one sum per token. What leaves is **the
stream**: one vector per token, which every layer after this reads and adds
to, and which `unembed` turns back into scores at the end.

Nothing has looked at anything else yet. At this point each vector knows which
token it is and where it stands, and no more.
"""

from typing import Any

from ..data import MixedData
from ..types import NotifyCallback, ServiceRegistryEntry
from .llm_model import ModelService, Problem, length, tokens_of, torch

TOKEN_EMBED_DESCRIPTOR = ServiceRegistryEntry(service_id="token-embed", service_name="Token Embed")


class TokenEmbedService(ModelService):
    service_id = TOKEN_EMBED_DESCRIPTOR.service_id
    service_name = TOKEN_EMBED_DESCRIPTOR.service_name
    version = TOKEN_EMBED_DESCRIPTOR.version
    capabilities = TOKEN_EMBED_DESCRIPTOR.capabilities
    uses_weights = True

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        tokens = tokens_of(input)
        if not tokens:
            return self.fail(notify, "expected { tokens } with at least one token")
        try:
            lib = torch()
            weights = self.weights()
            for token in tokens:
                if not 0 <= token < weights.vocabulary:
                    raise Problem(f"token {token} is not in this model's vocabulary")
            if len(tokens) > weights.positions:
                raise Problem(
                    f"{len(tokens)} tokens are more than the {weights.positions} positions this model has"
                )
            with lib.no_grad():
                ids = lib.tensor(tokens, dtype=lib.long)
                stream = weights.get("wte.weight")[ids] + weights.get("wpe.weight")[: len(tokens)]
        except Problem as problem:
            return self.fail(notify, str(problem))

        notify({"tokens": len(tokens), "width": weights.width, "size": length(stream[-1])}, None)
        return MixedData({**input, "shape": [len(tokens), weights.width]}, stream)
