from __future__ import annotations

# Service Documentation
# Service ID: unembed
# Service Name: Unembed
# Runtime: hkp-python
# Modes: one — the stream to one score per vocabulary entry
# Key Config: model (hkp-asset://…, the weights), config (hkp-asset://…),
#             tokenizer (hkp-asset://…, optional), top
# IO: in=MixedData stream
#     -> out=MixedData: one float32 score per vocabulary entry, for the token
#        that follows; meta = the tokens value with `applied: []`
# Arrays: n/a
# Binary: not supported on its own
# MixedData: its input and output

"""Unembed — where the forward pass ends: the stream becomes scores.

Only the last token's vector is read. It is compared with the vector of every
entry of the vocabulary — the same vectors `token-embed` looked up at the
start — and how well each matches is that entry's score (logit): how much the
model expects that token to come next. Nothing is chosen here; that is what
`sample` is for.

It reports the ``top`` most likely next tokens with their probabilities,
before any sampler has touched them. The tokenizer is only for saying them as
pieces.

What leaves is what hkp-rt's `llm-decode` hands on, so the same `sample` chain
follows either.
"""

from typing import Any

from ..data import MixedData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry
from .llm_model import ModelService, Problem, candidates, most_likely, number, stream_of, torch

UNEMBED_DESCRIPTOR = ServiceRegistryEntry(service_id="unembed", service_name="Unembed")


class UnembedService(ModelService):
    service_id = UNEMBED_DESCRIPTOR.service_id
    service_name = UNEMBED_DESCRIPTOR.service_name
    version = UNEMBED_DESCRIPTOR.version
    capabilities = UNEMBED_DESCRIPTOR.capabilities
    uses_weights = True
    uses_vocabulary = True
    vocabulary_optional = True

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self._top = 10
        super().__init__(config)

    def configure(self, config: JsonRecord) -> JsonRecord:
        top = number(config, "top")
        if top is not None:
            self._top = int(top)
        return super().configure(config)

    def get_state(self) -> JsonRecord:
        return {**super().get_state(), "top": self._top}

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        try:
            lib = torch()
            given = stream_of(input)
            if given is None:
                return self.fail(notify, "expected the stream, as token-embed hands it on")
            meta, stream = given
            weights = self.weights()
            vocabulary = self.vocabulary()
            if stream.shape[1] != weights.width:
                raise Problem(
                    f"a stream {stream.shape[1]} wide is not this model's, which is {weights.width} wide"
                )
            with lib.no_grad():
                last = weights.norm(stream[-1], "ln_f")
                scores = weights.get("wte.weight") @ last
        except Problem as problem:
            return self.fail(notify, str(problem))

        best, _ = most_likely(scores, self._top)
        notify({"candidates": candidates(best, vocabulary)}, None)
        out = {key: value for key, value in meta.items() if key != "shape"}
        return MixedData({**out, "applied": []}, scores)
