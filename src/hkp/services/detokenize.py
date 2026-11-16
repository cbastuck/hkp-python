from __future__ import annotations

# Service Documentation
# Service ID: detokenize
# Service Name: Detokenize
# Runtime: hkp-python
# Modes: one — token ids back to text
# Key Config: tokenizer (hkp-asset://…)
# IO: in=JSON { tokens, promptTokens? }
#     -> out=JSON { tokens, pieces, promptTokens, text, end }
# Arrays: n/a
# Binary: not supported
# MixedData: not supported

"""Detokenize — token ids become text again.

The last step of a round, and what a Loop around it asks its condition of:

  text   what has been generated so far — the tokens after the prompt, with
         control tokens left out
  end    whether the last token is one the model ends with (for GPT-2,
         ``<|endoftext|>``)

``pieces`` is filled in for every token, so the whole sequence can be shown
with its cuts.

The same service as hkp-rt's `detokenize`.
"""

from typing import Any

from ..types import JsonRecord, NotifyCallback, ServiceRegistryEntry
from .llm_model import ModelService, Problem, tokens_of

DETOKENIZE_DESCRIPTOR = ServiceRegistryEntry(service_id="detokenize", service_name="Detokenize")


class DetokenizeService(ModelService):
    service_id = DETOKENIZE_DESCRIPTOR.service_id
    service_name = DETOKENIZE_DESCRIPTOR.service_name
    version = DETOKENIZE_DESCRIPTOR.version
    capabilities = DETOKENIZE_DESCRIPTOR.capabilities
    uses_vocabulary = True

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        tokens = tokens_of(input)
        if tokens is None:
            return self.fail(notify, "expected { tokens }")
        try:
            vocabulary = self.vocabulary()
        except Problem as problem:
            return self.fail(notify, str(problem))
        for token in tokens:
            if not vocabulary.known(token):
                return self.fail(notify, f"token {token} is not in this tokenizer's vocabulary")

        prompt = input.get("promptTokens")
        if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0:
            prompt = 0
        prompt = min(prompt, len(tokens))
        out: JsonRecord = dict(input)
        out["pieces"] = [vocabulary.piece(token) for token in tokens]
        out["promptTokens"] = prompt
        out["text"] = vocabulary.text(tokens[prompt:])
        out["end"] = len(tokens) > prompt and tokens[-1] in vocabulary.special
        return out
