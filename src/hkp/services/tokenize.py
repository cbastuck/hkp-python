from __future__ import annotations

# Service Documentation
# Service ID: tokenize
# Service Name: Tokenize
# Runtime: hkp-python
# Modes: one — text to the token ids a model reads
# Key Config: tokenizer (hkp-asset://…), addSpecial
# IO: in=String or JSON ({text} | {prompt})
#     -> out=JSON { tokens: [id…], pieces: [text…], promptTokens }
# Arrays: n/a
# Binary: not supported
# MixedData: not supported

"""Tokenize — a string becomes token ids.

A model does not read characters. Its vocabulary is a fixed list of pieces —
whole words, parts of words, single bytes — and text is cut into the longest
pieces that list holds. ``pieces`` shows where the cuts fell; ``tokens`` is
what the model is given.

Everything this hands on counts as the prompt: ``promptTokens`` is how many
tokens there are, so that what is appended later can be told from it.

The same service as hkp-rt's `tokenize`, which takes its vocabulary from the
model file; here it is an asset of its own, a ``tokenizer.json``.
"""

from typing import Any

from ..data import TextData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry
from .llm_model import ModelService, Problem

TOKENIZE_DESCRIPTOR = ServiceRegistryEntry(service_id="tokenize", service_name="Tokenize")


def text_of(input: Any) -> str | None:
    """The text an input carries: itself, or its ``text`` or ``prompt``."""
    if isinstance(input, TextData):
        return input.text
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        for key in ("text", "prompt"):
            if isinstance(input.get(key), str):
                return input[key]
    return None


class TokenizeService(ModelService):
    service_id = TOKENIZE_DESCRIPTOR.service_id
    service_name = TOKENIZE_DESCRIPTOR.service_name
    version = TOKENIZE_DESCRIPTOR.version
    capabilities = TOKENIZE_DESCRIPTOR.capabilities
    uses_vocabulary = True

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        #: Whether what the tokenizer puts around a text by itself is added.
        #: GPT-2's puts nothing.
        self._add_special = True
        super().__init__(config)

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("addSpecial"), bool):
            self._add_special = config["addSpecial"]
        return super().configure(config)

    def get_state(self) -> JsonRecord:
        return {**super().get_state(), "addSpecial": self._add_special}

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        text = text_of(input)
        if text is None:
            return self.fail(notify, "expected text, { text } or { prompt }")
        try:
            vocabulary = self.vocabulary()
            tokens = vocabulary.encode(text, self._add_special)
        except Problem as problem:
            return self.fail(notify, str(problem))
        pieces = [vocabulary.piece(token) for token in tokens]
        notify({"count": len(tokens), "pieces": pieces}, None)
        return {"tokens": tokens, "pieces": pieces, "promptTokens": len(tokens)}
