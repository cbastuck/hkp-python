from __future__ import annotations

# Service Documentation
# Service ID: unembed
# Service Name: Unembed
# Runtime: hkp-python
# Modes: scores (default — ends the forward pass) or passThrough (reads the
#        early answer and hands the stream on)
# Key Config: model (hkp-asset://…, the weights), config (hkp-asset://…),
#             tokenizer (hkp-asset://…, optional), top, passThrough,
#             inspectLayer, inspectHead
# IO: in=MixedData stream
#     -> out=MixedData scores (or the input stream when passThrough is true)
# Arrays: n/a
# Binary: not supported on its own
# MixedData: its input and output

"""Unembed — where the forward pass ends: the stream becomes scores.

Only the last token's vector is read. It is compared with the vector of every
entry of the vocabulary — the same vectors `token-embed` looked up at the
start — and how well each matches is that entry's score (logit): how much the
model expects that token to come next. Nothing is chosen here; that is what
`sample` is for.

It reports:

  candidates  the ``top`` most likely next tokens with their probabilities,
              before any sampler has touched them
  layers      what the model's top guess would be after each layer that ran so
              far ("logit lens"), with how much attention and MLP added there
  attention   the attention pattern for ``inspectLayer`` and ``inspectHead``

Placed between two layers with ``passThrough: true``, it scores the stream at
that point and hands the stream on untouched, so an early answer can be
inspected anywhere in the stack.
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
        self._pass_through = False
        self._inspect_layer = 0
        self._inspect_head = 0
        self._last_trace: list[JsonRecord] = []
        self._last_candidates: list[JsonRecord] = []
        self._last_layers: list[JsonRecord] = []
        super().__init__(config)

    def configure(self, config: JsonRecord) -> JsonRecord:
        top = number(config, "top")
        if top is not None:
            self._top = int(top)
        if isinstance(config.get("passThrough"), bool):
            self._pass_through = config["passThrough"]
        re_inspect = False
        inspect_layer = number(config, "inspectLayer", least=0.0)
        if inspect_layer is not None and int(inspect_layer) != self._inspect_layer:
            self._inspect_layer = int(inspect_layer)
            re_inspect = True
        inspect_head = number(config, "inspectHead", least=0.0)
        if inspect_head is not None and int(inspect_head) != self._inspect_head:
            self._inspect_head = int(inspect_head)
            re_inspect = True
        state = super().configure(config)
        if re_inspect and self._last_trace and self._host is not None:
            try:
                self._host.notify(self._build_report(), self.uuid)
            except Exception:  # noqa: BLE001
                pass
        return state

    def get_state(self) -> JsonRecord:
        return {
            **super().get_state(),
            "top": self._top,
            "passThrough": self._pass_through,
            "inspectLayer": self._inspect_layer,
            "inspectHead": self._inspect_head,
        }

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
            trace = list(meta.get("_trace") or [])
            with lib.no_grad():
                wte = weights.get("wte.weight")
                last = weights.norm(stream[-1], "ln_f")
                scores = wte @ last
                layers_report = self._score_layers(lib, weights, wte, vocabulary, trace)
        except Problem as problem:
            return self.fail(notify, str(problem))

        best, _ = most_likely(scores, self._top)
        self._last_candidates = candidates(best, vocabulary)
        self._last_layers = layers_report
        self._last_trace = trace
        notify(self._build_report(), None)

        if self._pass_through:
            return MixedData(meta, stream)
        out = {key: value for key, value in meta.items() if key not in ("shape", "_trace")}
        return MixedData({**out, "applied": []}, scores)

    def _score_layers(
        self,
        lib: Any,
        weights: Any,
        wte: Any,
        vocabulary: Any,
        trace: list[JsonRecord],
    ) -> list[JsonRecord]:
        rows: list[JsonRecord] = []
        for idx, entry in enumerate(trace):
            vec = entry.get("vector")
            if not isinstance(vec, list) or len(vec) != weights.width:
                continue
            tensor = lib.tensor(vec, dtype=lib.float32)
            layer_scores = wte @ weights.norm(tensor, "ln_f")
            top_one, _ = most_likely(layer_scores, 1)
            if not top_one:
                continue
            token, prob = top_one[0]
            row: JsonRecord = {
                "step": idx + 1,
                "layer": entry.get("layer", idx),
                "token": token,
                "piece": vocabulary.piece(token) if vocabulary is not None else str(token),
                "p": round(prob, 4),
                "attentionAdded": entry.get("attentionAdded", 0.0),
                "mlpAdded": entry.get("mlpAdded", 0.0),
                "size": entry.get("size", 0.0),
            }
            rows.append(row)
        return rows

    def _build_report(self) -> JsonRecord:
        report: JsonRecord = {
            "candidates": self._last_candidates,
            "layers": self._last_layers,
        }
        with_patterns = [entry for entry in self._last_trace if entry.get("patterns")]
        if with_patterns:
            chosen = next(
                (entry for entry in with_patterns if entry.get("layer") == self._inspect_layer),
                with_patterns[-1],
            )
            patterns = chosen["patterns"]
            head = min(max(self._inspect_head, 0), len(patterns) - 1)
            report["attention"] = {
                "layer": chosen.get("layer", 0),
                "head": head,
                "heads": len(patterns),
                "disabledHeads": chosen.get("disabledHeads", []),
                "mask": chosen.get("mask", "causal"),
                "window": chosen.get("window", 4),
                "temperature": chosen.get("temperature", 1.0),
                "pieces": chosen.get("pieces", []),
                "pattern": patterns[head],
                "patterns": patterns,
            }
        return report
