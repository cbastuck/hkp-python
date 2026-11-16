from __future__ import annotations

# Service Documentation
# Service ID: sample
# Service Name: Sample
# Runtime: hkp-python
# Modes: repetition-penalty | top-k | top-p | min-p | temperature — scores to
#        scores; pick | greedy — scores to tokens, one appended
# Key Config: tokenizer (hkp-asset://…, optional), mode, k, p, minP,
#             temperature, penalty, lastN, seed, top
# IO: in=MixedData scores (as unembed or another sample hands them on)
#     -> out=MixedData scores, or JSON { tokens, pieces, … } for pick / greedy
# Arrays: n/a
# Binary: not supported on its own
# MixedData: its input, and its output in the modes that do not choose

"""Sample — one step of choosing the next token.

The model hands over a score for every entry of its vocabulary. What happens
between that and one token being chosen is a chain of small steps, and each
instance of this service is one of them, by ``mode``. The order on the board
is the order they are applied in.

  repetition-penalty  lowers the scores of tokens among the last ``lastN``
  top-k               keeps the ``k`` highest
  top-p               keeps the most likely until their probabilities sum to ``p``
  min-p               keeps what is at least ``minP`` as likely as the best
  temperature         divides the scores: below 1 sharper, above 1 flatter
  pick                chooses one at random, by probability, and appends it
  greedy              chooses the highest, and appends it

A step that narrows removes candidates; it does not reorder the scores. A
removed candidate is -infinity, so the next step reads the shape this one
wrote. ``pick`` and ``greedy`` end the chain: what leaves them is tokens again,
one longer.

Each call reports how many candidates are left and the ``top`` most likely of
them. The tokenizer is only for saying them as pieces; without one the service
does the same and reports ids.

The same service as hkp-rt's `sample`, whose steps are llama.cpp's samplers of
the same names; these are written out to do what those do.
"""

from typing import Any

from ..data import MixedData
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry
from .llm_model import (
    ModelService,
    Problem,
    candidates,
    most_likely,
    number,
    scores_of,
    tokens_of,
    torch,
)

SAMPLE_DESCRIPTOR = ServiceRegistryEntry(service_id="sample", service_name="Sample")

MODES = ("repetition-penalty", "top-k", "top-p", "min-p", "temperature", "pick", "greedy")


def _trimmed(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


class SampleService(ModelService):
    service_id = SAMPLE_DESCRIPTOR.service_id
    service_name = SAMPLE_DESCRIPTOR.service_name
    version = SAMPLE_DESCRIPTOR.version
    capabilities = SAMPLE_DESCRIPTOR.capabilities
    uses_vocabulary = True
    vocabulary_optional = True

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self._mode = "pick"
        self._k = 40
        self._p = 0.95
        self._min_p = 0.05
        self._temperature = 0.8
        self._penalty = 1.1
        self._last_n = 64
        #: None: a different sequence of choices each time.
        self._seed: int | None = None
        self._top = 10
        #: Kept between calls, so a seed gives one sequence of choices and not
        #: the same choice every round.
        self._generator: Any = None
        super().__init__(config)

    def configure(self, config: JsonRecord) -> JsonRecord:
        mode = config.get("mode")
        if isinstance(mode, str):
            if mode not in MODES:
                raise ValueError(f"sample: unknown mode '{mode}'")
            self._mode = mode
        for key, attribute, whole in (
            ("k", "_k", True),
            ("p", "_p", False),
            ("minP", "_min_p", False),
            ("temperature", "_temperature", False),
            ("penalty", "_penalty", False),
            ("lastN", "_last_n", True),
            ("top", "_top", True),
        ):
            value = number(config, key)
            if value is not None:
                setattr(self, attribute, int(value) if whole else value)
        if "seed" in config:
            seed = config["seed"]
            self._seed = seed if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 else None
            self._generator = None
        return super().configure(config)

    def get_state(self) -> JsonRecord:
        return {
            **super().get_state(),
            "mode": self._mode,
            "k": self._k,
            "p": self._p,
            "minP": self._min_p,
            "temperature": self._temperature,
            "penalty": self._penalty,
            "lastN": self._last_n,
            "seed": self._seed,
            "top": self._top,
        }

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        try:
            lib = torch()
            given = scores_of(input)
            if given is None:
                return self.fail(notify, "expected scores, as unembed hands them on")
            meta, scores = given
            vocabulary = self.vocabulary()
        except Problem as problem:
            return self.fail(notify, str(problem))
        if vocabulary is not None and scores.numel() != vocabulary.size:
            return self.fail(notify, "these scores are not for this tokenizer's vocabulary")
        tokens = tokens_of(meta) or []
        remaining = int(lib.isfinite(scores).sum())
        if remaining == 0:
            return self.fail(notify, "no candidate is left to choose from")

        with lib.no_grad():
            if self._mode in ("pick", "greedy"):
                probabilities = lib.softmax(scores.double(), dim=-1)
                if self._mode == "greedy":
                    chosen = int(lib.argmax(scores))
                else:
                    chosen = int(lib.multinomial(probabilities, 1, generator=self._random(lib)))
                piece = vocabulary.piece(chosen) if vocabulary is not None else None
                said: JsonRecord = {"token": chosen, "p": float(probabilities[chosen])}
                if piece is not None:
                    said["piece"] = piece
                notify({"mode": self._mode, "chosen": said, "remaining": remaining}, None)

                out = {key: value for key, value in meta.items() if key != "applied"}
                out["tokens"] = [*tokens, chosen]
                if piece is not None and isinstance(out.get("pieces"), list):
                    out["pieces"] = [*out["pieces"], piece]
                return out

            narrowed = self._apply(lib, scores, tokens)

        best, left = most_likely(narrowed, self._top)
        notify(
            {"mode": self._mode, "remaining": left, "candidates": candidates(best, vocabulary)},
            None,
        )
        applied = meta.get("applied")
        return MixedData(
            {**meta, "applied": [*(applied if isinstance(applied, list) else []), self._describe()]},
            narrowed,
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _random(self, lib: Any) -> Any:
        if self._generator is None:
            self._generator = lib.Generator()
            if self._seed is None:
                self._generator.seed()
            else:
                self._generator.manual_seed(self._seed)
        return self._generator

    def _apply(self, lib: Any, scores: Any, tokens: list[int]) -> Any:
        """The scores after this step. A new tensor: what came in is somebody
        else's to keep."""
        removed = float("-inf")
        mode = self._mode
        if mode == "temperature":
            if self._temperature <= 0:
                # Nothing left to divide by: only the best stays.
                return lib.where(scores == scores.max(), scores, removed)
            return scores / self._temperature
        if mode == "repetition-penalty":
            out = scores.clone()
            recent = [t for t in tokens[-self._last_n :] if 0 <= t < out.numel()] if self._last_n > 0 else []
            if recent and self._penalty > 0:
                seen = lib.tensor(sorted(set(recent)), dtype=lib.long)
                values = out[seen]
                # Away from likely either way: a positive score shrinks, a
                # negative one grows.
                out[seen] = lib.where(values <= 0, values * self._penalty, values / self._penalty)
            return out
        if mode == "top-k":
            alive = int(lib.isfinite(scores).sum())
            if self._k <= 0 or self._k >= alive:
                return scores.clone()
            kept = lib.topk(scores, self._k).indices
        else:
            probabilities = lib.softmax(scores.double(), dim=-1)
            if mode == "min-p":
                kept = lib.nonzero(probabilities >= self._min_p * probabilities.max()).flatten()
            else:  # top-p
                ordered = lib.sort(probabilities, descending=True)
                # Up to and including the one that takes the sum to `p`.
                before = lib.cumsum(ordered.values, dim=-1) - ordered.values
                count = max(int((before < self._p).sum()), 1)
                kept = ordered.indices[:count]
        out = lib.full_like(scores, removed)
        out[kept] = scores[kept]
        return out

    def _describe(self) -> str:
        """The step as its scores say it was applied."""
        if self._mode == "top-k":
            return f"top-k {self._k}"
        if self._mode == "top-p":
            return f"top-p {_trimmed(self._p)}"
        if self._mode == "min-p":
            return f"min-p {_trimmed(self._min_p)}"
        if self._mode == "temperature":
            return f"temperature {_trimmed(self._temperature)}"
        if self._mode == "repetition-penalty":
            return f"repetition-penalty {_trimmed(self._penalty)} over {self._last_n}"
        return self._mode
