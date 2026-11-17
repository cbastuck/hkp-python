from __future__ import annotations

"""What the services that lay a language model out on a board share.

`text-generation` answers in one call. These take that call apart: tokenize,
then one service per step of the model itself — token-embed, attention, mlp,
unembed — then sample and detokenize. Each reads what the one before it handed
on, and nothing here is hidden inside another.

**What a service loads** is a board asset, named as ``hkp-asset://<id>``:

  tokenizer  a ``tokenizer.json``: the vocabulary and how text is cut into it
  model      the weights, a ``.safetensors`` file that says its ``sha256``
  config     the model's ``config.json``: how many layers, heads, how wide

The asset store fetches each once per server, and its model cache parses each
once per runtime, so thirty services naming one model hold one model.

**Tokens** travel as JSON::

    { "tokens": [15496, 995], "pieces": ["Hello", " world"], "promptTokens": 2 }

**The stream** is the model's working memory: one vector per token, which
every layer reads and adds to. It travels as MixedData — a float32 tensor of
shape ``[tokens, width]`` with the tokens value and ``shape`` beside it — and
stays a tensor between services of one runtime.

**Scores** are MixedData too: one float32 per vocabulary entry, with the
tokens they follow and what has been done to them (``applied``) beside it. A
candidate a sampler removed is -infinity, so the entries never move and every
sampler reads the shape it writes.

The values are the ones hkp-rt's services of the same names pass
(``hkp-rt/lib/src/services/llm_model.h``), so `sample` and `detokenize` mean
the same on both.
"""

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..assets import parse_asset_ref
from ..data import MixedData
from ..models import ModelLease
from ..types import JsonRecord, NotifyCallback, RuntimeHost, ServiceConfiguration

INSTALL_HINT = (
    "PyTorch is not installed — run in hkp-python: uv sync --extra transformer"
)

#: How long a service waits for the assets it names to be known. A pipeline
#: nested in a Loop is built before the Loop is attached to its board's assets,
#: so for a moment the references of what is inside it name nothing.
_ASSETS_KNOWN_WITHIN_S = 2.0
#: At most this often is a fetch's progress reported.
_PROGRESS_EVERY_S = 0.25


class Problem(Exception):
    """Why something cannot be loaded or run, in words a panel can show."""


def torch() -> Any:
    try:
        import torch as module  # noqa: PLC0415 — an optional extra
    except ImportError as error:
        raise Problem(INSTALL_HINT) from error
    return module


# ── Values ───────────────────────────────────────────────────────────────────


def tokens_of(value: Any) -> list[int] | None:
    """The token ids a tokens value carries, or None when it is not one."""
    if not isinstance(value, dict):
        return None
    tokens = value.get("tokens")
    if not isinstance(tokens, list):
        return None
    if not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens):
        return None
    return tokens


def number(config: JsonRecord, key: str, least: float = 0.0) -> float | None:
    """A number, or one written as text — which is how a facade knob sends its
    value. None when there is none worth taking."""
    value = config.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= least:
        return float(value)
    return None


def _tensor(payload: Any, shape: list[int]) -> Any:
    """A payload as a float32 tensor: what it already is inside one runtime,
    and little-endian bytes when it arrived from another."""
    lib = torch()
    if isinstance(payload, lib.Tensor):
        return payload
    if isinstance(payload, (bytes, bytearray, memoryview)):
        data = bytearray(payload)
        count = math.prod(shape)
        if len(data) != count * 4:
            return None
        return lib.frombuffer(data, dtype=lib.float32).reshape(shape) if count else lib.zeros(shape)
    return None


def scores_of(value: Any) -> tuple[JsonRecord, Any] | None:
    """Scores as their JSON side and a tensor with one entry per vocabulary
    entry, or None when the value is not scores."""
    mixed = MixedData.of(value)
    if mixed is None or tokens_of(mixed.meta) is None:
        return None
    payload = mixed.payload
    count = mixed.size // 4
    scores = _tensor(payload, [count])
    if scores is None or scores.dim() != 1 or scores.numel() == 0:
        return None
    return mixed.meta, scores


def stream_of(value: Any) -> tuple[JsonRecord, Any] | None:
    """The stream as its JSON side and a ``[tokens, width]`` tensor, or None
    when the value is not one."""
    mixed = MixedData.of(value)
    if mixed is None:
        return None
    tokens = tokens_of(mixed.meta)
    shape = mixed.meta.get("shape")
    if (
        tokens is None
        or not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(side, int) and side > 0 for side in shape)
        or shape[0] != len(tokens)
    ):
        return None
    stream = _tensor(mixed.payload, shape)
    if stream is None or list(stream.shape) != shape:
        return None
    return mixed.meta, stream


def layers_of(value: Any) -> list[JsonRecord]:
    """What the layers the stream went through recorded about it, earliest
    first: for each, ``layer``, what attention and MLP added, the last token's
    vector (a tensor) and the attention patterns (a ``[heads, tokens, tokens]``
    tensor) with how attention was set. Kept in the stream's ``local`` side,
    so it costs a reference per layer and never leaves the runtime; a stream
    that came from another runtime has none."""
    mixed = MixedData.of(value)
    layers = mixed.local.get("layers") if mixed is not None else None
    return list(layers) if isinstance(layers, list) else []


def matrix(tensor: Any) -> list[list[float]]:
    """A 2D tensor as a panel reads it: rows of numbers, four places."""
    return [[round(cell, 4) for cell in row] for row in tensor.tolist()]


def most_likely(scores: Any, count: int) -> tuple[list[tuple[int, float]], int]:
    """The ``count`` most likely tokens among those still in the running, each
    with the probability it has among them, and how many are in the running."""
    lib = torch()
    remaining = int(lib.isfinite(scores).sum())
    if remaining == 0:
        return [], 0
    probabilities = lib.softmax(scores.double(), dim=-1)
    best = lib.topk(probabilities, min(max(count, 0), remaining))
    return list(zip(best.indices.tolist(), best.values.tolist())), remaining


def length(vector: Any) -> float:
    """How long a vector is, as a report says it."""
    return round(float(vector.norm()), 3)


# ── What is loaded ───────────────────────────────────────────────────────────


class Vocabulary:
    """A tokenizer: the model's list of pieces, and text cut into them."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.size = tokenizer.get_vocab_size(with_added_tokens=True)
        #: The control tokens — for GPT-2 the one, ``<|endoftext|>``.
        self.special = {
            token
            for token, added in tokenizer.get_added_tokens_decoder().items()
            if added.special
        }
        self._pieces: dict[int, str] = {}

    def known(self, token: int) -> bool:
        return 0 <= token < self.size

    def encode(self, text: str, add_special: bool) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special).ids

    def piece(self, token: int) -> str:
        """A token as a panel shows it: on its own, so a character spread over
        two tokens shows as two replacement marks."""
        piece = self._pieces.get(token)
        if piece is None:
            piece = self._tokenizer.decode([token], skip_special_tokens=False)
            self._pieces[token] = piece
        return piece

    def text(self, tokens: list[int]) -> str:
        """What the tokens say together, control tokens left out."""
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def candidates(self, best: list[tuple[int, float]]) -> list[JsonRecord]:
        return [{"token": token, "piece": self.piece(token), "p": p} for token, p in best]


def candidates(best: list[tuple[int, float]], vocabulary: Vocabulary | None) -> list[JsonRecord]:
    """The most likely tokens as a table shows them; without a tokenizer, by
    their ids alone."""
    if vocabulary is not None:
        return vocabulary.candidates(best)
    return [{"token": token, "p": p} for token, p in best]


def load_vocabulary(content: bytes) -> Vocabulary:
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415 — an optional extra
    except ImportError as error:
        raise Problem(
            "tokenizers is not installed — run in hkp-python: uv sync --extra transformer"
        ) from error
    try:
        return Vocabulary(Tokenizer.from_str(content.decode("utf-8")))
    except Exception as error:  # noqa: BLE001 — whatever the library makes of it
        raise Problem(f"the tokenizer could not be read: {error}") from error


@dataclass
class Weights:
    """A GPT-2 shaped model: its tensors by name, and how it is laid out."""

    tensors: dict[str, Any]
    layers: int
    heads: int
    width: int
    positions: int
    vocabulary: int
    epsilon: float

    def get(self, name: str) -> Any:
        tensor = self.tensors.get(name)
        if tensor is None:
            raise Problem(f"the model has no {name}")
        return tensor

    def layer(self, index: int, name: str) -> Any:
        return self.get(f"h.{index}.{name}")

    def norm(self, stream: Any, name: str) -> Any:
        """The stream as a layer reads it: each token's vector shifted and
        scaled to a standard size first, by the model's own ``name``."""
        return torch().nn.functional.layer_norm(
            stream, (self.width,), self.get(f"{name}.weight"), self.get(f"{name}.bias"), self.epsilon
        )


def load_weights(path: Any, config: bytes) -> Weights:
    torch()
    try:
        from safetensors.torch import load_file  # noqa: PLC0415 — an optional extra
    except ImportError as error:
        raise Problem(
            "safetensors is not installed — run in hkp-python: uv sync --extra transformer"
        ) from error
    try:
        said = json.loads(config.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise Problem(f"the model's config is not JSON: {error}") from error
    if not isinstance(said, dict):
        raise Problem("the model's config is not an object")
    try:
        loaded = load_file(str(path))
    except Exception as error:  # noqa: BLE001
        raise Problem(f"the weights are not a safetensors file: {error}") from error
    # Some exports keep everything under `transformer.`.
    tensors = {name.removeprefix("transformer."): tensor.float() for name, tensor in loaded.items()}
    embedding = tensors.get("wte.weight")
    positions = tensors.get("wpe.weight")
    if embedding is None or positions is None or embedding.dim() != 2 or positions.dim() != 2:
        raise Problem("the weights are not a GPT-2 shaped model (no wte.weight and wpe.weight)")
    heads = said.get("n_head")
    if not isinstance(heads, int) or heads < 1:
        raise Problem("the model's config does not say n_head")
    width = int(embedding.shape[1])
    if width % heads:
        raise Problem(f"a width of {width} does not divide into {heads} heads")
    layers = said.get("n_layer")
    if not isinstance(layers, int):
        layers = len({name.split(".")[1] for name in tensors if name.startswith("h.")})
    epsilon = said.get("layer_norm_epsilon")
    return Weights(
        tensors=tensors,
        layers=layers,
        heads=heads,
        width=width,
        positions=int(positions.shape[0]),
        vocabulary=int(embedding.shape[0]),
        epsilon=float(epsilon) if isinstance(epsilon, (int, float)) else 1e-5,
    )


# ── The service they all are ─────────────────────────────────────────────────


def _fetched(status: str, progress: float | None) -> str:
    """How much has arrived, as a panel shows it: "76%" while fetching."""
    if status != "fetching":
        return ""
    return f"{int(progress * 100)}%" if progress is not None else "…"


class ModelService:
    """A service that names what it loads and says where it is with it.

    Fetching and loading run on a thread of the service's own — a first use
    downloads the weights, which takes minutes — and until they are there
    ``process`` passes nothing on and says why.

    State, beside the service's own:
      tokenizer, model, config   the references, whichever the service takes
      status    idle | fetching | loading | ready | error   (read-only)
      progress  0…1 while fetching, when the source says its size (read-only)
      fetched   the same as text, "76%", for a panel to show     (read-only)
      problem   why status is error                          (read-only)
    """

    #: Whether the service runs the model: it names ``model`` and ``config``.
    uses_weights = False
    #: Whether it names a ``tokenizer``.
    uses_vocabulary = False
    #: Whether it works without the tokenizer, saying less.
    vocabulary_optional = False

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._host: RuntimeHost | None = None
        names = (["model", "config"] if self.uses_weights else []) + (
            ["tokenizer"] if self.uses_vocabulary else []
        )
        self._references: dict[str, str] = {name: "" for name in names}
        self._use = threading.Lock()
        self._status = "idle"
        self._problem = ""
        self._progress: float | None = None
        self._weights: ModelLease | None = None
        self._vocabulary: ModelLease | None = None
        #: Which load is the current one. A thread that finds another number
        #: here was overtaken, and what it loaded is let go of.
        self._load = 0
        if config.state:
            self.configure(config.state)

    def set_host(self, host: RuntimeHost) -> None:
        self._host = host
        self._begin()

    def destroy(self) -> None:
        with self._use:
            self._load += 1
            held = self._take()
        for lease in held:
            lease.release()

    def configure(self, config: JsonRecord) -> JsonRecord:
        changed = False
        for name in self._references:
            value = config.get(name)
            if isinstance(value, str) and value.strip() != self._references[name]:
                self._references[name] = value.strip()
                changed = True
        if changed:
            self._begin()
        return self.get_state()

    def get_state(self) -> JsonRecord:
        with self._use:
            state: JsonRecord = {
                **self._references,
                "status": self._status,
                "problem": self._problem,
                "fetched": _fetched(self._status, self._progress),
            }
            if self._status == "fetching" and self._progress is not None:
                state["progress"] = self._progress
        return state

    # ── For the services ─────────────────────────────────────────────────────

    def weights(self) -> Weights:
        """The model, or a Problem saying why there is none yet."""
        with self._use:
            if self._weights is not None:
                return self._weights.value
            raise Problem(self._why_not("model"))

    def vocabulary(self) -> Vocabulary | None:
        """The tokenizer, or a Problem saying why there is none yet. None only
        where the service works without one and none is named."""
        with self._use:
            if self._vocabulary is not None:
                return self._vocabulary.value
            if self.vocabulary_optional and not self._references["tokenizer"]:
                return None
            raise Problem(self._why_not("tokenizer"))

    def fail(self, notify: NotifyCallback, problem: str) -> None:
        """Why this call passes nothing on: reported, logged, nothing returned."""
        notify({"error": problem}, None)
        if self._host is not None:
            self._host.log("error", "service.failed", {"message": problem})
        return None

    # ── Private ──────────────────────────────────────────────────────────────

    def _why_not(self, name: str) -> str:
        if not self._references.get(name):
            return f"no {name} is named"
        if self._status == "error":
            return self._problem
        return f"the {name} is not loaded yet ({self._status})"

    def _take(self) -> list[ModelLease]:
        held = [lease for lease in (self._weights, self._vocabulary) if lease is not None]
        self._weights = None
        self._vocabulary = None
        return held

    def _wanted(self) -> dict[str, str] | None:
        """The references to load, or None while one that is needed is missing."""
        named = dict(self._references)
        for name, reference in named.items():
            if not reference and not (name == "tokenizer" and self.vocabulary_optional):
                return None
        return named

    def _begin(self) -> None:
        with self._use:
            self._load += 1
            load = self._load
            held = self._take()
            wanted = self._wanted()
            if any(self._references.values()) or self._status != "idle":
                self._status, self._problem, self._progress = "idle", "", None
        for lease in held:
            lease.release()
        if self._host is None or wanted is None or not any(wanted.values()):
            return
        self._say(load, "fetching")
        threading.Thread(
            target=self._load_all, args=(load, wanted), name=f"model-{self.uuid}", daemon=True
        ).start()

    def _say(self, load: int, status: str, problem: str = "", progress: float | None = None) -> bool:
        """Sets where the service is, and tells whoever watches. False when
        this load was overtaken, and nothing was said."""
        with self._use:
            if load != self._load:
                return False
            self._status, self._problem, self._progress = status, problem, progress
            said: JsonRecord = {
                "status": status,
                "problem": problem,
                "fetched": _fetched(status, progress),
            }
            if progress is not None:
                said["progress"] = progress
        if self._host is not None:
            try:
                self._host.notify(said, self.uuid)
            except Exception:  # noqa: BLE001 — nobody watching is not a failure
                pass
        return True

    def _store(self, wanted: dict[str, str]) -> Any:
        """The assets, once they know what this service names — or, when they
        never do, as they are, so the reason names the asset."""
        deadline = time.monotonic() + _ASSETS_KNOWN_WITHIN_S
        while True:
            host = self._host
            store = host.assets() if host is not None and hasattr(host, "assets") else None
            if store is not None and all(
                store.descriptor(parse_asset_ref(reference) or "") is not None
                for reference in wanted.values()
                if reference
            ):
                return store
            if time.monotonic() >= deadline:
                if store is None:
                    raise Problem("this runtime keeps no assets")
                return store
            time.sleep(0.02)

    def _load_all(self, load: int, wanted: dict[str, str]) -> None:
        held: list[ModelLease] = []
        try:
            store = self._store(wanted)
            weights = vocabulary = None
            if wanted.get("tokenizer"):
                vocabulary = self._load_vocabulary(store, wanted["tokenizer"])
                held.append(vocabulary)
            if wanted.get("model"):
                weights = self._load_weights(load, store, wanted["model"], wanted["config"])
                held.append(weights)
            with self._use:
                if load != self._load:
                    raise _Overtaken()
                self._weights, self._vocabulary = weights, vocabulary
                held = []
            self._say(load, "ready")
        except _Overtaken:
            pass
        except Problem as problem:
            self._say(load, "error", str(problem))
        except Exception as error:  # noqa: BLE001 — said, never raised on a thread
            self._say(load, "error", f"could not be loaded: {error}")
        finally:
            for lease in held:
                lease.release()

    def _load_vocabulary(self, store: Any, reference: str) -> ModelLease:
        resolution = store.resolve(reference)
        if resolution.asset is None:
            raise Problem(resolution.problem)
        content = resolution.asset.content
        key = ("tokenizer", hashlib.sha256(content).hexdigest())
        return store.models.acquire(key, lambda: load_vocabulary(content))

    def _load_weights(self, load: int, store: Any, model: str, config: str) -> ModelLease:
        described = store.resolve(config)
        if described.asset is None:
            raise Problem(described.problem)
        last_said = 0.0

        def progress(received: int, expected: int | None) -> None:
            nonlocal last_said
            if load != self._load:
                raise _Overtaken("no longer wanted")
            now = time.monotonic()
            if expected and now - last_said >= _PROGRESS_EVERY_S:
                last_said = now
                self._say(load, "fetching", "", min(received / expected, 1.0))

        file = store.resolve_path(model, progress)
        if load != self._load:
            raise _Overtaken()
        if file.path is None:
            raise Problem(file.problem)
        if not self._say(load, "loading"):
            raise _Overtaken()
        content = described.asset.content
        key = ("transformer", file.sha256, hashlib.sha256(content).hexdigest())
        return store.models.acquire(key, lambda: load_weights(file.path, content))


class _Overtaken(Exception):
    """This load is no longer the one the service wants."""
