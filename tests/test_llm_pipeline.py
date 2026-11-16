"""A language model laid out as services: tokenize, token-embed, attention,
mlp, unembed, sample and detokenize.

What needs no model is pinned always: the values the services pass each other,
and what a service says when PyTorch, safetensors or tokenizers is not
installed. What runs a model uses a tiny two-layer GPT-2 built in the test
with fixed weights and stored expected scores, so no model is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from hkp.assets import AssetFileCache, AssetStore, read_assets_payload
from hkp.data import MixedData, TextData
from hkp.runtime import HostedRuntime
from hkp.services import llm_model
from hkp.services.attention import ATTENTION_DESCRIPTOR, AttentionService
from hkp.services.detokenize import DETOKENIZE_DESCRIPTOR, DetokenizeService
from hkp.services.llm_model import (
    INSTALL_HINT,
    most_likely,
    number,
    scores_of,
    stream_of,
    tokens_of,
)
from hkp.services.loop import LOOP_DESCRIPTOR, LoopService
from hkp.services.mlp import MLP_DESCRIPTOR, MlpService
from hkp.services.sample import SAMPLE_DESCRIPTOR, SampleService
from hkp.services.sub_service import SUB_SERVICE_DESCRIPTOR, SubService
from hkp.services.token_embed import TOKEN_EMBED_DESCRIPTOR, TokenEmbedService
from hkp.services.tokenize import TOKENIZE_DESCRIPTOR, TokenizeService
from hkp.services.unembed import UNEMBED_DESCRIPTOR, UnembedService
from hkp.types import RuntimeConfiguration, ServiceConfiguration


# ── Values and configuration (no model needed) ───────────────────────────────


def test_tokens_and_numbers_are_read_only_when_they_are_what_they_claim() -> None:
    assert tokens_of({"tokens": [1, 2, 3]}) == [1, 2, 3]
    assert tokens_of({"tokens": [1, True]}) is None
    assert tokens_of({"tokens": [1, "two"]}) is None
    assert tokens_of([1, 2]) is None

    assert number({"p": 0.95}, "p") == 0.95
    assert number({"p": "0.65"}, "p") == 0.65
    assert number({"p": "wide"}, "p") is None
    assert number({"p": True}, "p") is None
    assert number({"p": -1}, "p", least=0.0) is None


def test_sample_refuses_a_mode_it_does_not_have() -> None:
    sample = SampleService(ServiceConfiguration(service_id="sample", uuid="s"))
    with pytest.raises(ValueError, match="unknown mode"):
        sample.configure({"mode": "nucleus"})
    assert sample.get_state()["mode"] == "pick"

    sample.configure({"mode": "top-k", "k": 5})
    assert sample.get_state()["k"] == 5

    # A facade knob sends its value as text.
    sample.configure({"temperature": "0.65", "k": "12", "p": "wide", "seed": 7})
    state = sample.get_state()
    assert state["temperature"] == 0.65
    assert state["k"] == 12
    assert state["p"] == 0.95
    assert state["seed"] == 7

    sample.configure({"seed": None})
    assert sample.get_state()["seed"] is None


def test_without_the_transformer_extra_each_service_says_why_it_cannot_run(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)

    embed = TokenEmbedService(ServiceConfiguration(service_id="token-embed", uuid="embed"))
    said: list[dict] = []
    assert embed.process({"tokens": [0]}, lambda payload, _id=None: said.append(payload)) is None
    assert said[-1]["error"] == INSTALL_HINT


# ── Tiny GPT-2 fixture ──────────────────────────────────────────────────────

#: Stored expected scores from the tiny two-layer, two-head, width-8 GPT-2
#: below for the prompt `[1, 2, 3]` ("red orange yellow"), computed from the
#: reference GPT-2 equations in float32.
EXPECTED_SCORES = [
    0.16792691,
    -0.05589226,
    0.28067783,
    -0.11176306,
    0.33256307,
    -0.88121158,
    -0.11823995,
    -0.76363623,
]

VOCAB_WORDS = ["<|endoftext|>", "red", "orange", "yellow", "green", "blue", "indigo", "violet"]


def _reference_gpt2(tensors: dict[str, Any], tokens: list[int], layers: int, heads: int, eps: float = 1e-5):
    """The textbook GPT-2 forward pass on its own, to check the services against."""
    import torch
    import torch.nn.functional as F

    width = tensors["wte.weight"].shape[1]
    each = width // heads
    x = tensors["wte.weight"][tokens] + tensors["wpe.weight"][: len(tokens)]
    for i in range(layers):
        h = F.layer_norm(x, (width,), tensors[f"h.{i}.ln_1.weight"], tensors[f"h.{i}.ln_1.bias"], eps)
        qkv = h @ tensors[f"h.{i}.attn.c_attn.weight"] + tensors[f"h.{i}.attn.c_attn.bias"]
        q, k, v = (p.reshape(len(tokens), heads, each).transpose(0, 1) for p in qkv.split(width, dim=-1))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(each)
        mask = torch.triu(torch.ones(len(tokens), len(tokens), dtype=torch.bool), diagonal=1)
        att = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
        y = (att @ v).transpose(0, 1).reshape(len(tokens), width)
        x = x + (y @ tensors[f"h.{i}.attn.c_proj.weight"] + tensors[f"h.{i}.attn.c_proj.bias"])

        h = F.layer_norm(x, (width,), tensors[f"h.{i}.ln_2.weight"], tensors[f"h.{i}.ln_2.bias"], eps)
        m = F.gelu(h @ tensors[f"h.{i}.mlp.c_fc.weight"] + tensors[f"h.{i}.mlp.c_fc.bias"], approximate="tanh")
        x = x + (m @ tensors[f"h.{i}.mlp.c_proj.weight"] + tensors[f"h.{i}.mlp.c_proj.bias"])

    last = F.layer_norm(x[-1], (width,), tensors["ln_f.weight"], tensors["ln_f.bias"], eps)
    return tensors["wte.weight"] @ last


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    torch = pytest.importorskip("torch", reason="needs uv sync --extra transformer")
    safetensors = pytest.importorskip("safetensors.torch", reason="needs uv sync --extra transformer")
    tokenizers = pytest.importorskip("tokenizers", reason="needs uv sync --extra transformer")

    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {word: idx for idx, word in enumerate(VOCAB_WORDS)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens([AddedToken("<|endoftext|>", special=True)])
    tokenizer_json = tok.to_str()

    gen = torch.Generator().manual_seed(17)
    vocab_size = len(VOCAB_WORDS)
    positions = 16
    width = 8
    heads = 2
    layers = 2

    def rand(*shape: int):
        return torch.randn(shape, generator=gen, dtype=torch.float32) * 0.2

    tensors: dict[str, Any] = {
        "wte.weight": rand(vocab_size, width),
        "wpe.weight": rand(positions, width),
        "ln_f.weight": torch.ones(width) + rand(width),
        "ln_f.bias": rand(width),
    }
    for i in range(layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.ones(width) + rand(width)
        tensors[f"h.{i}.ln_1.bias"] = rand(width)
        tensors[f"h.{i}.attn.c_attn.weight"] = rand(width, 3 * width)
        tensors[f"h.{i}.attn.c_attn.bias"] = rand(3 * width)
        tensors[f"h.{i}.attn.c_proj.weight"] = rand(width, width)
        tensors[f"h.{i}.attn.c_proj.bias"] = rand(width)
        tensors[f"h.{i}.ln_2.weight"] = torch.ones(width) + rand(width)
        tensors[f"h.{i}.ln_2.bias"] = rand(width)
        tensors[f"h.{i}.mlp.c_fc.weight"] = rand(width, 4 * width)
        tensors[f"h.{i}.mlp.c_fc.bias"] = rand(4 * width)
        tensors[f"h.{i}.mlp.c_proj.weight"] = rand(4 * width, width)
        tensors[f"h.{i}.mlp.c_proj.bias"] = rand(width)

    folder: Path = tmp_path_factory.mktemp("tiny-gpt2")
    weights_bytes = safetensors.save(tensors)
    weights_sha = hashlib.sha256(weights_bytes).hexdigest()
    (folder / weights_sha).write_bytes(weights_bytes)

    config_json = json.dumps(
        {
            "n_layer": layers,
            "n_head": heads,
            "n_embd": width,
            "n_ctx": positions,
            "layer_norm_epsilon": 1e-5,
        }
    )
    assets = read_assets_payload(
        {
            "weights": {
                "mediaType": "application/octet-stream",
                "url": "https://example.invalid/model.safetensors",
                "sha256": weights_sha,
            },
            "config": {"mediaType": "application/json", "text": config_json},
            "tokenizer": {"mediaType": "application/json", "text": tokenizer_json},
        }
    )
    return {
        "folder": folder,
        "assets": assets,
        "tensors": tensors,
        "layers": layers,
        "heads": heads,
        "width": width,
        "vocab_size": vocab_size,
    }


def _wait_ready(service: Any) -> None:
    for _ in range(400):
        state = service.get_state()
        if state.get("status") == "ready":
            return
        if state.get("status") == "error":
            pytest.fail(f"{service.uuid} failed to load: {state.get('problem')}")
        time.sleep(0.01)
    pytest.fail(f"{service.uuid} did not become ready in time: {service.get_state()}")


def _factory(cls: Any):
    return lambda cfg, _cs=None: cls(cfg)


def _runtime(tiny: dict[str, Any], services: list[ServiceConfiguration]) -> tuple[HostedRuntime, list[tuple[str, dict]]]:
    factories = {
        TOKENIZE_DESCRIPTOR.service_id: _factory(TokenizeService),
        TOKEN_EMBED_DESCRIPTOR.service_id: _factory(TokenEmbedService),
        ATTENTION_DESCRIPTOR.service_id: _factory(AttentionService),
        MLP_DESCRIPTOR.service_id: _factory(MlpService),
        UNEMBED_DESCRIPTOR.service_id: _factory(UnembedService),
        SAMPLE_DESCRIPTOR.service_id: _factory(SampleService),
        DETOKENIZE_DESCRIPTOR.service_id: _factory(DetokenizeService),
        SUB_SERVICE_DESCRIPTOR.service_id: lambda cfg, cs: SubService(cfg, cs),
        LOOP_DESCRIPTOR.service_id: lambda cfg, cs: LoopService(cfg, cs),
    }

    def create_service(cfg: ServiceConfiguration) -> Any:
        return factories[cfg.service_id](cfg, create_service)

    rt = HostedRuntime(
        RuntimeConfiguration(id="py", name="Python", services=[]),
        create_service,
    )
    rt._asset_store = AssetStore(file_cache=AssetFileCache(tiny["folder"]))
    rt._asset_store.replace(tiny["assets"])
    said: list[tuple[str, dict]] = []
    rt.register_notification_target(
        lambda n: said.append((n.instance_id, n.payload))
        if isinstance(n.payload, dict) and not n.payload.get("__internal")
        else None
    )
    for svc in services:
        rt.add_service(svc)
    return rt, said


def _layer_entry(index: int) -> dict[str, Any]:
    """One use of the `layer` block as the board expands it: attention -> mlp."""
    weights_state = {"model": "hkp-asset://weights", "config": "hkp-asset://config", "layer": index}
    return {
        "instanceId": f"layer-{index}",
        "serviceId": "sub-service",
        "serviceName": f"Layer {index}",
        "state": {
            "pipeline": [
                {"instanceId": "attention", "serviceId": "attention", "state": weights_state},
                {"instanceId": "mlp", "serviceId": "mlp", "state": weights_state},
            ]
        },
    }


# ── Model tests ──────────────────────────────────────────────────────────────


def test_scores_and_streams_are_read_both_in_process_and_from_bytes(tiny_model) -> None:
    torch = llm_model.torch()

    tensor_scores = MixedData({"tokens": [1, 2]}, torch.tensor([0.5, -1.0], dtype=torch.float32))
    meta, got = scores_of(tensor_scores)
    assert meta["tokens"] == [1, 2] and got.tolist() == pytest.approx([0.5, -1.0])

    byte_scores = MixedData({"tokens": [1, 2]}, tensor_scores.to_bytes())
    assert scores_of(byte_scores)[1].tolist() == pytest.approx([0.5, -1.0])
    assert scores_of(MixedData({"tokens": "bad"}, b"\x00" * 8)) is None

    stream_tensor = torch.ones((2, 4), dtype=torch.float32)
    stream = MixedData({"tokens": [1, 2], "shape": [2, 4]}, stream_tensor)
    assert list(stream_of(stream)[1].shape) == [2, 4]
    from_bytes = MixedData({"tokens": [1, 2], "shape": [2, 4]}, stream.to_bytes())
    assert list(stream_of(from_bytes)[1].shape) == [2, 4]
    assert stream_of(MixedData({"tokens": [1], "shape": [2, 4]}, stream_tensor)) is None

    best, remaining = most_likely(torch.tensor([1.0, float("-inf"), 3.0, 2.0]), 2)
    assert remaining == 3
    assert [t for t, _ in best] == [2, 3]
    all_best, _ = most_likely(torch.tensor([1.0, float("-inf"), 3.0, 2.0]), 10)
    assert sum(p for _, p in all_best) == pytest.approx(1.0)


def test_every_service_shares_one_loaded_model_and_one_tokenizer(tiny_model) -> None:
    weights_cfg = {"model": "hkp-asset://weights", "config": "hkp-asset://config"}
    tok_cfg = {"tokenizer": "hkp-asset://tokenizer"}
    rt, _ = _runtime(
        tiny_model,
        [
            ServiceConfiguration(service_id="tokenize", uuid="tok", state=tok_cfg),
            ServiceConfiguration(service_id="token-embed", uuid="embed", state=weights_cfg),
            ServiceConfiguration(service_id="attention", uuid="attn", state={**weights_cfg, "layer": 0}),
            ServiceConfiguration(service_id="mlp", uuid="mlp", state={**weights_cfg, "layer": 0}),
            ServiceConfiguration(service_id="unembed", uuid="unembed", state={**weights_cfg, **tok_cfg}),
            ServiceConfiguration(service_id="sample", uuid="sample", state=tok_cfg),
            ServiceConfiguration(service_id="detokenize", uuid="detok", state=tok_cfg),
        ],
    )
    for uuid in ("tok", "embed", "attn", "mlp", "unembed", "sample", "detok"):
        _wait_ready(rt.get_service(uuid))

    keys = rt.assets().models.keys()
    assert len(keys) == 2
    assert {k[0] for k in keys} == {"tokenizer", "transformer"}

    for uuid in ("tok", "embed", "attn", "mlp", "unembed", "sample", "detok"):
        rt.remove_service(uuid)
    assert rt.assets().models.keys() == []


def test_tokenize_cuts_text_and_detokenize_turns_the_new_tokens_back(tiny_model) -> None:
    tok_cfg = {"tokenizer": "hkp-asset://tokenizer"}
    rt, said = _runtime(
        tiny_model,
        [
            ServiceConfiguration(service_id="tokenize", uuid="tok", state=tok_cfg),
            ServiceConfiguration(service_id="detokenize", uuid="detok", state=tok_cfg),
        ],
    )
    tok = rt.get_service("tok")
    detok = rt.get_service("detok")
    _wait_ready(tok)
    _wait_ready(detok)

    tokens = tok.process(TextData("red orange"), lambda p, _i=None: said.append(("tok", p)))
    assert tokens == {"tokens": [1, 2], "pieces": ["red", "orange"], "promptTokens": 2}
    assert said[-1][1] == {"count": 2, "pieces": ["red", "orange"]}

    grown = {**tokens, "tokens": [1, 2, 3, 4]}
    out = detok.process(grown, lambda *_: None)
    assert out["text"] == "yellow green"
    assert out["end"] is False
    assert out["pieces"] == ["red", "orange", "yellow", "green"]

    # Emitting `<|endoftext|>` (token 0) after the prompt marks the turn ended
    # and stays out of `text`.
    ended = detok.process({**tokens, "tokens": [1, 2, 3, 0]}, lambda *_: None)
    assert ended["text"] == "yellow"
    assert ended["end"] is True


def test_the_forward_pass_matches_the_reference_model_and_stored_scores(tiny_model) -> None:
    weights_cfg = {"model": "hkp-asset://weights", "config": "hkp-asset://config"}
    rt, said = _runtime(
        tiny_model,
        [
            ServiceConfiguration(service_id="token-embed", uuid="embed", state=weights_cfg),
            ServiceConfiguration(service_id="attention", uuid="attn-0", state={**weights_cfg, "layer": 0}),
            ServiceConfiguration(service_id="mlp", uuid="mlp-0", state={**weights_cfg, "layer": 0}),
            ServiceConfiguration(service_id="attention", uuid="attn-1", state={**weights_cfg, "layer": 1}),
            ServiceConfiguration(service_id="mlp", uuid="mlp-1", state={**weights_cfg, "layer": 1}),
            ServiceConfiguration(
                service_id="unembed",
                uuid="unembed",
                state={**weights_cfg, "tokenizer": "hkp-asset://tokenizer", "top": 3},
            ),
        ],
    )
    for uuid in ("embed", "attn-0", "mlp-0", "attn-1", "mlp-1", "unembed"):
        _wait_ready(rt.get_service(uuid))

    result = rt.process({"tokens": [1, 2, 3], "pieces": ["red", "orange", "yellow"], "promptTokens": 3}, lambda *_: None)
    meta, scores = scores_of(result)
    assert meta == {"tokens": [1, 2, 3], "pieces": ["red", "orange", "yellow"], "promptTokens": 3, "applied": []}

    expected = _reference_gpt2(
        tiny_model["tensors"], [1, 2, 3], tiny_model["layers"], tiny_model["heads"]
    )
    assert scores.tolist() == pytest.approx(expected.tolist(), abs=1e-5)
    assert scores.tolist() == pytest.approx(EXPECTED_SCORES, abs=1e-5)

    by_id = dict(said)
    assert by_id["embed"]["tokens"] == 3 and by_id["embed"]["width"] == 8
    assert by_id["attn-0"]["layer"] == 0 and by_id["attn-0"]["heads"] == 2
    assert "added" in by_id["mlp-1"] and "size" in by_id["mlp-1"]
    assert len(by_id["unembed"]["candidates"]) == 3
    assert by_id["unembed"]["candidates"][0]["token"] == int(scores.argmax())
    assert by_id["unembed"]["candidates"][0]["piece"] == VOCAB_WORDS[int(scores.argmax())]


def test_each_sampler_narrows_the_candidates_and_the_last_one_chooses(tiny_model) -> None:
    torch = llm_model.torch()
    tok_cfg = {"tokenizer": "hkp-asset://tokenizer"}
    rt, _ = _runtime(
        tiny_model,
        [
            ServiceConfiguration(service_id="sample", uuid="penalty", state={**tok_cfg, "mode": "repetition-penalty", "penalty": 1.5}),
            ServiceConfiguration(service_id="sample", uuid="top-k", state={**tok_cfg, "mode": "top-k", "k": 4}),
            ServiceConfiguration(service_id="sample", uuid="temperature", state={**tok_cfg, "mode": "temperature", "temperature": 0.5}),
            ServiceConfiguration(service_id="sample", uuid="top-p", state={**tok_cfg, "mode": "top-p", "p": 0.7}),
            ServiceConfiguration(service_id="sample", uuid="min-p", state={**tok_cfg, "mode": "min-p", "minP": 0.2}),
            ServiceConfiguration(service_id="sample", uuid="greedy", state={**tok_cfg, "mode": "greedy"}),
            ServiceConfiguration(service_id="sample", uuid="pick-a", state={**tok_cfg, "mode": "pick", "seed": 11}),
            ServiceConfiguration(service_id="sample", uuid="pick-b", state={**tok_cfg, "mode": "pick", "seed": 11}),
        ],
    )
    for uuid in ("penalty", "top-k", "temperature", "top-p", "min-p", "greedy", "pick-a", "pick-b"):
        _wait_ready(rt.get_service(uuid))

    raw = torch.tensor(EXPECTED_SCORES, dtype=torch.float32)
    winner = int(raw.argmax())
    initial = MixedData({"tokens": [1, 2], "pieces": ["red", "orange"], "promptTokens": 2, "applied": []}, raw)

    # Repetition penalty lowers a token that was already said.
    with_winner = MixedData({"tokens": [1, winner], "applied": []}, raw)
    penalised = scores_of(rt.get_service("penalty").process(with_winner, lambda *_: None))[1]
    assert penalised[winner] < raw[winner]

    notes: list[dict] = []
    after_k = rt.get_service("top-k").process(initial, lambda p, _i=None: notes.append(p))
    assert notes[-1]["remaining"] == 4
    before_p = notes[-1]["candidates"][0]["p"]

    after_t = rt.get_service("temperature").process(after_k, lambda p, _i=None: notes.append(p))
    assert notes[-1]["candidates"][0]["p"] > before_p

    after_p = rt.get_service("top-p").process(after_t, lambda p, _i=None: notes.append(p))
    assert 1 <= notes[-1]["remaining"] <= 4

    after_min_p = rt.get_service("min-p").process(after_p, lambda p, _i=None: notes.append(p))
    meta, _ = scores_of(after_min_p)
    assert meta["applied"] == ["top-k 4", "temperature 0.5", "top-p 0.7", "min-p 0.2"]

    chosen = rt.get_service("greedy").process(after_min_p, lambda p, _i=None: notes.append(p))
    assert chosen == {
        "tokens": [1, 2, winner],
        "pieces": ["red", "orange", VOCAB_WORDS[winner]],
        "promptTokens": 2,
    }
    assert notes[-1]["chosen"]["token"] == winner

    # The same seed picks the same sequence of tokens.
    picks_a = [rt.get_service("pick-a").process(initial, lambda *_: None)["tokens"][-1] for _ in range(4)]
    picks_b = [rt.get_service("pick-b").process(initial, lambda *_: None)["tokens"][-1] for _ in range(4)]
    assert picks_a == picks_b


def test_a_loop_of_layers_steps_and_runs_with_layer_blocks(tiny_model) -> None:
    weights_cfg = {"model": "hkp-asset://weights", "config": "hkp-asset://config"}
    tok_cfg = {"tokenizer": "hkp-asset://tokenizer"}
    rt, said = _runtime(
        tiny_model,
        [
            ServiceConfiguration(service_id="tokenize", uuid="tokenize-svc", state=tok_cfg),
            ServiceConfiguration(
                service_id="loop",
                uuid="generate-svc",
                state={
                    "until": "params.end",
                    "maxRounds": 4,
                    "mode": "step",
                    "pipeline": [
                        {"instanceId": "embed", "serviceId": "token-embed", "state": weights_cfg},
                        _layer_entry(0),
                        _layer_entry(1),
                        {"instanceId": "unembed", "serviceId": "unembed", "state": {**weights_cfg, **tok_cfg}},
                        {"instanceId": "greedy", "serviceId": "sample", "state": {**tok_cfg, "mode": "greedy"}},
                        {"instanceId": "piece", "serviceId": "detokenize", "state": tok_cfg},
                    ],
                },
            ),
        ],
    )
    _wait_ready(rt.get_service("tokenize-svc"))
    loop = rt.get_service("generate-svc")
    for inner in ("embed", "unembed", "greedy", "piece"):
        _wait_ready(loop.find_nested(inner))
    for i in (0, 1):
        layer_svc = loop.find_nested(f"layer-{i}")
        _wait_ready(layer_svc.find_nested("attention"))
        _wait_ready(layer_svc.find_nested("mlp"))

    # Round 1: step from the prompt.
    first = rt.process("red orange", lambda *_: None)
    assert first["promptTokens"] == 2
    assert len(first["tokens"]) == 3
    assert first["tokens"][-1] == 2  # argmax of EXPECTED_SCORES is token 2 ("orange")

    # Round 2: a call with nothing on the loop steps one more token.
    second = loop.process(None, lambda *_: None)
    assert len(second["tokens"]) == 4

    # Then run to the limit.
    loop.configure({"mode": "run"})
    final = loop.process(None, lambda *_: None)
    assert len(final["tokens"]) == 6
    assert loop.get_state()["done"] is True
    assert loop.get_state()["reason"] == "maxRounds"
    assert any(instance == "generate-svc.layer-0.attention" for instance, _ in said)


def test_services_refuse_out_of_range_tokens_and_layers_and_work_without_optional_tokenizer(
    tiny_model,
) -> None:
    weights_cfg = {"model": "hkp-asset://weights", "config": "hkp-asset://config"}
    rt, _ = _runtime(
        tiny_model,
        [
            ServiceConfiguration(service_id="token-embed", uuid="embed", state=weights_cfg),
            ServiceConfiguration(service_id="attention", uuid="bad-layer", state={**weights_cfg, "layer": 99}),
            ServiceConfiguration(service_id="unembed", uuid="unembed-notok", state=weights_cfg),
            ServiceConfiguration(service_id="sample", uuid="greedy-notok", state={"mode": "greedy"}),
        ],
    )
    for uuid in ("embed", "bad-layer", "unembed-notok"):
        _wait_ready(rt.get_service(uuid))

    errors: list[dict] = []
    assert (
        rt.get_service("embed").process({"tokens": [999]}, lambda p, _i=None: errors.append(p))
        is None
    )
    assert "not in this model's vocabulary" in errors[-1]["error"]

    stream = rt.get_service("embed").process({"tokens": [1, 2]}, lambda *_: None)
    assert (
        rt.get_service("bad-layer").process(stream, lambda p, _i=None: errors.append(p)) is None
    )
    assert "has no layer 99" in errors[-1]["error"]

    # Unembed and Sample work without a tokenizer, reporting token ids alone.
    notes: list[dict] = []
    scores = rt.get_service("unembed-notok").process(stream, lambda p, _i=None: notes.append(p))
    assert "piece" not in notes[-1]["candidates"][0]
    chosen = rt.get_service("greedy-notok").process(scores, lambda p, _i=None: notes.append(p))
    assert len(chosen["tokens"]) == 3
    assert "piece" not in notes[-1]["chosen"]

