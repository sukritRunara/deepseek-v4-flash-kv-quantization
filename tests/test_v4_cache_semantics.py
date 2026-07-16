"""Deterministic cache-semantics tests for DeepSeek-V4 (tiny random model, CPU, no weights).

Configuration (src/v4_kv_quant/tiny_model.py) covers all three V4 attention layer
types: sliding_attention / compressed_sparse_attention (m=4, indexer) /
heavily_compressed_attention (m'=8). Boundary values are read from the config —
never hardcoded.

Two model fixtures:

* ``model_selective`` — production-like ``index_topk=2``: used for structural
  tests (counts, shapes, storage sharing, boundaries, determinism).
* ``model_dense`` — ``index_topk=64`` (>= any entry count reachable here): used
  for numerical-equality tests between computation paths.

Why the split (measured, see docs/V4_CACHE_ARCHITECTURE.md §6.8): with a
*selective* indexer, batched prefill and token-by-token decode produce index
scores that differ by ~1e-7 float noise; when two candidate entries are
near-tied (common with random weights), the selected top-k SET flips and the
logits legitimately diverge (observed up to ~1e-1). With a non-selective
indexer the same paths agree to ~2e-7. Equality across paths is therefore only
defined up to top-k tie-breaking — a property that quantization experiments
must measure via top-k overlap metrics rather than logit closeness.

Observed numerical baselines on this machine (fp32 CPU, tiny model):
  use_cache=True vs False, same one-shot prefill : bit-exact (0.0)
  full forward vs token-by-token decode (dense)  : max |d| 2.4e-7
  one-shot vs chunked prefill (dense)            : max |d| 2.2e-7
Tolerances below use ~40x margin over those measurements.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)

from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids

ATOL = 1e-5
RTOL = 1e-5
BATCH = 2

CSA = "compressed_sparse_attention"
HCA = "heavily_compressed_attention"
SLIDING = "sliding_attention"


@pytest.fixture(scope="module")
def model_selective():
    return build_tiny_model(seed=0)


@pytest.fixture(scope="module")
def model_dense():
    return build_tiny_model(seed=0, index_topk=64)


def full_forward_logits(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(ids, use_cache=False).logits


def decode_token_by_token(model, ids: torch.Tensor, cache=None) -> tuple[torch.Tensor, DynamicCache]:
    """Teacher-forced single-token cached decode over all of ``ids``."""
    cache = cache if cache is not None else DynamicCache(config=model.config)
    steps = []
    with torch.no_grad():
        for t in range(ids.shape[1]):
            out = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            steps.append(out.logits[:, 0])
    return torch.stack(steps, dim=1), cache


def chunked_prefill(model, ids: torch.Tensor, chunk_sizes: list[int]) -> tuple[torch.Tensor, DynamicCache]:
    assert sum(chunk_sizes) == ids.shape[1]
    cache = DynamicCache(config=model.config)
    outs, pos = [], 0
    with torch.no_grad():
        for size in chunk_sizes:
            outs.append(model(ids[:, pos : pos + size], past_key_values=cache, use_cache=True).logits)
            pos += size
    return torch.cat(outs, dim=1), cache


def layer_index(model, layer_type: str) -> int:
    return model.config.layer_types.index(layer_type)


def assert_cache_states_close(cache_a, cache_b, config) -> None:
    """Full state-by-state comparison of two caches built over the same tokens."""
    for i, (a, b) in enumerate(zip(cache_a.layers, cache_b.layers)):
        assert type(a) is type(b)
        assert a.cumulative_length == b.cumulative_length, f"layer {i} cumulative_length"
        torch.testing.assert_close(a.keys, b.keys, atol=ATOL, rtol=RTOL, msg=f"layer {i} keys")
        if not hasattr(a, "compressed_kv"):
            continue
        assert a.entry_count == b.entry_count, f"layer {i} entry_count"
        for name in a.compressed_kv:
            torch.testing.assert_close(
                a.compressed_kv[name], b.compressed_kv[name], atol=ATOL, rtol=RTOL,
                msg=f"layer {i} compressed_kv[{name}]",
            )
            torch.testing.assert_close(
                a.buffer_kv[name], b.buffer_kv[name], atol=ATOL, rtol=RTOL,
                msg=f"layer {i} buffer_kv[{name}]",
            )
            torch.testing.assert_close(
                a.buffer_gate[name], b.buffer_gate[name], atol=ATOL, rtol=RTOL,
                msg=f"layer {i} buffer_gate[{name}]",
            )
        if hasattr(a, "overlap_kv"):
            for name in a.overlap_kv:
                if a.overlap_kv[name] is None:
                    assert b.overlap_kv[name] is None, f"layer {i} overlap_kv[{name}]"
                else:
                    torch.testing.assert_close(
                        a.overlap_kv[name], b.overlap_kv[name], atol=ATOL, rtol=RTOL,
                        msg=f"layer {i} overlap_kv[{name}]",
                    )


# ---------------------------------------------------------------------------
# Path-equality tests (dense indexer: equality is well-defined)
# ---------------------------------------------------------------------------


def test_full_forward_vs_token_by_token_decode(model_dense):
    ids = deterministic_input_ids(BATCH, 27)
    full = full_forward_logits(model_dense, ids)
    stepped, cache = decode_token_by_token(model_dense, ids)
    torch.testing.assert_close(stepped, full, atol=ATOL, rtol=RTOL)
    assert cache.layers[0].cumulative_length == 27


@pytest.mark.parametrize(
    "chunks",
    [
        [13, 14],          # boundary not aligned to any rate
        [4, 8, 15],        # boundaries aligned to CSA rate
        [5, 1, 7, 8, 6],   # single-token chunk mid-prefill
        [1] * 9 + [18],    # decode-then-large-chunk
    ],
)
def test_one_shot_vs_chunked_prefill(model_dense, chunks):
    ids = deterministic_input_ids(BATCH, sum(chunks))
    with torch.no_grad():
        out = model_dense(ids, use_cache=True)
    one_shot_logits, one_shot_cache = out.logits, out.past_key_values
    chunked_logits, chunked_cache = chunked_prefill(model_dense, ids, chunks)
    torch.testing.assert_close(chunked_logits, one_shot_logits, atol=ATOL, rtol=RTOL)
    assert_cache_states_close(chunked_cache, one_shot_cache, model_dense.config)


def test_prefill_decode_prefill(model_dense):
    """prefill 9 -> decode 4 -> prefill chunk 8 -> decode 2, vs one-shot full forward."""
    ids = deterministic_input_ids(BATCH, 23)
    full = full_forward_logits(model_dense, ids)
    cache = DynamicCache(config=model_dense.config)
    logits = []
    with torch.no_grad():
        logits.append(model_dense(ids[:, :9], past_key_values=cache, use_cache=True).logits)
        for t in range(9, 13):
            logits.append(model_dense(ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits)
        logits.append(model_dense(ids[:, 13:21], past_key_values=cache, use_cache=True).logits)
        for t in range(21, 23):
            logits.append(model_dense(ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits)
    torch.testing.assert_close(torch.cat(logits, dim=1), full, atol=ATOL, rtol=RTOL)
    assert cache.layers[0].cumulative_length == 23


def test_cache_reset_reuse(model_dense):
    """Fresh caches are bit-identical; a consumed cache continues exactly like one-shot."""
    ids = deterministic_input_ids(BATCH, 16)
    stepped_a, cache_a = decode_token_by_token(model_dense, ids)
    stepped_b, cache_b = decode_token_by_token(model_dense, ids)
    assert torch.equal(stepped_a, stepped_b)  # identical paths are bit-exact
    assert_cache_states_close(cache_a, cache_b, model_dense.config)

    # reuse: continue decoding on cache_a; must match the full-forward suffix
    more = deterministic_input_ids(BATCH, 6, seed=7)
    all_ids = torch.cat([ids, more], dim=1)
    full = full_forward_logits(model_dense, all_ids)
    continued, _ = decode_token_by_token(model_dense, more, cache=cache_a)
    torch.testing.assert_close(continued, full[:, 16:], atol=ATOL, rtol=RTOL)


def test_sliding_window_rollover(model_dense):
    """Decode across the window boundary: keys stay bounded, numerics match full forward."""
    window = model_dense.config.sliding_window
    total = window + 6
    ids = deterministic_input_ids(BATCH, total)
    full = full_forward_logits(model_dense, ids)

    cache = DynamicCache(config=model_dense.config)
    with torch.no_grad():
        model_dense(ids[:, : window - 2], past_key_values=cache, use_cache=True)
    for t in range(window - 2, total):
        with torch.no_grad():
            out = model_dense(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        expected_len = min(t + 1, window - 1)  # update() keeps the last window-1 entries
        for layer in cache.layers:
            assert layer.keys.shape[-2] == expected_len
            assert layer.cumulative_length == t + 1
        torch.testing.assert_close(out.logits[:, 0], full[:, t], atol=ATOL, rtol=RTOL)


def test_use_cache_false_no_mutation(model_dense):
    """use_cache=False returns no cache, is bit-identical to the cached prefill, and stateless."""
    ids = deterministic_input_ids(BATCH, 21)
    with torch.no_grad():
        out_nc = model_dense(ids, use_cache=False)
        out_c = model_dense(ids, use_cache=True)
        out_nc2 = model_dense(ids, use_cache=False)
    assert out_nc.past_key_values is None
    assert out_c.past_key_values is not None
    assert torch.equal(out_nc.logits, out_c.logits)  # cache creation must not change math
    assert torch.equal(out_nc.logits, out_nc2.logits)  # no hidden state left behind


# ---------------------------------------------------------------------------
# Structural tests (selective indexer = production-like config)
# ---------------------------------------------------------------------------


def test_cache_layer_dispatch(model_selective):
    ids = deterministic_input_ids(BATCH, 9)
    with torch.no_grad():
        cache = model_selective(ids, use_cache=True).past_key_values
    types = [type(layer).__name__ for layer in cache.layers]
    expected = {
        SLIDING: "DynamicSlidingWindowLayer",
        CSA: "DeepseekV4CSACache",
        HCA: "DeepseekV4HCACache",
    }
    assert types == [expected[t] for t in model_selective.config.layer_types]
    assert isinstance(cache.layers[layer_index(model_selective, CSA)], DeepseekV4CSACache)
    assert isinstance(cache.layers[layer_index(model_selective, HCA)], DeepseekV4HCACache)
    # CSA cache extends HCA cache with indexer + overlap state
    csa_layer = cache.layers[layer_index(model_selective, CSA)]
    hca_layer = cache.layers[layer_index(model_selective, HCA)]
    assert set(csa_layer.compressed_kv) == {"compressor", "indexer"}
    assert set(hca_layer.compressed_kv) == {"compressor"}
    assert not hasattr(hca_layer, "overlap_kv")
    # module structure mirrors it: HCA compressor has no indexer
    csa_attn = model_selective.model.layers[layer_index(model_selective, CSA)].self_attn
    hca_attn = model_selective.model.layers[layer_index(model_selective, HCA)].self_attn
    assert hasattr(csa_attn.compressor, "indexer")
    assert not hasattr(hca_attn.compressor, "indexer")
    sliding_attn = model_selective.model.layers[layer_index(model_selective, SLIDING)].self_attn
    assert sliding_attn.compressor is None


@pytest.mark.parametrize("seq_len", [3, 4, 5, 7, 8, 9, 15, 16, 17, 21])
def test_exact_entry_counts_and_dims(model_selective, seq_len):
    """Entry counts / buffer lengths / state widths at and around every boundary."""
    config = model_selective.config
    ids = deterministic_input_ids(BATCH, seq_len)
    with torch.no_grad():
        cache = model_selective(ids, use_cache=True).past_key_values

    window = config.sliding_window
    head_dim = config.head_dim
    for i, layer in enumerate(cache.layers):
        layer_type = config.layer_types[i]
        assert layer.cumulative_length == seq_len
        assert layer.get_seq_length() == seq_len
        assert layer.keys.shape == (BATCH, config.num_key_value_heads, min(seq_len, window - 1), head_dim)
        if layer_type == SLIDING:
            continue
        rate = config.compress_rates[layer_type]
        n_entries, n_buffered = seq_len // rate, seq_len % rate
        series = 2 if layer_type == CSA else 1  # CSA projects Ca/Cb -> 2*dim per token
        for name in layer.compressed_kv:
            dim = config.index_head_dim if name == "indexer" else head_dim
            assert layer.entry_count[name] == n_entries
            assert layer.compressed_kv[name].shape == (BATCH, n_entries, dim)
            assert layer.buffer_kv[name].shape == (BATCH, n_buffered, series * dim)
            assert layer.buffer_gate[name].shape == (BATCH, n_buffered, series * dim)
        if layer_type == CSA:
            for name in layer.overlap_kv:
                dim = config.index_head_dim if name == "indexer" else head_dim
                if n_entries == 0:  # no window closed yet -> no Ca slice saved
                    assert layer.overlap_kv[name] is None
                else:
                    assert layer.overlap_kv[name].shape == (BATCH, rate, dim)
                    assert layer.overlap_gate[name].shape == (BATCH, rate, dim)


@pytest.mark.parametrize("layer_type", [CSA, HCA])
def test_decode_crosses_compression_boundary(model_selective, layer_type):
    """One decode step landing exactly on a window boundary emits exactly one entry."""
    config = model_selective.config
    rate = config.compress_rates[layer_type]
    idx = layer_index(model_selective, layer_type)
    prefill = 2 * rate - 1  # one token short of the second boundary

    ids = deterministic_input_ids(BATCH, prefill + 2)
    cache = DynamicCache(config=model_selective.config)
    with torch.no_grad():
        model_selective(ids[:, :prefill], past_key_values=cache, use_cache=True)
    layer = cache.layers[idx]
    assert layer.entry_count["compressor"] == 1
    assert layer.buffer_kv["compressor"].shape[1] == rate - 1

    # boundary step: buffer drains, one entry emitted
    with torch.no_grad():
        model_selective(ids[:, prefill : prefill + 1], past_key_values=cache, use_cache=True)
    assert layer.entry_count["compressor"] == 2
    assert layer.compressed_kv["compressor"].shape[1] == 2
    assert layer.buffer_kv["compressor"].shape[1] == 0

    # step after boundary: buffer regrows, no new entry
    with torch.no_grad():
        model_selective(ids[:, prefill + 1 : prefill + 2], past_key_values=cache, use_cache=True)
    assert layer.entry_count["compressor"] == 2
    assert layer.buffer_kv["compressor"].shape[1] == 1


def test_shared_kv_storage(model_selective):
    """K and V share storage on CSA/HCA layers; stock sliding layer keeps equal copies."""
    config = model_selective.config
    ids = deterministic_input_ids(BATCH, 11)
    with torch.no_grad():
        cache = model_selective(ids, use_cache=True).past_key_values
    for i, layer in enumerate(cache.layers):
        if config.layer_types[i] == SLIDING:
            assert layer.values.data_ptr() != layer.keys.data_ptr()
            assert torch.equal(layer.values, layer.keys)
        else:
            assert layer.values.data_ptr() == layer.keys.data_ptr()


def test_rope_nope_ordering_runtime(model_selective):
    """Runtime proof that keys are laid out [nope | rope] with rope = trailing qk_rope_head_dim.

    The window cache stores kv AFTER kv_norm and AFTER partial RoPE. Capturing the
    kv_norm output (pre-RoPE) and comparing with the cached keys shows:
      * leading head_dim - rope_dim channels are bit-identical (never rotated);
      * trailing rope_dim channels are identical at position 0 (rotation by angle 0)
        and different at every position >= 1 (non-trivial rotation).
    """
    config = model_selective.config
    rd = config.qk_rope_head_dim
    assert rd == int(config.head_dim * config.partial_rotary_factor)
    seq_len = config.sliding_window - 2  # all positions still in window, in order
    ids = deterministic_input_ids(BATCH, seq_len)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, layer in enumerate(model_selective.model.layers):
        hooks.append(
            layer.self_attn.kv_norm.register_forward_hook(
                lambda mod, args, out, i=i: captured.setdefault(i, out.detach().clone())
            )
        )
    try:
        with torch.no_grad():
            cache = model_selective(ids, use_cache=True).past_key_values
    finally:
        for h in hooks:
            h.remove()

    for i, layer in enumerate(cache.layers):
        pre_rope = captured[i]  # [B, S, head_dim]
        keys = layer.keys[:, 0]  # [B, S, head_dim]
        assert torch.equal(keys[..., : -rd], pre_rope[..., : -rd]), f"layer {i}: nope slice was modified"
        assert torch.allclose(keys[:, 0, -rd:], pre_rope[:, 0, -rd:], atol=1e-6), f"layer {i}: pos 0 rope != identity"
        for t in range(1, seq_len):
            assert not torch.allclose(keys[:, t, -rd:], pre_rope[:, t, -rd:], atol=1e-6), (
                f"layer {i}: rope slice unchanged at position {t}"
            )


def test_indexer_topk_causal_validity(model_selective):
    """Indexer picks are always past-only: index i valid iff i < (pos+1)//m, else -1."""
    config = model_selective.config
    csa_idx = layer_index(model_selective, CSA)
    rate = config.compress_rates[CSA]
    indexer = model_selective.model.layers[csa_idx].self_attn.compressor.indexer

    picks: list[torch.Tensor] = []
    hook = indexer.register_forward_hook(lambda mod, args, out: picks.append(out.detach().clone()))
    prefill = 21
    try:
        with torch.no_grad():
            out = model_selective(deterministic_input_ids(BATCH, prefill), use_cache=True)
            cache = out.past_key_values
            for t in range(prefill, prefill + 5):
                tok = deterministic_input_ids(BATCH, 1, seed=100 + t)
                model_selective(tok, past_key_values=cache, use_cache=True)
    finally:
        hook.remove()

    positions_seen = 0
    for chunk in picks:
        seq_len = chunk.shape[1]
        for s in range(seq_len):
            abs_pos = positions_seen + s
            allowed = (abs_pos + 1) // rate
            selected = chunk[:, s]
            valid = selected[selected >= 0]
            assert (valid < allowed).all(), f"pos {abs_pos}: pick >= causal threshold {allowed}"
            if allowed >= config.index_topk:
                assert (selected >= 0).all(), f"pos {abs_pos}: unexpected -1 with {allowed} entries available"
        positions_seen += seq_len
    assert positions_seen == prefill + 5


def test_deterministic_across_runs(model_selective):
    """Same seed -> same weights; same input -> bit-identical logits, twice over."""
    ids = deterministic_input_ids(BATCH, 19)
    first = full_forward_logits(model_selective, ids)
    second = full_forward_logits(model_selective, ids)
    assert torch.equal(first, second)
    rebuilt = build_tiny_model(seed=0)
    third = full_forward_logits(rebuilt, ids)
    assert torch.equal(first, third)
    assert torch.isfinite(first).all()


def test_yarn_compress_rope_variant():
    """The real V4-Flash uses YaRN on the compress rope; verify that path also runs."""
    model = build_tiny_model(
        seed=0,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        max_position_embeddings=256,
    )
    assert model.config.rope_parameters["compress"]["rope_type"] == "yarn"
    assert model.config.rope_parameters["main"]["rope_type"] == "default"
    ids = deterministic_input_ids(BATCH, 17)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    assert torch.isfinite(out.logits).all()
    csa_layer = out.past_key_values.layers[model.config.layer_types.index(CSA)]
    assert csa_layer.entry_count["compressor"] == 17 // model.config.compress_rates[CSA]
