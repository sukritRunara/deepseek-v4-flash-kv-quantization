"""Task 02 acceptance tests: official-policy QDQ simulation (Stage B).

Everything here is simulation — BF16/FP32 storage, no memory savings. Gates
(prompts/02_OFFICIAL_POLICY_QDQ_SIMULATION.md):

  1. identity policy => bit-exact vs stock DynamicCache (logits AND states);
  2. RoPE slice bit-untouched under every policy;
  3. QDQ exactly once per value (history bitwise-stable across appends/decode);
  4. chunked / token-by-token == one-shot under QDQ (dense indexer, Task-01 tolerances);
  5. Task-01 count/shape invariants preserved under QDQ;
  6. metrics harness runs end to end;
  7. production classes untouched (registry not hijacked).
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache
from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
    DeepseekV4IndexerScorer,
)

from v4_kv_quant.harness import run_teacher_forced
from v4_kv_quant.metrics import indexer_topk_overlap, logit_comparison_metrics, next_token_nll
from v4_kv_quant.policy import (
    NAMED_POLICIES,
    KVQuantPolicy,
    baseline_bf16,
    indexer_reference_qdq,
    main_fp8_nonrope_rope_bf16,
    reference_official_qdq,
)
from v4_kv_quant.qdq import (
    ceil_pow2,
    effective_group_size,
    fp4_e2m1_qdq,
    fp8_e4m3_qdq,
    hadamard_transform,
)
from v4_kv_quant.qdq_cache import QDQCache, build_qdq_cache, indexer_query_qdq
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids, tiny_v4_config

ATOL = 1e-5
RTOL = 1e-5
BATCH = 2
CSA = "compressed_sparse_attention"


@pytest.fixture(scope="module")
def model_selective():
    return build_tiny_model(seed=0)


@pytest.fixture(scope="module")
def model_dense():
    return build_tiny_model(seed=0, index_topk=64)


# ---------------------------------------------------------------------------
# QDQ primitive unit tests (production widths and group sizes)
# ---------------------------------------------------------------------------


def test_ceil_pow2_exact():
    x = torch.tensor([0.5, 1.0, 1.0000001, 1.5, 2.0, 3.0, 448.0, 1e-4, 2.0**-20])
    out = ceil_pow2(x)
    expected = torch.tensor([0.5, 1.0, 2.0, 2.0, 2.0, 4.0, 512.0, 2.0**-13, 2.0**-20])
    assert torch.equal(out, expected)
    # every output is a power of two and >= input
    mantissa, _ = torch.frexp(out)
    assert (mantissa == 0.5).all()
    assert (out >= x).all()


def test_fp8_representable_roundtrip():
    # values that are exactly e4m3-grid * power-of-2 scale must round-trip bit-exactly
    x = torch.ones(4, 448)
    y = fp8_e4m3_qdq(x, group_size=64)
    assert torch.equal(y, x)
    y_no_pow2 = fp8_e4m3_qdq(x, group_size=64, pow2_scale=False)
    assert torch.equal(y_no_pow2, x)


def test_fp8_scale_properties():
    torch.manual_seed(3)
    x = torch.randn(16, 448) * 10
    y, scales = fp8_e4m3_qdq(x, group_size=64, return_scales=True)
    assert scales.shape == (16, 7)
    mantissa, _ = torch.frexp(scales)
    assert (mantissa == 0.5).all(), "ue8m0 scales must be powers of two"
    grouped = x.unflatten(-1, (7, 64))
    amax = grouped.abs().amax(-1)
    assert (scales >= amax / 448.0 - 1e-12).all(), "round-up scale must cover amax"
    assert (scales <= (amax / 448.0).clamp_min(1e-4 / 448.0) * 2 + 1e-12).all()
    # error bound: e4m3 has 3 mantissa bits -> RNE relative error <= 2^-4 (+ scale quantum)
    err = (x - y).abs()
    bound = x.abs() * 2.0**-4 + scales.repeat_interleave(64, dim=-1) * 0.25
    assert (err <= bound).all()


def test_fp8_idempotent_and_group_independent():
    torch.manual_seed(4)
    x = torch.randn(8, 448)
    y = fp8_e4m3_qdq(x, group_size=64)
    assert torch.equal(fp8_e4m3_qdq(y, group_size=64), y), "QDQ must be idempotent"
    # perturbing one group must not change any other group's output
    x2 = x.clone()
    x2[:, :64] *= 5.0
    y2 = fp8_e4m3_qdq(x2, group_size=64)
    assert torch.equal(y2[:, 64:], y[:, 64:])
    assert not torch.equal(y2[:, :64], y[:, :64])


def test_fp8_amax_floor_and_empty():
    tiny = torch.full((2, 64), 1e-7)
    y = fp8_e4m3_qdq(tiny, group_size=64)
    assert torch.isfinite(y).all()
    empty = torch.empty(2, 0, 64)
    assert fp8_e4m3_qdq(empty, group_size=64).shape == empty.shape


def test_fp8_bf16_pipeline():
    torch.manual_seed(5)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    y = fp8_e4m3_qdq(x, group_size=64)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


def test_fp4_grid_roundtrip_and_scale():
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for scale in (1.0, 4.0, 0.25):
        x = torch.cat([grid, -grid]).unsqueeze(0) * scale  # amax = 6*scale -> scale exact
        y, scales = fp4_e2m1_qdq(x, group_size=16, return_scales=True)
        assert torch.equal(y, x)
        assert torch.equal(scales, torch.full_like(scales, scale))


def test_fp4_tie_rounding_is_nearest_even():
    # amax=6 forces scale=1; midpoints must round to the even-mantissa neighbor
    x = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]])
    y = fp4_e2m1_qdq(x, group_size=8)
    assert torch.equal(y, torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]]))
    y_neg = fp4_e2m1_qdq(-x, group_size=8)
    assert torch.equal(y_neg, -torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]]))


def test_fp4_official_width():
    torch.manual_seed(6)
    x = torch.randn(8, 128) * 3
    y, scales = fp4_e2m1_qdq(x, group_size=32, return_scales=True)
    assert scales.shape == (8, 4)
    mantissa, _ = torch.frexp(scales)
    assert (mantissa == 0.5).all()
    assert (y.abs() <= scales.repeat_interleave(32, dim=-1) * 6.0 + 1e-6).all()
    assert torch.equal(fp4_e2m1_qdq(y, group_size=32), y)


def test_hadamard_properties():
    torch.manual_seed(7)
    x = torch.randn(5, 3, 128)
    h = hadamard_transform(x)
    torch.testing.assert_close(hadamard_transform(h), x, atol=1e-5, rtol=1e-5)  # involution
    torch.testing.assert_close(
        h.square().sum(-1), x.square().sum(-1), atol=1e-4, rtol=1e-5
    )  # orthonormal
    q, k = torch.randn(4, 16), torch.randn(4, 16)
    torch.testing.assert_close(  # rotation preserves dot products (why rotated-basis storage works)
        (hadamard_transform(q) * hadamard_transform(k)).sum(-1), (q * k).sum(-1), atol=1e-4, rtol=1e-5
    )
    # explicit H4 (Sylvester, orthonormal)
    eye_h = hadamard_transform(torch.eye(4))
    expected = torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float32
    ).T / 2.0
    torch.testing.assert_close(eye_h, expected, atol=1e-6, rtol=0)


def test_effective_group_size():
    assert effective_group_size(448, 64) == 64
    assert effective_group_size(24, 64) == 24  # tiny-model nope width: one group
    assert effective_group_size(16, 32) == 16  # tiny-model indexer width
    assert effective_group_size(128, 32) == 32
    with pytest.raises(ValueError):
        effective_group_size(100, 64)  # never silently fall back


# ---------------------------------------------------------------------------
# Layer-level tests (synthetic tensors, no model forward)
# ---------------------------------------------------------------------------


def _layer_pair(layer_cls_stock, layer_cls_qdq, policy):
    config = tiny_v4_config()
    return config, layer_cls_stock(config), layer_cls_qdq(config, policy)


def test_layer_window_write_qdq_and_rope_untouched():
    from v4_kv_quant.qdq_cache import QDQHCACacheLayer

    config, stock, qdq = _layer_pair(DeepseekV4HCACache, QDQHCACacheLayer, main_fp8_nonrope_rope_bf16())
    rd = config.qk_rope_head_dim
    torch.manual_seed(8)
    kv = torch.randn(BATCH, 1, 5, config.head_dim)
    stock.update(kv.clone(), kv.clone())
    qdq.update(kv.clone(), kv.clone())
    assert torch.equal(qdq.keys[..., -rd:], stock.keys[..., -rd:]), "RoPE slice modified"
    group = effective_group_size(config.head_dim - rd, 64)
    expected_nope = fp8_e4m3_qdq(stock.keys[..., :-rd], group_size=group)
    assert torch.equal(qdq.keys[..., :-rd], expected_nope)
    assert not torch.equal(qdq.keys[..., :-rd], stock.keys[..., :-rd])
    assert qdq.values.data_ptr() == qdq.keys.data_ptr()  # K=V aliasing preserved


def test_layer_compressed_write_qdq_once():
    from v4_kv_quant.qdq_cache import QDQHCACacheLayer

    config, stock, qdq = _layer_pair(DeepseekV4HCACache, QDQHCACacheLayer, main_fp8_nonrope_rope_bf16())
    rd = config.qk_rope_head_dim
    torch.manual_seed(9)
    first = torch.randn(BATCH, 3, config.head_dim)
    second = torch.randn(BATCH, 2, config.head_dim)
    stock.update_compressor_states("compressor", first.clone())
    qdq.update_compressor_states("compressor", first.clone())
    group = effective_group_size(config.head_dim - rd, 64)
    assert torch.equal(qdq.compressed_kv["compressor"][..., -rd:], first[..., -rd:])
    assert torch.equal(
        qdq.compressed_kv["compressor"][..., :-rd], fp8_e4m3_qdq(first[..., :-rd], group_size=group)
    )
    snapshot = qdq.compressed_kv["compressor"].clone()
    qdq.update_compressor_states("compressor", second.clone())
    assert qdq.entry_count["compressor"] == 5
    assert torch.equal(qdq.compressed_kv["compressor"][:, :3], snapshot), "history re-quantized"
    # empty emission (no window closed this forward) is a no-op on history
    qdq.update_compressor_states("compressor", first.new_zeros(BATCH, 0, config.head_dim))
    assert qdq.entry_count["compressor"] == 5


def test_layer_indexer_write_rotated_fp4():
    from v4_kv_quant.qdq_cache import QDQCSACacheLayer

    config = tiny_v4_config()
    policy = reference_official_qdq()
    layer = QDQCSACacheLayer(config, policy)
    torch.manual_seed(10)
    entries = torch.randn(BATCH, 4, config.index_head_dim)
    layer.update_compressor_states("indexer", entries.clone())
    group = effective_group_size(config.index_head_dim, 32)
    expected = fp4_e2m1_qdq(hadamard_transform(entries), group_size=group)
    assert torch.equal(layer.compressed_kv["indexer"], expected)


def test_layer_identity_policy_bitwise():
    from v4_kv_quant.qdq_cache import QDQCSACacheLayer

    config, stock, qdq = _layer_pair(DeepseekV4CSACache, QDQCSACacheLayer, baseline_bf16())
    torch.manual_seed(11)
    kv = torch.randn(BATCH, 1, 6, config.head_dim)
    entries = torch.randn(BATCH, 2, config.head_dim)
    stock.update(kv.clone(), kv.clone())
    qdq.update(kv.clone(), kv.clone())
    stock.update_compressor_states("compressor", entries.clone())
    qdq.update_compressor_states("compressor", entries.clone())
    assert torch.equal(qdq.keys, stock.keys)
    assert torch.equal(qdq.compressed_kv["compressor"], stock.compressed_kv["compressor"])


def test_registry_not_hijacked():
    # constructing/importing QDQ layers must never replace the stock registry entries
    assert DYNAMIC_LAYER_TYPE_MAPPING["heavily_compressed_attention"] is DeepseekV4HCACache
    assert DYNAMIC_LAYER_TYPE_MAPPING["compressed_sparse_attention"] is DeepseekV4CSACache
    config = tiny_v4_config()
    stock_cache = DynamicCache(config=config)
    assert type(stock_cache.layers[1]) is DeepseekV4CSACache
    assert type(stock_cache.layers[2]) is DeepseekV4HCACache


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


def _decode_with_cache(model, ids, cache):
    logits = []
    with torch.no_grad():
        for t in range(ids.shape[1]):
            out = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            logits.append(out.logits[:, 0])
    return torch.stack(logits, dim=1)


def test_identity_policy_bit_exact_model(model_selective):
    ids = deterministic_input_ids(BATCH, 21)
    baseline = run_teacher_forced(model_selective, ids, prefill_len=9, policy=None)
    identity = run_teacher_forced(model_selective, ids, prefill_len=9, policy=baseline_bf16())
    assert torch.equal(identity.logits, baseline.logits), "identity policy must be bit-exact"
    for base_chunk, id_chunk in zip(baseline.indexer_picks, identity.indexer_picks, strict=True):
        assert torch.equal(base_chunk, id_chunk)


def test_identity_policy_states_bit_exact(model_selective):
    ids = deterministic_input_ids(BATCH, 17)
    config = model_selective.config
    stock = DynamicCache(config=config)
    qdq = build_qdq_cache(config, baseline_bf16())
    with torch.no_grad():
        model_selective(ids, past_key_values=stock, use_cache=True)
        model_selective(ids, past_key_values=qdq, use_cache=True)
    for stock_layer, qdq_layer in zip(stock.layers, qdq.layers):
        assert torch.equal(qdq_layer.keys, stock_layer.keys)
        if hasattr(stock_layer, "compressed_kv"):
            for name in stock_layer.compressed_kv:
                assert torch.equal(qdq_layer.compressed_kv[name], stock_layer.compressed_kv[name])
                assert torch.equal(qdq_layer.buffer_kv[name], stock_layer.buffer_kv[name])
            assert qdq_layer.entry_count == stock_layer.entry_count


def test_fp8_policy_effects_and_invariants(model_selective):
    ids = deterministic_input_ids(BATCH, 21)
    baseline = run_teacher_forced(model_selective, ids, prefill_len=9, policy=None)
    quantized = run_teacher_forced(model_selective, ids, prefill_len=9, policy=main_fp8_nonrope_rope_bf16())
    metrics = logit_comparison_metrics(baseline.logits, quantized.logits)
    assert metrics["nan_count"] == 0 and metrics["inf_count"] == 0
    assert metrics["max_abs_logit_err"] > 0.0, "FP8 QDQ should have a measurable effect"
    # Task-01 structural invariants must hold under QDQ
    config = model_selective.config
    cache = build_qdq_cache(config, main_fp8_nonrope_rope_bf16())
    with torch.no_grad():
        model_selective(ids, past_key_values=cache, use_cache=True)
    for i, layer in enumerate(cache.layers):
        layer_type = config.layer_types[i]
        assert layer.cumulative_length == 21
        assert layer.keys.shape[-2] == min(21, config.sliding_window - 1)
        if layer_type != "sliding_attention":
            rate = config.compress_rates[layer_type]
            for name in layer.compressed_kv:
                assert layer.entry_count[name] == 21 // rate
                assert layer.buffer_kv[name].shape[1] == 21 % rate


def test_layer0_window_rope_slice_matches_baseline_across_runs(model_selective):
    """Layer 0 sees identical inputs in baseline and QDQ runs (embeddings), so its stored
    window RoPE slice must be bit-identical across runs while its nope slice differs."""
    ids = deterministic_input_ids(BATCH, 6)  # < window: all positions retained in order
    config = model_selective.config
    rd = config.qk_rope_head_dim
    stock = DynamicCache(config=config)
    qdq = build_qdq_cache(config, main_fp8_nonrope_rope_bf16())
    with torch.no_grad():
        model_selective(ids, past_key_values=stock, use_cache=True)
        model_selective(ids, past_key_values=qdq, use_cache=True)
    assert torch.equal(qdq.layers[0].keys[..., -rd:], stock.layers[0].keys[..., -rd:])
    assert not torch.equal(qdq.layers[0].keys[..., :-rd], stock.layers[0].keys[..., :-rd])


def test_chunked_equals_oneshot_under_qdq(model_dense):
    ids = deterministic_input_ids(BATCH, 27)
    policy = main_fp8_nonrope_rope_bf16()
    config = model_dense.config
    one_shot_cache = build_qdq_cache(config, policy)
    with torch.no_grad():
        one_shot = model_dense(ids, past_key_values=one_shot_cache, use_cache=True).logits
    for chunks in ([13, 14], [5, 1, 7, 8, 6]):
        cache = build_qdq_cache(config, policy)
        outs, pos = [], 0
        with torch.no_grad():
            for size in chunks:
                outs.append(model_dense(ids[:, pos : pos + size], past_key_values=cache, use_cache=True).logits)
                pos += size
        torch.testing.assert_close(torch.cat(outs, dim=1), one_shot, atol=ATOL, rtol=RTOL)
        # cache states also agree: QDQ boundary is per-token/per-entry, chunking-invariant
        for a, b in zip(cache.layers, one_shot_cache.layers):
            torch.testing.assert_close(a.keys, b.keys, atol=ATOL, rtol=RTOL)
            if hasattr(a, "compressed_kv"):
                for name in a.compressed_kv:
                    torch.testing.assert_close(
                        a.compressed_kv[name], b.compressed_kv[name], atol=ATOL, rtol=RTOL
                    )


def test_token_by_token_equals_oneshot_under_qdq(model_dense):
    ids = deterministic_input_ids(BATCH, 27)
    policy = main_fp8_nonrope_rope_bf16()
    config = model_dense.config
    with torch.no_grad():
        one_shot = model_dense(ids, past_key_values=build_qdq_cache(config, policy), use_cache=True).logits
    stepped = _decode_with_cache(model_dense, ids, build_qdq_cache(config, policy))
    torch.testing.assert_close(stepped, one_shot, atol=ATOL, rtol=RTOL)


def test_indexer_policy_end_to_end(model_selective):
    ids = deterministic_input_ids(BATCH, 33)
    baseline = run_teacher_forced(model_selective, ids, prefill_len=17, policy=None)
    quantized = run_teacher_forced(model_selective, ids, prefill_len=17, policy=indexer_reference_qdq())
    assert torch.isfinite(quantized.logits).all()
    overlap = indexer_topk_overlap(baseline.indexer_picks, quantized.indexer_picks)
    assert overlap["positions"] > 0
    assert 0.0 <= overlap["mean_overlap"] <= 1.0
    # scorer restored after the context manager exits
    csa_idx = model_selective.config.layer_types.index(CSA)
    scorer = model_selective.model.layers[csa_idx].self_attn.compressor.indexer.scorer
    assert type(scorer) is DeepseekV4IndexerScorer


def test_reference_official_policy_full_run(model_selective):
    ids = deterministic_input_ids(BATCH, 33)
    baseline = run_teacher_forced(model_selective, ids, prefill_len=17, policy=None)
    quantized = run_teacher_forced(model_selective, ids, prefill_len=17, policy=reference_official_qdq())
    metrics = logit_comparison_metrics(baseline.logits, quantized.logits)
    assert metrics["nan_count"] == 0 and metrics["inf_count"] == 0
    nll_base = next_token_nll(baseline.logits, ids)
    nll_quant = next_token_nll(quantized.logits, ids)
    assert torch.isfinite(torch.tensor([nll_base, nll_quant])).all()


def test_qdq_cache_repr_and_policy_attached():
    config = tiny_v4_config()
    cache = QDQCache(config, reference_official_qdq())
    assert "reference_official_qdq" in repr(cache)
    assert len(cache.layers) == config.num_hidden_layers


# ---------------------------------------------------------------------------
# Policy serialization
# ---------------------------------------------------------------------------


def test_policy_json_roundtrip(tmp_path):
    for name, factory in NAMED_POLICIES.items():
        policy = factory()
        path = tmp_path / f"{name}.json"
        policy.to_json(path)
        restored = KVQuantPolicy.from_json(path)
        assert restored == policy
    assert baseline_bf16().is_identity
    assert not reference_official_qdq().is_identity


def test_policy_version_rejected():
    bad = baseline_bf16().to_dict()
    bad["version"] = 99
    with pytest.raises(ValueError, match="version"):
        KVQuantPolicy.from_dict(bad)
