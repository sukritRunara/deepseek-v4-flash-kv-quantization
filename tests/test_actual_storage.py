"""Task 04 acceptance tests: actual low-precision storage (Stage C).

Gates (prompts/04_ACTUAL_STORAGE_PROTOTYPE.md): load(store(x)) == qdq(x) bitwise;
storage cache bitwise-equal to the Stage-B QDQ cache at model level; bf16 policy
identity; write-once codes; trim without history retention; count invariants;
real byte reduction with scales included.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from v4_kv_quant.harness import run_teacher_forced
from v4_kv_quant.memory import cache_memory_report, compare_reports
from v4_kv_quant.policy import (
    StatePolicy,
    baseline_bf16,
    main_fp8_nonrope_rope_bf16,
    reference_official_qdq,
)
from v4_kv_quant.qdq import fp4_e2m1_qdq, fp8_e4m3_qdq, hadamard_transform
from v4_kv_quant.qdq_cache import QDQHCACacheLayer
from v4_kv_quant.storage import fp4_store, fp8_store, load, stored_bytes
from v4_kv_quant.storage_cache import (
    QuantizedStorageCache,
    QuantizedStorageHCALayer,
    QuantizedStorageSlidingLayer,
)
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids, tiny_v4_config

BATCH = 2


@pytest.fixture(scope="module")
def model():
    return build_tiny_model(seed=0)


@pytest.fixture(scope="module")
def config():
    return tiny_v4_config()


# ---------------------------------------------------------------------------
# Storage primitives: load(store(x)) == qdq(x) bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp8_store_load_matches_qdq(dtype):
    torch.manual_seed(30)
    x = (torch.randn(8, 448) * 5).to(dtype)
    stored = fp8_store(x, group_size=64)
    assert stored.codes.dtype == torch.float8_e4m3fn
    assert stored.scales.dtype == torch.float8_e8m0fnu  # 1-byte power-of-2 scales
    assert stored.scales.shape == (8, 7)
    roundtrip = load(stored)
    assert roundtrip.dtype == dtype
    assert torch.equal(roundtrip, fp8_e4m3_qdq(x, group_size=64))
    # non-pow2 scales fall back to fp32 scale storage, still bitwise-parity with QDQ
    stored_lin = fp8_store(x, group_size=64, pow2_scale=False)
    assert stored_lin.scales.dtype == torch.float32
    assert torch.equal(load(stored_lin), fp8_e4m3_qdq(x, group_size=64, pow2_scale=False))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp4_store_load_matches_qdq(dtype):
    torch.manual_seed(31)
    x = (torch.randn(8, 128) * 3).to(dtype)
    stored = fp4_store(x, group_size=32)
    assert stored.codes.dtype == torch.uint8
    assert stored.codes.shape == (8, 64)  # two nibbles per byte
    assert stored.scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(load(stored), fp4_e2m1_qdq(x, group_size=32))


def test_fp4_store_load_edge_values():
    # grid values, exact midpoints (RNE ties), zeros, negatives — all must match the QDQ sim
    x = torch.tensor([[0.0, -0.0, 0.25, -0.75, 1.25, -1.75, 2.5, -3.5, 5.0, -6.0, 6.0, 0.5,
                       -1.0, 1.5, -2.0, 3.0]])
    stored = fp4_store(x, group_size=16)
    assert torch.equal(load(stored), fp4_e2m1_qdq(x, group_size=16))


def test_fp4_rotated_matches_rotated_qdq():
    torch.manual_seed(32)
    x = torch.randn(4, 6, 16)
    rotated = hadamard_transform(x)
    assert torch.equal(load(fp4_store(rotated, group_size=16)), fp4_e2m1_qdq(rotated, group_size=16))


def test_stored_bytes_itemized():
    x = torch.randn(4, 448)
    fp8_bytes = stored_bytes(fp8_store(x, group_size=64))
    assert fp8_bytes == {"codes": 4 * 448, "scales": 4 * 7}  # 1 B codes + 1 B e8m0 scales
    fp4_bytes = stored_bytes(fp4_store(torch.randn(4, 128), group_size=32))
    assert fp4_bytes == {"codes": 4 * 64, "scales": 4 * 4}


# ---------------------------------------------------------------------------
# Layer level: storage layer values == QDQ layer values, write-once, trim
# ---------------------------------------------------------------------------


def test_storage_window_matches_qdq_layer(config):
    policy = main_fp8_nonrope_rope_bf16()
    qdq_layer = QDQHCACacheLayer(config, policy)
    storage_layer = QuantizedStorageHCALayer(config, policy)
    torch.manual_seed(33)
    for step_len in (5, 1, 4, 1):  # prefill + decode-ish sequence crossing the window
        kv = torch.randn(BATCH, 1, step_len, config.head_dim)
        full_qdq = qdq_layer.update(kv.clone(), kv.clone())[0]
        full_storage = storage_layer.update(kv.clone(), kv.clone())[0]
        assert torch.equal(full_storage, full_qdq)
    assert storage_layer.cumulative_length == qdq_layer.cumulative_length == 11
    assert storage_layer.keys.numel() == 0, "placeholder only — no BF16 KV storage retained"
    assert storage_layer._window.rows == config.sliding_window - 1


def test_storage_compressed_write_once_and_matches(config):
    policy = main_fp8_nonrope_rope_bf16()
    qdq_layer = QDQHCACacheLayer(config, policy)
    storage_layer = QuantizedStorageHCALayer(config, policy)
    torch.manual_seed(34)
    first = torch.randn(BATCH, 3, config.head_dim)
    second = torch.randn(BATCH, 2, config.head_dim)
    out_qdq = qdq_layer.update_compressor_states("compressor", first.clone())
    out_storage = storage_layer.update_compressor_states("compressor", first.clone())
    assert torch.equal(out_storage, out_qdq)
    codes_snapshot = storage_layer._compressed.tensors["codes"].clone()
    out_qdq = qdq_layer.update_compressor_states("compressor", second.clone())
    out_storage = storage_layer.update_compressor_states("compressor", second.clone())
    assert torch.equal(out_storage, out_qdq)
    assert storage_layer.entry_count["compressor"] == 5
    assert torch.equal(
        storage_layer._compressed.tensors["codes"][:, :3].view(torch.uint8),
        codes_snapshot.view(torch.uint8),
    ), "codes must be written exactly once"
    # empty emission is a no-op
    out_storage = storage_layer.update_compressor_states("compressor", first.new_zeros(BATCH, 0, config.head_dim))
    assert out_storage.shape[-2] == 5 and storage_layer.entry_count["compressor"] == 5


def test_window_trim_drops_history(config):
    policy = main_fp8_nonrope_rope_bf16()
    layer = QuantizedStorageSlidingLayer(config, policy)
    torch.manual_seed(35)
    for _ in range(20):  # decode far past the window
        layer.update(torch.randn(BATCH, 1, 1, config.head_dim), None)
    assert layer._window.rows == config.sliding_window - 1
    for name, t in layer._window.tensors.items():
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size(), (
            f"{name}: trimmed store retains hidden history"
        )


# ---------------------------------------------------------------------------
# Model level: bitwise equivalence with the Stage-B QDQ cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_factory", [main_fp8_nonrope_rope_bf16, reference_official_qdq])
def test_storage_cache_bitwise_equals_qdq_cache(model, policy_factory):
    ids = deterministic_input_ids(BATCH, 27)
    policy = policy_factory()
    simulated = run_teacher_forced(model, ids, prefill_len=13, policy=policy)
    actual = run_teacher_forced(model, ids, prefill_len=13, policy=policy, storage=True)
    assert torch.equal(actual.logits, simulated.logits), (
        "Stage-C storage must be numerically indistinguishable from Stage-B QDQ"
    )
    for sim_chunk, act_chunk in zip(simulated.indexer_picks, actual.indexer_picks, strict=True):
        assert torch.equal(sim_chunk, act_chunk)


def test_storage_cache_bf16_policy_matches_stock(model):
    ids = deterministic_input_ids(BATCH, 21)
    baseline = run_teacher_forced(model, ids, prefill_len=9)
    raw_storage = run_teacher_forced(model, ids, prefill_len=9, policy=baseline_bf16(), storage=True)
    assert torch.equal(raw_storage.logits, baseline.logits)


def test_storage_cache_count_invariants(model):
    ids = deterministic_input_ids(BATCH, 21)
    config = model.config
    cache = QuantizedStorageCache(config, reference_official_qdq())
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    for i, layer in enumerate(cache.layers):
        layer_type = config.layer_types[i]
        assert layer.cumulative_length == 21
        assert layer.get_seq_length() == 21
        assert layer._window.rows == min(21, config.sliding_window - 1)
        if layer_type == "sliding_attention":
            continue
        rate = config.compress_rates[layer_type]
        assert layer.entry_count["compressor"] == 21 // rate
        assert layer.buffer_kv["compressor"].shape[1] == 21 % rate  # stock BF16 buffers intact
        assert layer._compressed.rows == 21 // rate
        if layer_type == "compressed_sparse_attention":
            assert layer.entry_count["indexer"] == 21 // rate
            assert layer._indexer.rows == 21 // rate


# ---------------------------------------------------------------------------
# Memory accounting: real reduction, honestly counted
# ---------------------------------------------------------------------------


def _fill_caches(model, seq_len=64, decode=8):
    ids = deterministic_input_ids(BATCH, seq_len + decode)
    config = model.config
    caches = {
        "baseline_bf16_stock": DynamicCache(config=config),
        "stage_b_qdq_sim": __import__("v4_kv_quant.qdq_cache", fromlist=["QDQCache"]).QDQCache(
            config, reference_official_qdq()
        ),
        "stage_c_storage": QuantizedStorageCache(config, reference_official_qdq()),
    }
    for cache in caches.values():
        with torch.no_grad():
            model(ids[:, :seq_len], past_key_values=cache, use_cache=True)
            for t in range(seq_len, seq_len + decode):
                model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    return caches


def test_memory_reports_show_real_reduction(model):
    caches = _fill_caches(model)
    reports = {name: cache_memory_report(cache, label=name) for name, cache in caches.items()}
    base = reports["baseline_bf16_stock"]["total_logical_bytes"]
    sim = reports["stage_b_qdq_sim"]["total_logical_bytes"]
    storage = reports["stage_c_storage"]["total_logical_bytes"]
    assert sim == base, "Stage-B QDQ simulation must save exactly nothing (CLAUDE.md constraint 7)"
    assert storage < base, "Stage-C must show a real byte reduction"
    comparison = compare_reports(reports["baseline_bf16_stock"], reports["stage_c_storage"])
    assert comparison["ratio"] < 0.75  # fp32 tiny model: fp8 nope+scales+raw rope is well under 3/4
    # the stock sliding-only layer duplicates V; storage layers must not
    stock_sliding_states = reports["baseline_bf16_stock"]["layers"][0]["states"]
    assert any("duplicate" in name for name in stock_sliding_states)
    storage_sliding_states = reports["stage_c_storage"]["layers"][0]["states"]
    assert not any("duplicate" in name for name in storage_sliding_states)
    # scales are itemized inside the quantized totals
    csa_states = reports["stage_c_storage"]["layers"][1]["states"]
    assert any(name.endswith(".scales") for name in csa_states)
    assert any(name.endswith(".codes") for name in csa_states)
    # BF16 compressor buffers still counted
    assert any(name.startswith("buffer_kv") for name in csa_states)
