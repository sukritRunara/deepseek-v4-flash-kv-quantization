"""MappedStorageCache (D-016): Stage-C storage for per-group precision maps.

Contract mirrors D-009 at map granularity: bitwise identical logits AND indexer
picks vs `MappedQDQCache` under the same map, on the tiny model. Segments reuse
the `load(store(x)) == qdq(x)` primitives, so equality here plus the existing
per-format contracts is the whole argument.
"""

from __future__ import annotations

import pytest
import torch

from v4_kv_quant.harness import run_teacher_forced
from v4_kv_quant.mapped_storage_cache import MappedStorageCache, SegmentedStore
from v4_kv_quant.precision_map import MapEntry, PrecisionMap
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids, tiny_v4_config

BATCH = 2


@pytest.fixture(scope="module")
def model():
    return build_tiny_model(seed=0)


def _map(name, entries):
    return PrecisionMap(name=name, entries=entries)


def _maps(config):
    """Representative maps: empty, one fp8 group, one fp4 group, mixed + indexer."""
    nope = config.head_dim - config.qk_rope_head_dim  # tiny: 24 - 8 = 16
    g = 8
    yield "empty", _map("empty", [])
    yield "fp8_group", _map("fp8_group", [
        MapEntry(layer_idx=1, state="window_kv", group_index=0, start=0, end=g, kind="fp8_e4m3",
                 scale_group_size=g)])
    yield "fp4_group", _map("fp4_group", [
        MapEntry(layer_idx=2, state="compressed_kv", group_index=1, start=g, end=2 * g,
                 kind="fp4_e2m1", scale_group_size=g)])
    yield "mixed", _map("mixed", [
        MapEntry(layer_idx=0, state="window_kv", group_index=0, start=0, end=nope, kind="fp8_e4m3",
                 scale_group_size=g),
        MapEntry(layer_idx=1, state="window_kv", group_index=1, start=g, end=2 * g, kind="fp4_e2m1",
                 scale_group_size=g),
        MapEntry(layer_idx=1, state="compressed_kv", group_index=0, start=0, end=nope,
                 kind="fp8_e4m3", scale_group_size=g),
        MapEntry(layer_idx=1, state="indexer_kv", group_index=0, start=0,
                 end=config.index_head_dim, kind="fp4_e2m1_hadamard",
                 scale_group_size=config.index_head_dim),
        MapEntry(layer_idx=2, state="window_kv", group_index=0, start=0, end=nope, kind="fp4_e2m1",
                 scale_group_size=g),
    ])


def test_mapped_storage_bitwise_equals_mapped_qdq(model):
    ids = deterministic_input_ids(BATCH, 27)
    for name, pm in _maps(model.config):
        sim = run_teacher_forced(model, ids, prefill_len=13, precision_map=pm)
        act = run_teacher_forced(model, ids, prefill_len=13, precision_map=pm,
                                 storage=True)
        assert torch.equal(act.logits, sim.logits), name
        for s, a in zip(sim.indexer_picks, act.indexer_picks, strict=True):
            assert torch.equal(s, a), name


def test_mapped_storage_empty_map_matches_stock(model):
    ids = deterministic_input_ids(BATCH, 21)
    baseline = run_teacher_forced(model, ids, prefill_len=9)
    stored = run_teacher_forced(model, ids, prefill_len=9,
                                precision_map=_map("empty", []), storage=True)
    assert torch.equal(stored.logits, baseline.logits)


def test_mapped_storage_actually_stores_low_precision(model):
    config = model.config
    nope = config.head_dim - config.qk_rope_head_dim
    pm = _map("fp8_all_window", [
        MapEntry(layer_idx=i, state="window_kv", group_index=0, start=0, end=nope, kind="fp8_e4m3",
                 scale_group_size=8) for i in range(len(config.layer_types))])
    cache = MappedStorageCache(config, pm)
    ids = deterministic_input_ids(BATCH, 21)
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    tensors = cache.layers[0].memory_state_tensors()
    code_dtypes = {t.dtype for n, t in tensors.items() if ".codes" in n}
    assert code_dtypes == {torch.float8_e4m3fn}
    assert any(".raw" in n for n in tensors)  # rope segment stays raw
    assert cache.layers[0].keys.numel() == 0 if cache.layers[0].keys is not None else True


def test_segmented_store_rejects_overlaps():
    with pytest.raises(ValueError):
        SegmentedStore([
            MapEntry(layer_idx=0, state="window_kv", group_index=0, start=0, end=8, kind="fp8_e4m3",
                     scale_group_size=8),
            MapEntry(layer_idx=0, state="window_kv", group_index=0, start=4, end=12, kind="fp8_e4m3",
                     scale_group_size=8),
        ], nope_end=16, rope_dim=8)
