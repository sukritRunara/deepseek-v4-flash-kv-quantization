"""Task 03 acceptance tests: calibration and precision-map plumbing.

Gates (prompts/03_CALIBRATION_PLUMBING.md): empty map bit-exact; single-group entry
perturbs only its slice; full-coverage map == Task-02 policy cache bitwise; stats
collector is pass-through; sweep deterministic with indexer overlap; smoke produces a
valid map with provenance.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from v4_kv_quant.harness import run_teacher_forced
from v4_kv_quant.mapped_cache import MappedHCACacheLayer, MappedQDQCache
from v4_kv_quant.metrics import next_token_nll
from v4_kv_quant.policy import main_fp8_nonrope_rope_bf16
from v4_kv_quant.precision_map import MapEntry, PrecisionMap
from v4_kv_quant.qdq import fp8_e4m3_qdq
from v4_kv_quant.sensitivity import (
    SensitivityRecord,
    build_map_from_sweep,
    measure_target,
    run_sensitivity_sweep,
)
from v4_kv_quant.stats import StatsCollectorCache, StatsRecorder
from v4_kv_quant.targets import QuantTarget, enumerate_targets, nope_width
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids, tiny_v4_config

BATCH = 2
CSA = "compressed_sparse_attention"
HCA = "heavily_compressed_attention"


@pytest.fixture(scope="module")
def model():
    return build_tiny_model(seed=0)


@pytest.fixture(scope="module")
def config():
    return tiny_v4_config()


# ---------------------------------------------------------------------------
# Target taxonomy
# ---------------------------------------------------------------------------


def test_enumerate_targets_tiny(config):
    targets = enumerate_targets(config, group_size_main=8)
    # layer0 sliding: 3 window groups; layer1 CSA: 3+3+1; layer2 HCA: 3+3
    assert len(targets) == 16
    by_layer_state = {}
    for t in targets:
        by_layer_state.setdefault((t.layer_idx, t.state), []).append(t)
    assert set(by_layer_state) == {
        (0, "window_kv"),
        (1, "window_kv"), (1, "compressed_kv"), (1, "indexer_kv"),
        (2, "window_kv"), (2, "compressed_kv"),
    }
    width = nope_width(config)
    for (layer, state), group in by_layer_state.items():
        if state == "indexer_kv":
            assert len(group) == 1
            assert (group[0].start, group[0].end) == (0, config.index_head_dim)
        else:
            assert [(t.start, t.end) for t in group] == [(0, 8), (8, 16), (16, 24)]
            assert group[-1].end == width
    # production-style request falls back to one whole-width group on the tiny model
    assert len(enumerate_targets(config, group_size_main=64)) == 6
    assert [t.layer_idx for t in enumerate_targets(config, group_size_main=8, layer_indices=[1])] == [1] * 7


# ---------------------------------------------------------------------------
# PrecisionMap validation and serialization
# ---------------------------------------------------------------------------


def test_precision_map_validation(config):
    ok = PrecisionMap(entries=[MapEntry(1, "window_kv", 0, 0, 8, "fp8_e4m3")])
    ok.validate(config)
    cases = [
        (MapEntry(99, "window_kv", 0, 0, 8, "fp8_e4m3"), "layer_idx"),
        (MapEntry(0, "compressed_kv", 0, 0, 8, "fp8_e4m3"), "not present"),  # sliding layer
        (MapEntry(1, "window_kv", 0, 0, 999, "fp8_e4m3"), "outside width"),
        (MapEntry(1, "window_kv", 0, 0, 8, "fp4_e2m1_hadamard"), "not in"),  # rotation invalid on main
        (MapEntry(1, "indexer_kv", 0, 0, 8, "fp4_e2m1_hadamard"), "full vector"),
        (MapEntry(2, "indexer_kv", 0, 0, 16, "fp4_e2m1_hadamard"), "not present"),  # HCA has no indexer
    ]
    for entry, match in cases:
        with pytest.raises(ValueError, match=match):
            PrecisionMap(entries=[entry]).validate(config)
    overlapping = PrecisionMap(entries=[
        MapEntry(1, "window_kv", 0, 0, 12, "fp8_e4m3"),
        MapEntry(1, "window_kv", 1, 8, 16, "fp8_e4m3"),
    ])
    with pytest.raises(ValueError, match="overlaps"):
        overlapping.validate(config)


def test_precision_map_json_roundtrip(tmp_path, config):
    original = PrecisionMap(
        name="roundtrip",
        entries=[
            MapEntry(1, "window_kv", 1, 8, 16, "fp8_e4m3"),
            MapEntry(1, "indexer_kv", 0, 0, 16, "fp4_e2m1_hadamard", scale_group_size=32),
        ],
        provenance={"seed": 0},
    )
    original.validate(config)
    path = tmp_path / "map.json"
    original.to_json(path)
    restored = PrecisionMap.from_json(path)
    assert restored == original
    bad = original.to_dict()
    bad["version"] = 99
    with pytest.raises(ValueError, match="version"):
        PrecisionMap.from_dict(bad)


# ---------------------------------------------------------------------------
# Mapped cache semantics
# ---------------------------------------------------------------------------


def test_empty_map_bit_exact(model):
    ids = deterministic_input_ids(BATCH, 21)
    config = model.config
    stock = DynamicCache(config=config)
    mapped = MappedQDQCache(config, PrecisionMap(name="empty"))
    with torch.no_grad():
        logits_stock = model(ids, past_key_values=stock, use_cache=True).logits
        logits_mapped = model(ids, past_key_values=mapped, use_cache=True).logits
    assert torch.equal(logits_mapped, logits_stock)
    for a, b in zip(mapped.layers, stock.layers):
        assert torch.equal(a.keys, b.keys)
        if hasattr(b, "compressed_kv"):
            for name in b.compressed_kv:
                assert torch.equal(a.compressed_kv[name], b.compressed_kv[name])


def test_single_group_perturbs_only_its_slice(config):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACache

    layer_idx = config.layer_types.index(HCA)
    entry = MapEntry(layer_idx, "window_kv", 1, 8, 16, "fp8_e4m3")
    single = PrecisionMap(name="one-group", entries=[entry])
    single.validate(config)
    stock = DeepseekV4HCACache(config)
    mapped = MappedHCACacheLayer(config, single, layer_idx)
    torch.manual_seed(20)
    kv = torch.randn(BATCH, 1, 5, config.head_dim)
    entries = torch.randn(BATCH, 2, config.head_dim)
    stock.update(kv.clone(), kv.clone())
    mapped.update(kv.clone(), kv.clone())
    stock.update_compressor_states("compressor", entries.clone())
    mapped.update_compressor_states("compressor", entries.clone())
    # only window channels 8..16 differ; everything else bitwise identical
    assert not torch.equal(mapped.keys[..., 8:16], stock.keys[..., 8:16])
    assert torch.equal(mapped.keys[..., :8], stock.keys[..., :8])
    assert torch.equal(mapped.keys[..., 16:], stock.keys[..., 16:])  # incl. rope slice
    assert torch.equal(mapped.compressed_kv["compressor"], stock.compressed_kv["compressor"])
    expected = fp8_e4m3_qdq(stock.keys[..., 8:16], group_size=8)
    assert torch.equal(mapped.keys[..., 8:16], expected)


def test_mapped_partition_matches_primitive(config):
    layer_idx = config.layer_types.index(HCA)
    width = nope_width(config)
    partition = PrecisionMap(
        name="partition",
        entries=[MapEntry(layer_idx, "window_kv", g, g * 8, (g + 1) * 8, "fp8_e4m3") for g in range(width // 8)],
    )
    partition.validate(config)
    mapped = MappedHCACacheLayer(config, partition, layer_idx)
    torch.manual_seed(21)
    kv = torch.randn(BATCH, 1, 5, config.head_dim)
    mapped.update(kv.clone(), kv.clone())
    expected = fp8_e4m3_qdq(kv[..., :width], group_size=8)
    assert torch.equal(mapped.keys[..., :width], expected)
    assert torch.equal(mapped.keys[..., width:], kv[..., width:])


def test_full_coverage_matches_policy_cache(model):
    """A map covering every main state whole-width must reproduce the Task-02 policy
    cache bitwise (same effective scale-group geometry)."""
    ids = deterministic_input_ids(BATCH, 21)
    config = model.config
    entries = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        entries.append(MapEntry(layer_idx, "window_kv", 0, 0, nope_width(config), "fp8_e4m3"))
        if layer_type != "sliding_attention":
            entries.append(MapEntry(layer_idx, "compressed_kv", 0, 0, nope_width(config), "fp8_e4m3"))
    full_map = PrecisionMap(name="full", entries=entries)
    from_policy = run_teacher_forced(model, ids, prefill_len=9, policy=main_fp8_nonrope_rope_bf16())
    from_map = run_teacher_forced(model, ids, prefill_len=9, precision_map=full_map)
    assert torch.equal(from_map.logits, from_policy.logits)


# ---------------------------------------------------------------------------
# Stats collector
# ---------------------------------------------------------------------------


def test_stats_collector_bit_exact_and_populated(model):
    ids = deterministic_input_ids(BATCH, 21)
    config = model.config
    stock = DynamicCache(config=config)
    recorder = StatsRecorder(group_size_main=8, group_size_indexer=32)
    collector = StatsCollectorCache(config, recorder)
    with torch.no_grad():
        logits_stock = model(ids, past_key_values=stock, use_cache=True).logits
        logits_stats = model(ids, past_key_values=collector, use_cache=True).logits
    assert torch.equal(logits_stats, logits_stock), "stats collection must not alter values"
    summary = recorder.summary()
    assert set(summary) == {
        "layer0/window_kv", "layer1/window_kv", "layer1/compressed_kv", "layer1/indexer_kv",
        "layer2/window_kv", "layer2/compressed_kv",
    }
    for key, stats in summary.items():
        assert stats["elements"] > 0, key
        assert stats["amax"] > 0, key
        assert stats["rms_qdq_err"] >= 0, key
        assert len(stats["per_group_amax"]) >= 1
        assert max(stats["per_group_amax"]) == pytest.approx(stats["amax"], rel=1e-6)


def test_stats_recorder_amax_manual(config):
    recorder = StatsRecorder(group_size_main=8)
    nope = torch.zeros(1, 1, 2, 24)
    nope[0, 0, 0, 3] = -5.0   # group 0
    nope[0, 0, 1, 20] = 2.0   # group 2
    recorder.record_main(1, "window_kv", nope)
    stats = recorder.summary()["layer1/window_kv"]
    assert stats["amax"] == 5.0
    assert stats["per_group_amax"] == [5.0, 0.0, 2.0]
    assert stats["elements"] == 48


# ---------------------------------------------------------------------------
# Sensitivity sweep and map building
# ---------------------------------------------------------------------------


def test_sensitivity_deterministic_and_nonzero(model):
    ids = deterministic_input_ids(BATCH, 25)
    targets = enumerate_targets(model.config, group_size_main=8, layer_indices=[0])  # 3 window groups
    baseline, baseline_nll, records = run_sensitivity_sweep(model, ids, prefill_len=13, targets=targets)
    assert len(records) == 3
    assert all(r.score > 0 for r in records), "FP8 QDQ on layer-0 window must have an effect"
    assert all(r.extra["nan_count"] == 0 for r in records)
    repeat = measure_target(model, ids, 13, targets[0], baseline, baseline_nll)
    assert repeat.score == records[0].score
    assert repeat.kl_mean == records[0].kl_mean


def test_indexer_target_reports_overlap(model):
    ids = deterministic_input_ids(BATCH, 25)
    csa_idx = model.config.layer_types.index(CSA)
    indexer_targets = [
        t for t in enumerate_targets(model.config, layer_indices=[csa_idx]) if t.state == "indexer_kv"
    ]
    assert len(indexer_targets) == 1
    _, _, records = run_sensitivity_sweep(model, ids, prefill_len=13, targets=indexer_targets)
    overlap = records[0].indexer_overlap
    assert overlap is not None and overlap["positions"] > 0
    assert 0.0 <= overlap["mean_overlap"] <= 1.0


def _fake_record(layer_idx, state, group_index, score, overlap=None):
    width = 8
    target = QuantTarget(layer_idx, HCA if state != "indexer_kv" else CSA, state,
                         group_index, group_index * width, (group_index + 1) * width)
    if state == "indexer_kv":
        target = QuantTarget(layer_idx, CSA, state, 0, 0, 16)
    return SensitivityRecord(
        target=target, kind="fp8_e4m3", nll_delta=score / 2, kl_mean=score / 2,
        max_abs_logit_err=score, top1_agreement=1.0,
        indexer_overlap={"mean_overlap": overlap} if overlap is not None else None,
    )


def test_build_map_from_sweep_fractions():
    records = [_fake_record(2, "window_kv", g, score) for g, score in enumerate([0.1, 0.4, 0.2, 0.3])]
    records.append(_fake_record(1, "indexer_kv", 0, 0.05, overlap=0.95))
    records.append(_fake_record(1, "indexer_kv", 0, 0.05, overlap=0.5))
    built = build_map_from_sweep(records, name="test", fp8_fraction=0.5, fp4_fraction=0.25,
                                 indexer_min_overlap=0.9)
    main_entries = [e for e in built.entries if e.state == "window_kv"]
    # ascending scores: g0 (0.1) -> fp4 (least sensitive 25%), g2 (0.2) -> fp8, g3 (0.3) -> fp8? no:
    # n=4, n_fp4=1, n_fp8=2 -> g0 fp4; g2, g3 fp8; g1 (most sensitive) stays bf16 (no entry)
    kinds = {e.group_index: e.kind for e in main_entries}
    assert kinds == {0: "fp4_e2m1", 2: "fp8_e4m3", 3: "fp8_e4m3"}
    indexer_entries = built.indexer_entries()
    assert len(indexer_entries) == 1  # only the overlap>=0.9 record qualifies
    assert indexer_entries[0].kind == "fp4_e2m1_hadamard"
    assert built.provenance["fp8_fraction"] == 0.5
    with pytest.raises(ValueError):
        build_map_from_sweep(records, name="bad", fp8_fraction=0.9, fp4_fraction=0.4)


def test_smoke_end_to_end(model, tmp_path):
    """Mini calibration: sweep layer-1 targets, build map, validate, held-out eval."""
    config = model.config
    calib_ids = deterministic_input_ids(BATCH, 25, seed=11)
    heldout_ids = deterministic_input_ids(BATCH, 25, seed=12)
    assert not torch.equal(calib_ids, heldout_ids), "calibration and held-out sets must differ"
    targets = enumerate_targets(config, group_size_main=8, layer_indices=[1])
    _, baseline_nll, records = run_sensitivity_sweep(model, calib_ids, prefill_len=13, targets=targets)
    built = build_map_from_sweep(
        records, name="smoke", fp8_fraction=0.5, fp4_fraction=0.0, indexer_min_overlap=0.0,
        provenance={"calib_seed": 11, "model_seed": 0},
    )
    built.validate(config)
    path = tmp_path / "map.json"
    built.to_json(path)
    restored = PrecisionMap.from_json(path)
    assert restored.provenance["calib_seed"] == 11
    assert restored.indexer_entries(), "indexer_min_overlap=0 must admit the indexer entry"
    # held-out evaluation with the built map runs and stays finite
    baseline_held = run_teacher_forced(model, heldout_ids, prefill_len=13)
    quant_held = run_teacher_forced(model, heldout_ids, prefill_len=13, precision_map=restored)
    assert torch.isfinite(quant_held.logits).all()
    nll_delta = next_token_nll(quant_held.logits, heldout_ids) - next_token_nll(
        baseline_held.logits, heldout_ids
    )
    assert torch.isfinite(torch.tensor(nll_delta))
