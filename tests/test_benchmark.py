"""Task 05 acceptance tests: benchmark engine and landing checks.

Timings themselves are not asserted (machine-dependent); structure, fairness
guarantees, byte accounting, and check logic are.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch

from v4_kv_quant.bench import BenchSettings, quantization_overhead_microbench, run_benchmark
from v4_kv_quant.landing import checks_passed, run_landing_checks
from v4_kv_quant.tiny_model import build_tiny_model, tiny_v4_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model():
    return build_tiny_model(seed=0)


# ---------------------------------------------------------------------------
# Settings / config plumbing
# ---------------------------------------------------------------------------


def test_settings_from_config_file():
    settings = BenchSettings.from_file(REPO_ROOT / "configs/bench_tiny_local.json")
    assert settings.model == "tiny"
    assert settings.variants == ["baseline", "qdq", "storage"]
    assert settings.policy == "reference_official_qdq"
    # CLI-style overrides win; None overrides are ignored
    overridden = BenchSettings.from_file(
        REPO_ROOT / "configs/bench_tiny_local.json", trials=2, policy=None, prompt_lens=[8]
    )
    assert overridden.trials == 2
    assert overridden.policy == "reference_official_qdq"
    assert overridden.prompt_lens == [8]


def test_settings_validation():
    with pytest.raises(ValueError, match="unknown variants"):
        BenchSettings(variants=["baseline", "bogus"])
    with pytest.raises(ValueError, match="unknown policy"):
        BenchSettings(policy="nope")
    with pytest.raises(ValueError, match="trials"):
        BenchSettings(trials=0)
    with pytest.raises(ValueError, match="prefill_chunk"):
        BenchSettings(prefill_chunk=0)


def test_chunked_prefill_equivalent_accounting(model):
    """prefill_chunk changes memory shape, not semantics: same tokens through the same
    cache => identical cache byte accounting as one-shot, and the runpod config's chunk
    setting parses through the schema."""
    base = BenchSettings(
        device="cpu", batch=1, prompt_lens=[12], decode_tokens=2, trials=1, warmup=0, seed=0
    )
    chunked = BenchSettings(
        device="cpu", batch=1, prompt_lens=[12], decode_tokens=2, trials=1, warmup=0, seed=0,
        prefill_chunk=5,  # deliberately not a divisor of 12 or the compress rates
    )
    rows_base = {r["variant"]: r for r in run_benchmark(model, base)["results"]}
    rows_chunked = {r["variant"]: r for r in run_benchmark(model, chunked)["results"]}
    for variant in rows_base:
        assert "error" not in rows_chunked[variant]
        assert (rows_chunked[variant]["median"]["cache_logical_bytes"]
                == rows_base[variant]["median"]["cache_logical_bytes"])
        assert (rows_chunked[variant]["median"]["cache_storage_bytes"]
                == rows_base[variant]["median"]["cache_storage_bytes"])
    settings = BenchSettings.from_file(REPO_ROOT / "configs/bench_runpod_4gpu.json")
    assert settings.prefill_chunk == 2048


def test_runpod_config_parses_with_same_schema():
    settings = BenchSettings.from_file(REPO_ROOT / "configs/bench_runpod_4gpu.json")
    assert settings.model_path is not None
    assert settings.device_map == "auto"
    assert settings.variants == ["baseline", "qdq", "storage"]


# ---------------------------------------------------------------------------
# Benchmark engine (CPU, tiny sizes — structure and fairness, not speed)
# ---------------------------------------------------------------------------


def test_run_benchmark_structure_and_accounting(model):
    settings = BenchSettings(
        device="cpu", batch=1, prompt_lens=[12], decode_tokens=4, trials=2, warmup=1, seed=0
    )
    report = run_benchmark(model, settings)
    assert report["settings"]["resolved_device"] == "cpu"
    assert "non_transferability" in report and "constraint 8" in report["non_transferability"]
    rows = {row["variant"]: row for row in report["results"]}
    assert set(rows) == {"baseline", "qdq", "storage"}
    for row in rows.values():
        assert len(row["trials"]) == 2  # warmup excluded
        for trial in row["trials"]:
            assert len(trial["itl_s"]) == 4
            assert trial["ttft_s"] > 0
            assert trial["peak_memory"] is None  # cpu run
        median = row["median"]
        assert median["prefill_tokens_per_s"] > 0
        assert median["decode_tokens_per_s"] > 0
        assert median["itl_p50_ms"] > 0
    # identical token streams -> cache byte accounting mirrors the Task-04 result
    assert rows["qdq"]["median"]["cache_logical_bytes"] == rows["baseline"]["median"]["cache_logical_bytes"]
    assert rows["storage"]["median"]["cache_logical_bytes"] < rows["baseline"]["median"]["cache_logical_bytes"]
    # deterministic byte accounting across repeated runs
    report2 = run_benchmark(model, settings)
    rows2 = {row["variant"]: row for row in report2["results"]}
    for name in rows:
        assert rows2[name]["median"]["cache_logical_bytes"] == rows[name]["median"]["cache_logical_bytes"]


def test_quantization_overhead_microbench_structure():
    config = tiny_v4_config()
    overhead = quantization_overhead_microbench(config, device="cpu", context_tokens=48, batch=1, iters=3)
    expected_keys = {
        "fp8_encode_window_step", "fp8_decode_window_full", "fp8_decode_compressed_full",
        "fp4_encode_indexer_entry", "fp4_decode_indexer_full",
    }
    assert set(overhead) == expected_keys
    for entry in overhead.values():
        assert entry["median_us"] > 0
        assert entry["iters"] == 3


# ---------------------------------------------------------------------------
# Landing checks
# ---------------------------------------------------------------------------


# expectations file for the host the suite is running on, and for the "other" host —
# the identity tests below must hold on both the GX10 (aarch64) and RunPod (x86_64)
_EXPECTATIONS_BY_MACHINE = {
    "aarch64": "configs/expectations_gx10.json",
    "x86_64": "configs/expectations_runpod.json",
}


def _expectations_for(this_host: bool) -> dict:
    machine = platform.machine()
    if machine not in _EXPECTATIONS_BY_MACHINE:
        pytest.skip(f"no expectations file for machine {machine!r}")
    name = _EXPECTATIONS_BY_MACHINE[machine] if this_host else next(
        p for m, p in _EXPECTATIONS_BY_MACHINE.items() if m != machine
    )
    return json.loads((REPO_ROOT / name).read_text())


def test_landing_checks_pass_with_this_hosts_expectations():
    expect = _expectations_for(this_host=True)
    if not torch.cuda.is_available():
        expect["require_cuda"] = False
    checks = run_landing_checks(expect, REPO_ROOT)
    failed = [c for c in checks if c["status"] == "FAIL"]
    assert not failed, f"unexpected failures on this machine: {failed}"
    assert checks_passed(checks)
    names = {c["check"] for c in checks}
    assert {"platform_machine", "vendor_model_pinned", "vendor_transformers_pinned",
            "no_weights_materialized", "stage_c_bitwise_gate"} <= names


def test_landing_checks_fail_cleanly_with_other_hosts_expectations():
    """The other host's expectations must FAIL on platform identity — as structured
    results, not exceptions — proving the same command works on both hosts."""
    expect = _expectations_for(this_host=False)
    if not torch.cuda.is_available():
        expect["require_cuda"] = False
    checks = run_landing_checks(expect, REPO_ROOT, tiny_gate=False)
    by_name = {c["check"]: c for c in checks}
    assert by_name["platform_machine"]["status"] == "FAIL"  # aarch64 vs x86_64
    assert not checks_passed(checks)
    # environment-independent checks still pass
    assert by_name["vendor_model_pinned"]["status"] == "PASS"
    assert by_name["no_weights_materialized"]["status"] == "PASS"


def test_source_pins_match_reproducibility_doc():
    pins = json.loads((REPO_ROOT / "configs/source_pins.json").read_text())
    doc = (REPO_ROOT / "docs/REPRODUCIBILITY.md").read_text()
    assert pins["model_sha"] in doc
    assert pins["transformers_sha"] in doc
