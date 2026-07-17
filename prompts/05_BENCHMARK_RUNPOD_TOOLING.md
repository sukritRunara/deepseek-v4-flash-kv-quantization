# Task 05 — Benchmark Harness and RunPod Tooling (DGX plan Phase 6)

## Objective

One command structure that benchmarks baseline / Stage-B QDQ / Stage-C storage caches on
the tiny model locally today and on the full checkpoint on RunPod later, entirely
configuration-driven. Plus the cheap one-GPU landing test and guarded launch scripts the
handoff checklist requires.

## Scope

1. `src/v4_kv_quant/bench.py` — benchmark engine:
   - identical fixed token streams across variants (same prompts/seeds, CLAUDE.md rule);
   - warmup runs (discarded) then N trials; CUDA synchronization around every timed span;
   - per-trial and median metrics: TTFT (prefill forward + first-token argmax), prefill
     throughput, per-token inter-token latency, decode throughput;
   - peak allocated/reserved memory per visible GPU (None on CPU);
   - actual cache bytes (logical + allocator) via Task-04 accounting at end of trial;
   - quantization/dequantization overhead micro-benchmarks on the real state shapes
     (fp8/fp4 encode of one decode step; decode of full window/compressed/indexer stores).
2. `tools/benchmark_cache.py` — CLI over a JSON config (`--config`) with flag overrides;
   `model: "tiny"` locally, `model_path` + `device_map` for the full checkpoint on RunPod;
   per-variant table + JSON with per-trial arrays. Banner: same-node comparisons only,
   tiny/GX10 numbers non-transferable (CLAUDE.md constraint 8).
3. `src/v4_kv_quant/landing.py` + `tools/runpod_landing_test.py` — source-only landing
   checks against an EXPECTATIONS file: platform arch, CUDA + GPU name + minimum compute
   capability, FP8/FP4 dtype availability, vendor checkouts at pinned SHAs
   (`configs/source_pins.json`), no weights materialized, python-dev headers (WARN),
   tiny-model forward, and the Stage-C-vs-Stage-B bitwise gate. `--run-suite` runs pytest.
   Same tool passes with `configs/expectations_gx10.json` locally and validates a RunPod
   pod with `configs/expectations_runpod.json`.
4. `configs/` — `bench_tiny_local.json`, `bench_runpod_4gpu.json`, `expectations_gx10.json`,
   `expectations_runpod.json`, `source_pins.json`. All paths/hardware assumptions live here.
5. `scripts/runpod/` — templates, all refusing to run on non-x86_64 hosts:
   - `setup_env.sh`: venv + torch (index URL configurable) + pinned vendor clones
     (source-only, `GIT_LFS_SKIP_SMUDGE=1`) + editable installs + landing test;
   - `download_model.sh`: weight download gated on `RUNPOD_ALLOW_WEIGHTS=1`, pinned
     revision, free-disk check — cannot run accidentally on the GX10;
   - `launch_4gpu_bench.sh`: full-model benchmark via `device_map=auto` over 4 GPUs
     using `configs/bench_runpod_4gpu.json`.
6. `tests/test_benchmark.py` — config loading/validation; a real 2-trial CPU benchmark of
   all three variants (structure, medians, storage bytes < baseline); micro-bench sanity;
   landing checks pass with GX10 expectations and fail cleanly with RunPod expectations
   on this machine.

## Acceptance gates

1. `python tools/benchmark_cache.py --config configs/bench_tiny_local.json` runs end to
   end locally (tiny model), producing per-trial + median JSON with all CLAUDE.md-required
   records (cache bytes incl. scales, peaks, prefill/TTFT/decode/ITL, QDQ overhead).
2. The identical CLI accepts `configs/bench_runpod_4gpu.json` (full-model paths) — only
   the config changes for RunPod.
3. `python tools/runpod_landing_test.py --expect configs/expectations_gx10.json` passes
   on this machine; the RunPod expectations file fails only on platform/GPU checks here.
4. Timing outputs carry the non-transferability label; no GX10-vs-target comparisons.
5. Guarded scripts refuse to execute on aarch64 / without explicit opt-in env vars.
6. Full test suite green.
