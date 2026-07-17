# Project Status

## Current phase

GX10 local development — **Tasks 01–05 complete** (Tasks 01–03: 2026-07-16; Tasks 04–05: 2026-07-17).

## Active task

Task 05 (`prompts/05_BENCHMARK_RUNPOD_TOOLING.md`) — done. Remaining: the local completion
gate itself (freeze revisions, results manifest, commit/tag `dgx-phase-complete-v1`,
handoff checklist preflight).

## Checklist (Task 05)

- [x] Benchmark engine (`src/v4_kv_quant/bench.py`): identical token streams across
      variants, warmup + N trials, CUDA sync, per-trial + median TTFT / prefill / decode /
      ITL, peak alloc/reserved per GPU, actual cache bytes, QDQ overhead micro-bench
- [x] `tools/benchmark_cache.py` — config-driven; SAME command for tiny local
      (`configs/bench_tiny_local.json`, runs end to end) and full model
      (`configs/bench_runpod_4gpu.json`, schema-validated); non-transferability labeled
- [x] Landing test (`src/v4_kv_quant/landing.py` + `tools/runpod_landing_test.py`):
      expectation-driven checks (platform, GPU/CC, dtypes, pinned vendor SHAs, no weights,
      python-dev, tiny forward, Stage-C bitwise gate, optional pytest suite);
      GX10 expectations PASS (exit 0), RunPod expectations FAIL correctly here (exit 1)
- [x] `configs/source_pins.json` + expectations files; pins cross-checked against
      REPRODUCIBILITY.md by test
- [x] `scripts/runpod/{setup_env,download_model,launch_4gpu_bench}.sh` — all refuse on
      non-x86_64; weight download additionally gated on RUNPOD_ALLOW_WEIGHTS=1 + disk check
- [x] Full suite green: 90 passed

## New finding (2026-07-17)

**GX10 CUDA model execution is blocked**: this torch build routes `torch.bmm` (used by V4's
grouped output projection) through a Triton kernel, and Triton cannot compile its launcher
stubs without `python3.12-dev`. All local work runs on CPU (unaffected); the landing test
records this as WARN locally and would FAIL a RunPod pod missing dev headers.
Details: `docs/REPRODUCIBILITY.md` limitation 1.

## Checklist (Task 04)

- [x] Storage primitives with bitwise contract `load(store(x)) == qdq(x)`
      (`src/v4_kv_quant/storage.py`): FP8 e4m3 codes + e8m0/fp32 scales; packed e2m1
      nibbles (sign + 3-bit index, two per uint8) + e8m0 scales
- [x] `QuantizedStorageCache` (`src/v4_kv_quant/storage_cache.py`): quantized tensors are
      the only persistent KV storage (keys/values stay empty placeholders); window
      append+trim re-contiguated (no hidden history); append-only compressed/indexer
      stores; entry_count/cumulative_length bookkeeping intact; buffers/overlap stock BF16;
      sliding-layer V duplication removed
- [x] **Model-level bitwise equivalence: storage cache == Stage-B QDQ cache** for
      `main_fp8_nonrope_rope_bf16` and `reference_official_qdq` (logits and indexer picks)
- [x] Honest memory accounting (`src/v4_kv_quant/memory.py`): logical + allocator storage
      bytes, K=V alias counted once, stock V-duplication flagged, scales itemized
- [x] `tools/measure_cache_memory.py`: Stage-B == 1.000x baseline (simulation saves
      nothing, proven); Stage-C == 0.438x on the fp32 tiny model (real reduction);
      per-layer/per-state itemization; labeled no-speed-claims
- [x] Full suite green: 82 passed

## Checklist (Task 03)

- [x] Target taxonomy `(layer, layer_type, state, contiguous group)`; indexer as one
      state-level target per CSA layer (`src/v4_kv_quant/targets.py`, decision D-008)
- [x] Versioned per-group `PrecisionMap` + validation + JSON (`src/v4_kv_quant/precision_map.py`)
- [x] `MappedQDQCache` map consumer; single-entry map = perturbation experiment; empty map
      bit-exact; full-coverage map == Task-02 policy cache bitwise (`src/v4_kv_quant/mapped_cache.py`)
- [x] Pass-through activation-stats collector, bit-exact, per-group amax + QDQ-error RMS
      (`src/v4_kv_quant/stats.py`)
- [x] One-target empirical perturbation sweep + ranking + map builder with explicit
      fractions/thresholds (`src/v4_kv_quant/sensitivity.py`); gradient-weighted ranking
      deferred (optional per CLAUDE.md)
- [x] Smoke calibration end to end (`tools/run_calibration_smoke.py`): 16 targets, map
      built + validated, held-out eval, token ids/seeds/provenance saved
- [x] Full suite green: 67 passed

## Checklist (Task 02)

- [x] QDQ primitives matching official kernels (`src/v4_kv_quant/qdq.py`): FP8 e4m3 g64
      ue8m0 round-up scales (exact via frexp), software FP4 e2m1 RNE g32, orthonormal FWHT
- [x] Versioned policy object + 5 named presets, JSON round trip (`src/v4_kv_quant/policy.py`)
- [x] QDQ cache-layer subclasses + indexer-query scorer wrapper (`src/v4_kv_quant/qdq_cache.py`);
      registry not hijacked (`_layer_type = None`, test-pinned)
- [x] Metrics + teacher-forced harness (`src/v4_kv_quant/{metrics,harness}.py`)
- [x] `tools/run_qdq_simulation.py` runs all policies; results labeled simulation/no savings
- [x] Acceptance gates test-pinned (`tests/test_qdq_simulation.py`, 27 tests): identity
      bit-exact, RoPE slice untouched, QDQ-once, chunked/decode==one-shot under QDQ,
      Task-01 invariants preserved
- [x] Full suite green: 54 passed

## Checklist (Task 01)

- [x] Environment captured (`artifacts/env/…20260716T222305Z` pre-venv, `…222905Z` post-venv)
- [x] Reference archive unpacked read-only (`reference/v2_mla_poc`, chmod a-w)
- [x] Official source repositories cloned without weights (14 MB, LFS pointers verified)
- [x] Source revisions pinned (`docs/REPRODUCIBILITY.md`: model `60d8d70`, transformers `150eb7c9`)
- [x] V4 cache architecture documented (`docs/V4_CACHE_ARCHITECTURE.md`)
- [x] Reference port map documented (`docs/REFERENCE_PORT_MAP.md`)
- [x] Tiny-model inspection tool created (`tools/inspect_v4_cache.py`, all runtime assertions pass)
- [x] Baseline cache semantic tests pass (`tests/test_v4_cache_semantics.py`: 27 passed)
- [x] Hardware/dtype smoke tool passes or limitations documented (`tools/hardware_smoke.py`:
      all required PASS; triton/torch.compile UNSUPPORTED — python3.12-dev missing, documented)
- [x] Quantization injection plan documented (`docs/QUANTIZATION_INJECTION_PLAN.md`)

## Key findings

- V4 is **not** MLA: shared-KV MQA, K=V single 512-dim vector `[nope 448 | rope 64]`,
  inverse-RoPE on attention output; sliding window (128) on every layer + CSA (m=4, indexer
  top-k 512) / HCA (m'=128) compressed streams appended on the KV axis.
- Official QDQ policy (QAT-aligned): FP8 e4m3 groups-of-64 with ue8m0 round-up scales on
  non-RoPE main KV; Hadamard + FP4 e2m1 groups-of-32 on indexer keys *and* queries; RoPE dims
  and compressor buffers never quantized. Transformers implementation has no QDQ.
- Measured: CSA indexer top-k selection flips under ~1e-7 noise when scores are near-tied →
  indexer quantization must be judged by top-k overlap/recall, not logit closeness.
- All quantization writes flow through 3 cache-layer methods → quantized cache = cache-layer
  subclasses; no attention/modeling edits needed (except indexer-query QDQ wrapper).

## Blockers

None. (Triton/torch.compile unavailable without `python3.12-dev` — irrelevant for Task 02,
documented in `docs/REPRODUCIBILITY.md`.)

## Next task

**Local completion gate (`docs/DGX_PHASE_PLAN.md`):** walk the gate checklist — freeze
dependency/source revisions (pip freeze into docs), generate a local results manifest,
confirm no uncommitted changes / no weights, complete the handoff checklist preflight
section, tag `dgx-phase-complete-v1`. After that, work moves to RunPod
(`docs/RUNPOD_HANDOFF_CHECKLIST.md` execution order).
