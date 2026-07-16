# Project Status

## Current phase

GX10 local development — **Tasks 01, 02, and 03 complete** (2026-07-16).

## Active task

Task 03 (`prompts/03_CALIBRATION_PLUMBING.md`) — done. Awaiting Task 04 definition
(expected: Stage-C actual-storage prototype, DGX plan Phase 5).

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

**Task 04 — Stage-C actual-storage prototype (DGX plan Phase 5):** custom cache layers that
really store FP8 (`float8_e4m3fn`) and packed FP4 (nibbles in `uint8`/`float4_e2m1fn_x2`
views) plus explicit e8m0/fp32 scale tensors, with pure-PyTorch dequantize-on-read;
non-RoPE/precise-slice split storage; honest memory accounting (values + scales + padding +
buffers + K=V aliasing + the sliding-layer V-duplication fix); correctness vs the Stage-B
QDQ baseline on tiny-model semantic and quality tests. No speed claims on GX10.
Then Task 05: benchmark harness + RunPod tooling (Phase 6).
