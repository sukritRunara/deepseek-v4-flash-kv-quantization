# Project Status

## Current phase

GX10 local development — **Task 01 and Task 02 complete** (2026-07-16).

## Active task

Task 02 (`prompts/02_OFFICIAL_POLICY_QDQ_SIMULATION.md`) — done. Awaiting Task 03 definition
(expected: calibration/sensitivity plumbing, DGX plan Phase 4).

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

**Task 03 — calibration and precision-policy plumbing (DGX plan Phase 4):** target taxonomy
(layer, layer type, state, contiguous group of 32/64), activation-range + quantization-error
collection, one-group empirical perturbation (ΔNLL, logit KL) using the Task-02 harness,
indexer top-k overlap/recall as the indexer metric, versioned precision-map JSON produced by
a tiny-model smoke calibration with separated calibration/held-out token sets. The Stage-C
actual-storage prototype (packed FP8/FP4 + scales, real memory accounting) can follow or run
in parallel — see `docs/QUANTIZATION_INJECTION_PLAN.md` and `docs/DGX_PHASE_PLAN.md`.
