# Project Status

## Current phase

Architecture discovery and tiny-model baseline on GX10 — **Task 01 complete** (2026-07-16).

## Active task

Task 01 (`prompts/01_ARCHITECTURE_DISCOVERY.md`) — done. Awaiting Task 02 definition.

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

**Task 02 — official-policy QDQ simulation (Stage B):** implement policy-configurable
cache-layer subclasses reproducing the official numerics (FP8 e4m3 g64 ue8m0 non-RoPE main KV;
Hadamard+FP4 e2m1 g32 indexer KV + queries) on the tiny model. Identity test (policy off =
bit-exact), precise-slice untouched test, QDQ-once test, chunked/incremental equivalence,
tiny-model logit/NLL deltas + indexer top-k overlap. Simulation only — no memory-saving claims.
Scope details: `docs/QUANTIZATION_INJECTION_PLAN.md`.
