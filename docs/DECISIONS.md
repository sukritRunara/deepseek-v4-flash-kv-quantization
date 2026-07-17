# Decision Log

Record architecture and experiment decisions here. Do not leave consequential decisions only in chat or terminal history.

### D-001 — Promote starter pack to repository root

**Date:** 2026-07-16
**Status:** accepted (user-approved)
**Context:** The git repo initially held the starter pack and the unzipped reference repo as
sibling directories; CLAUDE.md/START_HERE.md expect starter contents at the project root with
the reference at `reference/v2_mla_poc/`.
**Decision:** Move starter contents to the repo root; move the reference repo to
`reference/v2_mla_poc` and `chmod -R a-w`.
**Alternatives considered:** keep starter as a subdirectory and nest everything inside it.
**Evidence:** commit `9b0825b`.
**Consequences:** CLAUDE.md auto-loads at the repo root; all documented paths are canonical.
**Follow-up:** none.

### D-002 — Project venv with cu130 SBSA torch + editable pinned transformers

**Date:** 2026-07-16
**Status:** accepted
**Context:** No system torch/transformers exist; system-wide changes are prohibited during
discovery; GB10 is aarch64 with CUDA 13 / cc 12.1.
**Decision:** `python3 -m venv .venv`; `torch 2.13.0+cu130` from the cu130 wheel index
(CUDA verified working on GB10); transformers installed **editable from the pinned
`vendor/transformers` checkout** so tests exercise exactly the pinned revision; `vendor/`
gitignored with SHAs recorded in `docs/REPRODUCIBILITY.md`.
**Alternatives considered:** CPU-only torch (kept as fallback, not needed); uv (not installed);
tracking vendor sources in-repo (bloat, nested-git issues).
**Evidence:** `docs/REPRODUCIBILITY.md`; `tools/hardware_smoke.py` results.
**Consequences:** ARM64 venv is non-portable — RunPod rebuilds from the recorded pins
(CLAUDE.md constraint 5).
**Follow-up:** freeze full `pip freeze` at DGX-phase completion.

### D-003 — Quantization injects as cache-layer subclasses, not attention patches

**Date:** 2026-07-16
**Status:** accepted
**Context:** All V4 cache writes flow through three methods on per-layer cache classes
(`update`, `store_compression_weights`, `update_compressor_states`); the reference PoC's
attention-forward monkey-patch pattern targets attributes V4 doesn't have; upstream marks
generic `QuantizedCache` incompatible.
**Decision:** Implement QDQ (Stage B) and real storage (Stage C) as subclasses of
`DeepseekV4CSACache` / `DeepseekV4HCACache` / `DynamicSlidingWindowLayer`, constructed from a
serializable precision policy; never edit generated modeling files. The single exception:
indexer *query* QDQ wraps `DeepseekV4Indexer` externally.
**Alternatives considered:** forward-patching attention (fragile, rejected); editing
`modular_deepseek_v4.py` + regenerating (upstream-invasive, rejected for experiments).
**Evidence:** `docs/V4_CACHE_ARCHITECTURE.md` §2; `docs/QUANTIZATION_INJECTION_PLAN.md`.
**Consequences:** experiments select behavior purely by constructing the cache; baseline path
untouched.
**Follow-up:** Task 02 implements the policy object + subclasses.

### D-004 — Indexer quantization judged by top-k overlap, not logit closeness

**Date:** 2026-07-16
**Status:** accepted
**Context:** Measured on the tiny model: near-tied indexer scores flip the top-k set under
~1e-7 numerical noise, producing ~1e-1 logit divergence between mathematically equivalent
computation paths; with a non-selective indexer the same paths agree to 2.4e-7.
**Decision:** Path-equality tests use a dense-indexer fixture (`index_topk` ≥ entry count);
indexer quantization quality is measured by top-k overlap/recall + downstream NLL/KL, matching
CLAUDE.md calibration principles.
**Alternatives considered:** loosening logit tolerances (hides real regressions); forcing
deterministic tie-breaking upstream (diverges from shipped behavior).
**Evidence:** isolation experiment in `docs/V4_CACHE_ARCHITECTURE.md` §6.8; test-file docstring.
**Consequences:** Task 02 metrics suite must include a top-k overlap harness.
**Follow-up:** validate flip rates on real weights on RunPod (random weights overstate ties).

### D-005 — Do not install python3.12-dev during discovery; defer Triton/torch.compile

**Date:** 2026-07-16
**Status:** accepted
**Context:** Triton and inductor fail at gcc because `/usr/include/python3.12/Python.h` is
missing (system package). Installing it is a system-wide change, prohibited by CLAUDE.md
constraint 4 / task instructions; Stages A–B are pure PyTorch by design.
**Decision:** Record as documented limitation (`docs/REPRODUCIBILITY.md`,
`tools/hardware_smoke.py` report) and proceed without it.
**Alternatives considered:** `sudo apt install python3.12-dev` (deferred until a Triton
prototype actually needs it, with explicit sign-off).
**Evidence:** manual gcc reproduction; hardware_smoke UNSUPPORTED entries with root cause.
**Consequences:** no fused-kernel prototyping on GX10 until resolved; none for Tasks 01–02.
**Follow-up:** revisit before any Stage-D local prototyping.

### D-006 — Indexer QDQ stores keys in the Hadamard-rotated basis; queries rotated via scorer wrapper

**Date:** 2026-07-16
**Status:** accepted
**Context:** The official indexer path (model.py:368-370, 414-420) Hadamard-rotates both the
compressed indexer keys and the queries before FP4 QDQ, and scores in the rotated space
(orthonormal rotation preserves dot products; its purpose is outlier-spreading before FP4).
The only HF module that sees post-RoPE queries is `DeepseekV4IndexerScorer`.
**Decision:** `QDQCSACacheLayer` stores indexer entries rotated+FP4-QDQ'd (official-faithful);
`indexer_query_qdq` context manager swaps the scorer for a wrapper that rotates+QDQs queries
symmetrically. No edit to `DeepseekV4Indexer.forward`.
**Alternatives considered:** rotate→QDQ→inverse-rotate before storing (keeps original basis
but diverges from official numerics); editing the modular file (rejected, upstream-invasive).
**Evidence:** `test_layer_indexer_write_rotated_fp4`, `test_hadamard_properties` (dot-product
preservation), `test_indexer_policy_end_to_end` (scorer restored).
**Consequences:** baseline and QDQ runs must not share a live scorer swap; the context
manager guarantees restoration.
**Follow-up:** compare pure-PyTorch FWHT vs `fast_hadamard_transform` numerics on RunPod.

### D-007 — Software e2m1 rounding (RNE) instead of native FP4 cast

**Date:** 2026-07-16
**Status:** accepted
**Context:** torch 2.13 cannot cast to `float4_e2m1fn_x2` ("copy_kernel not implemented" —
probed); the official kernel's `T.Cast(FP4, …)` rounds to nearest-even on the e2m1 grid.
**Decision:** implement the e2m1 grid in software with explicit ties-to-even midpoint table;
verified idempotent, grid-exact, and tie-correct by unit tests.
**Alternatives considered:** nearest-value without tie rule (reference PoC behavior —
diverges from hardware on exact midpoints); waiting for native cast support.
**Evidence:** `tests/test_qdq_simulation.py::test_fp4_tie_rounding_is_nearest_even`.
**Consequences:** bit-exact parity with the official tilelang kernel on midpoints is expected
but must be spot-checked on RunPod where tilelang runs.
**Follow-up:** RunPod cross-check kernel-vs-software QDQ on identical inputs.

### D-008 — Indexer calibrated as one state-level target, not per-group

**Date:** 2026-07-16
**Status:** accepted
**Context:** Main-KV states quantize per contiguous channel group (official scale groups of
64), so per-group sensitivity maps directly onto deployable storage decisions. The indexer
path Hadamard-rotates the whole 128-dim vector before FP4 quantization; a channel group in
the original basis has no independent meaning after rotation, and packed FP4 storage would
quantize the whole rotated vector regardless.
**Decision:** `enumerate_targets` emits exactly one whole-vector `indexer_kv` target per CSA
layer; `PrecisionMap.validate` rejects partial indexer coverage. Indexer sensitivity is
judged by top-k overlap/recall (D-004) plus ΔNLL/KL.
**Alternatives considered:** per-group targets in the rotated basis (measurable but not
independently deployable; rejected as misleading granularity).
**Evidence:** `src/v4_kv_quant/targets.py` docstring; `test_precision_map_validation`
(partial indexer coverage rejected); smoke run showing the indexer as the dominant
sensitivity target.
**Consequences:** indexer precision decisions are per-layer on/off; finer control would
require changing the official rotation scheme itself.
**Follow-up:** revisit only if RunPod results show per-layer on/off is too coarse.

### D-009 — Stage-C correctness gate: bitwise equivalence with the Stage-B QDQ cache

**Date:** 2026-07-17
**Status:** accepted
**Context:** Stage C changes the storage representation (FP8 codes + e8m0 scales, packed
FP4 nibbles) but must not change numerics: the values attention consumes should be exactly
the values the validated Stage-B simulation produced, or Stage-B quality results would not
transfer to real storage.
**Decision:** The storage primitives carry a hard contract `load(store(x)) == qdq(x)`
(bitwise), and the model-level gate asserts `QuantizedStorageCache(policy)` produces
logits and indexer picks bitwise-identical to `QDQCache(policy)`. FP4 codes are stored as
sign + 3-bit magnitude index, two per uint8 (layout-compatible with a future
`float4_e2m1fn_x2` view); scales as `float8_e8m0fnu` when power-of-2, fp32 otherwise.
**Alternatives considered:** tolerance-based equivalence (hides representation bugs);
independent Stage-C numerics (would require re-running all quality experiments).
**Evidence:** `tests/test_actual_storage.py` (`test_*_matches_qdq`,
`test_storage_cache_bitwise_equals_qdq_cache`); `results/cache_memory.json` (0.438x
logical bytes on the fp32 tiny model; Stage-B exactly 1.000x).
**Consequences:** any future storage-format change must preserve the bitwise contract or
explicitly re-run the Stage-B quality suite.
**Follow-up:** revalidate the contract on CUDA/BF16 pipelines during RunPod bring-up.

### D-010 — CPU-only local execution; expectation-driven landing severity for CUDA readiness

**Date:** 2026-07-17
**Status:** accepted
**Context:** First CUDA forward of the tiny model revealed that this torch 2.13 build
dispatches CUDA `torch.bmm` (used by V4's grouped output projection) to a Triton-backed
`torch._native` kernel, whose launcher stubs compile against Python.h — absent without the
`python3.12-dev` system package. Installing system packages remains off-limits without
explicit owner sign-off (D-005).
**Decision:** All local development runs on CPU (`bench_tiny_local.json` pins `device:
"cpu"`). The landing test gains a `cuda_model_forward` check whose severity comes from the
expectations file: WARN on the GX10 (`require_cuda_model_forward: false`), FAIL on RunPod
(`true`, plus `require_python_dev: true`), so a mis-provisioned pod is caught before four
GPUs are rented.
**Alternatives considered:** installing python3.12-dev now (prohibited without sign-off);
forcing a non-Triton bmm path (no supported switch in this torch build).
**Evidence:** traceback through `torch/_native/ops/bmm_outer_product/triton_impl.py`;
`tests/test_benchmark.py::test_landing_checks_*`; REPRODUCIBILITY.md limitation 1.
**Consequences:** no GPU-side numbers from the GX10 at all (they were non-transferable
anyway); CPU covers every correctness gate in the suite.
**Follow-up:** if GX10 GPU validation is ever wanted, install python3.12-dev with owner
sign-off and re-run tools/hardware_smoke.py + the landing test.
