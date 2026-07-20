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

### D-015 — Owner delegates the B4 map decision for the overnight run (provisional)

**Date:** 2026-07-20
**Status:** accepted (owner-approved "Go", 2026-07-20; **map RATIFIED by owner
2026-07-20** — `moderate`: all main-KV non-RoPE FP8 e4m3, RoPE BF16, indexer BF16
is the project's calibrated precision map)
**Context:** The B4 plan reserved a stop point before refine/map for owner review of
rankings (D-012). Owner is asleep and explicitly delegated: run the full remaining
workflow overnight, with the agent choosing "something good" under a pre-stated rule.
Screening (run 1) also surfaced two design gaps that the delegated run must handle:
(1) **indexer screening at the 2k probe is vacuous** — `index_topk=512` ≥ compressed
entries at 2048 tokens (ratio 4), so top-k selection never binds and every indexer
target measures exactly zero with overlap 1.0 by construction; (2) **the map stage as
committed composes only refine records**, so the ~69 least-sensitive states (never
refined) would get no entry and stay BF16 — inverted for a deployable map.
**Decision:** (a) Re-measure the 21 indexer targets on an 8k probe (2048 entries ≫
topk 512, selection real) — new `indexer8k` stage; requires chunked prefill in the
teacher-forced harness (one-shot 8192 OOMs — same class as the stats-stage fix).
(b) Compose the map from refine records (group-64, top-15 states) + screening records
for non-refined states (state-level) + indexer8k records; fractions apply over that
combined ranked pool. (c) Build three candidates — conservative (fp8 0.75/fp4 0.0),
moderate (fp8 1.0/fp4 0.0), aggressive (fp8 0.85/fp4 0.15) — evaluate all on held-out
2k+8k, and select by guardrails: the chosen map must match or beat the official QDQ
policy on ΔNLL, KL, and top-1 agreement; indexer entries only for layers with 8k
mean top-k overlap ≥ 0.9 (D-012). If no candidate passes, fall back to the official
policy downstream and leave the map decision for the owner. (d) 32k held-out
spot-check runs on the selected map (D-012).
**Alternatives considered:** trusting the 2k indexer screening (rejected — measurement
does not measure anything at that length); refine-only map composition as committed
(rejected — leaves the insensitive majority unquantized); waiting for owner review
(explicitly overridden by owner for this run).
**Evidence:** `results/calibration_full/screening.json` (21/21 indexer targets score
exactly 0, overlap 1.0); model config `index_topk=512`, `compress_ratios` CSA=4;
WORKLOG 2026-07-20 B4 run 1.
**Consequences:** the shipped `precision_map.json` is PROVISIONAL until the owner
ratifies fractions and indexer gating; all candidates + heldout metrics are preserved
for that review.
**Follow-up:** owner morning review; revisit sweep breadth if refine shows flat
sensitivity (D-012 follow-up stands).

### D-014 — GCP G4 host P2P is healthy; host-staged workaround becomes opt-in

**Date:** 2026-07-20
**Status:** accepted
**Context:** D-013 made the D-011 workaround "conditional on `tools/p2p_stress_check.py`
results on the new host." On the GCP `g4-standard-48` (`deepseek-v4-flash-g4-4gpu`,
driver 610.43.02), the stress check passes natively: 0 corrupt transfers across all 12
ordered pairs in both phases (plain + concurrent-compute, 64 MiB × 20 iters) — the
RunPod fault was host-specific, as diagnosed.
**Decision:** `ensure_host_staged_p2p()` is now OPT-IN via `V4_KV_FORCE_HOST_STAGED_P2P=1`,
gated inside the function (env check before any CUDA interaction) so all call sites stay
unconditional. `tools/p2p_stress_check.py --workaround` arms it itself. Default on this
host: native P2P.
**Alternatives considered:** removing the workaround entirely (rejected — future hosts
may be faulty and the tool remains the health gate); per-call-site gating (rejected —
one central gate is harder to bypass accidentally).
**Evidence:** stress-check run 2026-07-20 (WORKLOG "GCP bring-up"); `tests/test_p2p_workaround.py`
pins the gate order.
**Consequences:** GCP timing numbers reflect direct PCIe P2P; the "host-staged D2D"
caveat on all RunPod Phase-B numbers does NOT apply to GCP results (one more reason
never to compare across the two nodes). On any future host, run the stress check first
and set the env var if it fails.
**Follow-up:** none.

### D-013 — Migrate Phase B from RunPod to GCP G4 (same GPU) — owner-approved

**Date:** 2026-07-17
**Status:** accepted (owner-approved)
**Context:** Owner has GCP credits; GCP G4 VMs ship the exact same GPU as the RunPod
pod (RTX PRO 6000 Blackwell Server Edition, SM120); the RunPod host has faulty PCIe
P2P (D-011).
**Decision:** Seal the RunPod pod after B3 (evidence in `artifacts/phase_b_runpod/`),
resume from B4 run 1 on a `g4-standard-48`. Bring-up checklist: `docs/GCP_TRANSITION.md`.
P2P workaround becomes conditional on `tools/p2p_stress_check.py` results on the new
host. All timing numbers regenerate on the GCP node (B7 covers baseline); bitwise
gates, calibration design, and code transfer unchanged.
**Consequences:** B2's RunPod numbers are historical record only; CLAUDE.md's "final
hardware is one RunPod node" is superseded by same-silicon GCP equivalence.
**Follow-up:** run the stress check before any multi-GPU work on the new instance.

### D-012 — B4 calibration design (owner-approved 2026-07-17)

**Date:** 2026-07-17
**Status:** accepted (owner-approved)
**Context:** Real-model calibration needs a corpus, sequence lengths, and a sweep plan;
a full per-group sweep (~580 targets × 2 formats) costs hours of 4-GPU time.
**Decision:** (1) Corpus: C4-English streaming (~80%) + code slice (~20%,
the-stack-smol), seeded, unpadded, non-overlapping windows, token ids saved; held-out
from disjoint stream regions. (2) Lengths: calibrate at 2k (32 seqs) + 8k (8 seqs) —
long enough to exercise CSA/HCA streams far beyond the 128 window; held-out eval at
2k/8k + one 32k spot-check. (3) Sweep: two-stage — FP4 state-level screening
(group_size = full nope width → ~100 targets) over a fixed [4, 2048] probe batch, then
group-level FP4 within the most sensitive states; FP8 spot-checks only (QAT-aligned FP8
expected near-lossless). (4) Ranking: ΔNLL + KL; indexer via top-k overlap
(D-004/D-008); `build_map_from_sweep` fractions chosen after inspecting distributions.
**Consequences:** rankings come from a probe subset (stats pass uses the full set);
documented in the map provenance. `datasets` added to the environment (setup_env.sh).
**Follow-up:** revisit sweep breadth if screening shows flat sensitivity.

### D-011 — Phase-B pod P2P is faulty; all multi-GPU runs disable CUDA peer access

**Date:** 2026-07-17
**Status:** accepted
**Context:** B1 baseline generation on the 4-GPU pod produced degenerate output
(all-BOS tokens / NaN logits / 1e36 activations, different on every run). A long
elimination (see WORKLOG "Phase B — B1 investigation") cleared the FP8/FP4 Triton
kernels (bit-accurate vs dequant reference across the full autotune config space, all
V4 shapes, NaN-poisoned allocator), the checkpoint bytes, the loader (GPU buffers
byte-match shards), and the mask/mHC/model code. A direct stress test then showed the
platform fault: **direct GPU-to-GPU copies silently corrupt** — 15-30/30 failures at
64-256 MiB, and 20/20 on every ordered pair when a copy overlaps compute on either
device (exactly the `device_map="auto"` pipeline regime). D2H/H2D are clean. This is a
host PCIe ACS/IOMMU misconfiguration, invisible to `nvidia-smi`.
**Decision:** `v4_kv_quant.p2p_workaround.ensure_host_staged_p2p()` (trigger torch's
lazy peer enablement, then `cudaDeviceDisablePeerAccess` on every pair, forcing host
staging) is called at the start of every multi-GPU run. Pod health is checked with
`tools/p2p_stress_check.py` (0/240 corrupt with the workaround vs ~100% without under
concurrent compute).
**Alternatives considered:** requesting a replacement pod (right long-term answer for
the final benchmark matrix — host-staged copies change inter-GPU latency — but
correctness work proceeds now; re-run the stress check on any new pod first);
single-GPU-per-process tensor parallelism (larger rework, defers the whole plan).
**Evidence:** scratchpad stress logs 2026-07-17; `tools/p2p_stress_check.py`;
mitigation-validated rerun of B1 generation.
**Consequences:** all Phase-B numbers carry a "host-staged D2D" caveat; benchmark
comparisons remain valid (same node, same handicap on all variants) but absolute
inter-GPU transfer costs are not representative of a healthy node. Report the fault to
RunPod; prefer a validated-healthy node for the final benchmark matrix (B7).
**Follow-up:** run `tools/p2p_stress_check.py` on every future pod before use.

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
