# GX10 Local Development Phase

## Goal

Leave the GX10 with a tested, documented, reproducible experiment framework that can move once to RunPod. The expensive cloud phase should begin with full-model loading and real calibration—not basic repository archaeology or ordinary Python debugging.

## Phase 0 — Preserve and inspect

- [x] Unpack the attached repository to `reference/v2_mla_poc/`.
- [x] Mark it read-only or enforce a no-edit policy.
- [x] Capture the GX10 environment.
- [x] Initialize Git and create the initial commit.
- [x] Clone official model and Transformers source without LFS weights.
- [x] Record exact revisions.

**Exit condition:** source and machine state are reproducible.

## Phase 1 — Architecture discovery

- [x] Trace sliding, CSA, HCA, compressor, indexer, and cache-state ownership.
- [x] Verify cache dimension order and RoPE slice at runtime.
- [x] Document write/read call paths and state lifetimes.
- [x] Compare official reference inference with Transformers behavior.
- [x] Identify generated versus canonical Transformers source.
- [x] Complete the reference port map.

**Exit condition:** every proposed quantization target has an exact source-level insertion point.

## Phase 2 — Tiny-model semantic baseline

Build a randomly initialized, reduced V4 configuration that exercises all relevant attention types.

Required tests:

- [x] full forward versus cached decode;
- [x] one-shot versus chunked prefill;
- [x] prefill/decode/prefill sequence;
- [x] cache reset and reuse;
- [x] compressor boundary behavior;
- [x] sliding-window rollover;
- [x] exact state shape and entry-count assertions;
- [x] deterministic repeated runs;
- [x] no-cache behavior.

**Exit condition:** baseline cache behavior is understood and protected by tests.

## Phase 3 — Official-policy QDQ simulation

Implement QDQ as a separate policy layer, not real storage.

Candidate policies:

```text
baseline_bf16
reference_official_qdq
main_fp8_nonrope_rope_bf16
main_fp4_nonrope_rope_bf16
mixed_group_policy
indexer_reference_qdq
```

- [x] Confirm identity when policy is disabled.
- [x] Confirm precise slice remains untouched.
- [x] Confirm QDQ occurs once at the intended write boundary.
- [x] Confirm chunked and incremental operation.
- [x] Measure tiny-model logit and loss differences.
- [x] Label all results as simulation/no memory saving.

**Exit condition:** numerical experiments can be selected entirely through configuration.

## Phase 4 — Calibration and precision policy plumbing

- [x] Define target taxonomy: layer type, layer index, cache state, group index.
- [x] Support contiguous groups of 32 and 64.
- [x] Collect activation ranges and quantization error.
- [x] Implement one-group empirical perturbation.
- [x] Implement logit KL and NLL delta.
- [x] Implement indexer top-k overlap/recall metrics.
- [ ] Optionally add gradient-weighted error ranking. *(deferred — optional; empirical perturbation is the primary truth, D-004)*
- [x] Save/load a versioned precision-policy JSON.
- [x] Separate calibration and held-out inputs.
- [x] Save exact token IDs and seeds.

**Exit condition:** a tiny-model smoke calibration produces a valid precision map and held-out evaluation.

## Phase 5 — Actual-storage prototype

Correctness prototype only; final performance remains deferred.

- [x] Separate non-RoPE and precise slices.
- [x] Store actual FP8 where supported, with explicit scales.
- [x] Prototype packed FP4 or byte-packed nibbles plus scales.
- [x] Dequantize on read in pure PyTorch.
- [x] Account for values, scales, padding, metadata, and temporary buffers.
- [x] Compare actual tensor bytes against the baseline.
- [x] Pass tiny-model semantic and quality tests.

**Exit condition:** at least one representation demonstrates real storage reduction in a tiny test without unsupported speed claims.

## Phase 6 — Benchmark and RunPod tooling

- [x] Create baseline/QDQ/storage benchmark CLIs.
- [x] Add warmup and CUDA synchronization.
- [x] Save per-trial and median results.
- [x] Capture peak allocated and reserved memory.
- [x] Support configurable prompt/token fixtures and context lengths.
- [x] Create source-only one-GPU RunPod landing test.
- [x] Create four-GPU full-model launch templates.
- [x] Ensure paths and hardware assumptions are configuration-driven.

**Exit condition:** the same command structure works locally with a tiny model and is ready for the full model.

## Local completion gate

All of the following must be true before renting four GPUs:

- [x] Architecture documentation complete.
- [x] Baseline semantic tests pass.
- [x] QDQ simulation tests pass.
- [x] Calibration smoke run completes.
- [x] Held-out tiny-model evaluation completes.
- [x] Actual-storage prototype is tested, or clearly deferred with a bounded implementation task.
- [x] Benchmark harness runs end to end.
- [x] No full checkpoint is present locally.
- [x] No uncommitted changes.
- [x] Source and dependency revisions frozen.
- [x] `docs/RUNPOD_HANDOFF_CHECKLIST.md` completed through the preflight section.
- [x] Git tag `dgx-phase-complete-v1` created.

## What not to optimize locally

- final tensor/model-parallel topology;
- RTX PRO 6000 throughput;
- SM120-only kernel tuning;
- full-model precision map;
- final long-context quality;
- production-serving integration such as vLLM.
