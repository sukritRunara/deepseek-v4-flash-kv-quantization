# GX10 Local Development Phase

## Goal

Leave the GX10 with a tested, documented, reproducible experiment framework that can move once to RunPod. The expensive cloud phase should begin with full-model loading and real calibration—not basic repository archaeology or ordinary Python debugging.

## Phase 0 — Preserve and inspect

- [ ] Unpack the attached repository to `reference/v2_mla_poc/`.
- [ ] Mark it read-only or enforce a no-edit policy.
- [ ] Capture the GX10 environment.
- [ ] Initialize Git and create the initial commit.
- [ ] Clone official model and Transformers source without LFS weights.
- [ ] Record exact revisions.

**Exit condition:** source and machine state are reproducible.

## Phase 1 — Architecture discovery

- [ ] Trace sliding, CSA, HCA, compressor, indexer, and cache-state ownership.
- [ ] Verify cache dimension order and RoPE slice at runtime.
- [ ] Document write/read call paths and state lifetimes.
- [ ] Compare official reference inference with Transformers behavior.
- [ ] Identify generated versus canonical Transformers source.
- [ ] Complete the reference port map.

**Exit condition:** every proposed quantization target has an exact source-level insertion point.

## Phase 2 — Tiny-model semantic baseline

Build a randomly initialized, reduced V4 configuration that exercises all relevant attention types.

Required tests:

- [ ] full forward versus cached decode;
- [ ] one-shot versus chunked prefill;
- [ ] prefill/decode/prefill sequence;
- [ ] cache reset and reuse;
- [ ] compressor boundary behavior;
- [ ] sliding-window rollover;
- [ ] exact state shape and entry-count assertions;
- [ ] deterministic repeated runs;
- [ ] no-cache behavior.

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

- [ ] Confirm identity when policy is disabled.
- [ ] Confirm precise slice remains untouched.
- [ ] Confirm QDQ occurs once at the intended write boundary.
- [ ] Confirm chunked and incremental operation.
- [ ] Measure tiny-model logit and loss differences.
- [ ] Label all results as simulation/no memory saving.

**Exit condition:** numerical experiments can be selected entirely through configuration.

## Phase 4 — Calibration and precision policy plumbing

- [ ] Define target taxonomy: layer type, layer index, cache state, group index.
- [ ] Support contiguous groups of 32 and 64.
- [ ] Collect activation ranges and quantization error.
- [ ] Implement one-group empirical perturbation.
- [ ] Implement logit KL and NLL delta.
- [ ] Implement indexer top-k overlap/recall metrics.
- [ ] Optionally add gradient-weighted error ranking.
- [ ] Save/load a versioned precision-policy JSON.
- [ ] Separate calibration and held-out inputs.
- [ ] Save exact token IDs and seeds.

**Exit condition:** a tiny-model smoke calibration produces a valid precision map and held-out evaluation.

## Phase 5 — Actual-storage prototype

Correctness prototype only; final performance remains deferred.

- [ ] Separate non-RoPE and precise slices.
- [ ] Store actual FP8 where supported, with explicit scales.
- [ ] Prototype packed FP4 or byte-packed nibbles plus scales.
- [ ] Dequantize on read in pure PyTorch.
- [ ] Account for values, scales, padding, metadata, and temporary buffers.
- [ ] Compare actual tensor bytes against the baseline.
- [ ] Pass tiny-model semantic and quality tests.

**Exit condition:** at least one representation demonstrates real storage reduction in a tiny test without unsupported speed claims.

## Phase 6 — Benchmark and RunPod tooling

- [ ] Create baseline/QDQ/storage benchmark CLIs.
- [ ] Add warmup and CUDA synchronization.
- [ ] Save per-trial and median results.
- [ ] Capture peak allocated and reserved memory.
- [ ] Support configurable prompt/token fixtures and context lengths.
- [ ] Create source-only one-GPU RunPod landing test.
- [ ] Create four-GPU full-model launch templates.
- [ ] Ensure paths and hardware assumptions are configuration-driven.

**Exit condition:** the same command structure works locally with a tiny model and is ready for the full model.

## Local completion gate

All of the following must be true before renting four GPUs:

- [ ] Architecture documentation complete.
- [ ] Baseline semantic tests pass.
- [ ] QDQ simulation tests pass.
- [ ] Calibration smoke run completes.
- [ ] Held-out tiny-model evaluation completes.
- [ ] Actual-storage prototype is tested, or clearly deferred with a bounded implementation task.
- [ ] Benchmark harness runs end to end.
- [ ] No full checkpoint is present locally.
- [ ] No uncommitted changes.
- [ ] Source and dependency revisions frozen.
- [ ] `docs/RUNPOD_HANDOFF_CHECKLIST.md` completed through the preflight section.
- [ ] Git tag `dgx-phase-complete-v1` created.

## What not to optimize locally

- final tensor/model-parallel topology;
- RTX PRO 6000 throughput;
- SM120-only kernel tuning;
- full-model precision map;
- final long-context quality;
- production-serving integration such as vLLM.
