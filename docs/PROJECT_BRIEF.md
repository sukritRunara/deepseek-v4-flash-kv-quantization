# Project Brief — DeepSeek-V4-Flash KV-Cache Quantization

*Written 2026-07-17, at the RunPod→GCP transition (B0–B3 complete, B4 designed and
ready). Audience: anyone who wants the story and the outlook without reading the
worklogs. A paste-ready short version is at the end. Deep references:
`docs/ENGINEERING_REPORT.md` (local-phase narrative), `docs/V4_CACHE_ARCHITECTURE.md`,
`docs/DECISIONS.md` (D-001…D-013), `docs/WORKLOG.md`, `docs/RUNPOD_PHASE_B_PLAN.md`.*

---

## What this project is

DeepSeek-V4-Flash is a 1M-context model. At long context, the memory that dominates
serving is not the weights — it's the **KV cache**, the per-token state kept per
*sequence*. This project asks: how small can V4's cache get (FP8, FP4, per-group mixed
precision) before output quality degrades — measured honestly, with real bytes and
real benchmarks?

The twist that makes it interesting: **DeepSeek already quantizes this cache** in
their own inference stack (FP8 on content channels, Hadamard+FP4 on the attention
indexer — and the model was *trained* with this in the loop). The Hugging Face
implementation — the only one most people can run — has **none of that**: pure BF16.
We reproduce the official recipe faithfully inside the HF stack, verify it, then
explore beyond it (FP4 on the main cache, per-group mixed precision).

## Why it matters (the arithmetic)

Measured on the real checkpoint (4× RTX PRO 6000 Blackwell, 2026-07-17): the BF16
cache costs **437 MiB per sequence at 65k context**, scaling ≈linearly — call it
~7 GiB/sequence at 1M context. Our real low-precision storage (measured, not
projected) roughly halves the quantized slices. Per sequence that's ~3–4 GiB at 1M;
across a serving batch of 16–32 long-context streams it's **50–100+ GiB** — the
difference between fitting a batch tier and not. A deferred speed win rides along:
V4's decode reads ≤512 indexer-selected entries regardless of context length, so a
fused selective-dequant kernel (Stage D) makes the read path cheap.

## What's done (and how sure we are)

**Local phase (ASUS GX10, complete, tagged `dgx-phase-complete-v1`):** architecture
discovery (V4 is *not* MLA — shared-KV MQA + sliding window + two compressed streams +
a learned top-k indexer), exact software reimplementation of the official quantization
numerics, calibration machinery, real packed storage, benchmark harness — 90+ tests,
every gate bitwise where physically possible.

**Full-model phase (Phase B, steps B0–B3 complete on a rented 4-GPU node):**
- **B1 — GO:** the native FP8/FP4 checkpoint runs correctly on RTX PRO 6000
  (SM120) across 4 GPUs; coherent generation verified.
- **B2:** baseline memory/timing matrix recorded (1k/8k/65k context).
- **B3 — the load-bearing result:** our instrumented cache is **bitwise identical**
  to stock when quantization is off, and real packed storage is **bitwise identical**
  to the simulation path under the official policy — on the real model. Every quality
  number we measure in simulation therefore transfers to real storage exactly.

Bring-up also surfaced three genuine environment bugs, all root-caused and neutralized
(a host with silently-corrupting GPU-to-GPU transfers; a PyTorch allocator mode that
fires spurious asserts; a `torch.ldexp` missing-device-guard bug in torch 2.13 — fixed
in our code, reportable upstream). None affect results; all are documented with
repros.

## What to expect

- **Official recipe (FP8 main + FP4 indexer): expected near-lossless.** It's
  QAT-aligned — the model was trained with it. Tiny-model results ordered exactly as
  theory predicts; the real-model divergence signal so far is dominated by indexer
  top-k tie flips, the failure mode we built dedicated metrics for.
- **The open questions (where the novelty lives):** does FP4 on the *main* KV hold on
  real weights, and how much does per-group mixed precision buy? That's B4/B5 —
  calibration design is owner-approved (D-012) and the tooling is committed.
- **Honest caveats:** held-out quality numbers don't exist yet (B5); the pure-PyTorch
  dequant read path is slower than baseline until Stage-D kernel fusion (expected,
  deferred); benchmark timings are same-node comparisons by rule.

## What's left

B4 (real-model calibration + precision map) → B5 (quality on held-out data) → B6
(storage runs) → B7 (final benchmark matrix). Executing on GCP `g4-standard-48` (same
GPU silicon; migration checklist in `docs/GCP_TRANSITION.md`). Estimated remaining
compute: about one day, mostly unattended (contingency only if a new
environment surprise appears — the known ones all have baked-in fixes and health
checks now).

---

## Short version (paste-ready)

> **DeepSeek-V4-Flash KV-cache quantization — status.** V4's official inference stack
> quantizes its KV cache (FP8/FP4, trained-in); the Hugging Face implementation
> everyone actually runs doesn't. We've reproduced the official recipe inside HF and
> verified it end-to-end on the real 149 GB checkpoint on 4× RTX PRO 6000: native
> FP8/FP4 inference works, and our quantized-cache machinery is **bitwise-identical**
> to stock when disabled — so simulation quality results transfer to real storage
> exactly. Measured stakes: the BF16 cache is ~437 MiB/sequence at 65k context
> (~7 GiB at 1M); real packed storage roughly halves it — 50–100+ GiB across a
> long-context serving batch. Next: calibration + quality evaluation on held-out data
> and the final benchmark matrix, running on GCP G4 (same GPUs, we have credits),
> ~1 compute-day remaining. Bonus finds along the way: a PyTorch `ldexp` multi-GPU
> bug (fixed locally, reportable upstream) and a cloud host with silently-corrupting
> PCIe P2P (detection tool now in the repo). Details: `docs/PROJECT_BRIEF.md` and
> `docs/ENGINEERING_REPORT.md` in the project repo.
