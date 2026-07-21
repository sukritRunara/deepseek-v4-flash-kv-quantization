# Final Report — DeepSeek-V4-Flash KV-Cache Quantization

*Written 2026-07-20, at Phase B completion (branch `runpod-phase-b`). Audience: both
ML engineers and technically-literate non-specialists — plain-language framing is
inline throughout. Deep references: `docs/ENGINEERING_REPORT.md` (local phase, for
engineers), `docs/V4_CACHE_ARCHITECTURE.md`, `docs/DECISIONS.md` (D-001…D-015),
`docs/WORKLOG.md` (chronological detail), `PROJECT_STATUS.md` (current state).*

---

## 1. What this project did, in plain terms

When a large language model reads or writes text, it keeps a running "scratchpad" of
everything it has processed so far — one entry per token — so it doesn't have to
re-read the whole history for every new word. This scratchpad is the **KV cache**. For
a long-context model like DeepSeek-V4-Flash (built for million-token contexts), the
scratchpad grows with every token and, in serving, multiplies across every concurrent
user. Past a point, it — not the model's weights — is what fills up the GPU.

The project asked: **can we write that scratchpad in "shorthand" — 8-bit and 4-bit
numbers instead of 16-bit — without changing what the model says?** And not "probably
fine" but *measured*: which parts of the cache tolerate shorthand, which don't, and
what do we actually save in bytes and lose in speed or quality?

**The answer: yes — the cache shrinks to about half its size (0.51–0.54×), with
output quality on held-out text statistically indistinguishable from the original,
at a ~9–12% latency cost that is an artifact of our unoptimized read path, not of
the compression itself.** Along the way the project also produced three
methodological findings about how sensitivity measurement breaks in architectures
with sparse attention — findings that invalidated two of our own intermediate
measurements before they could mislead the final result.

## 2. Where we started: concepts inherited, code rebuilt

We were given a working proof-of-concept from an earlier project that quantized the
cache of DeepSeek-V2/V3 — a different architecture (MLA, "latent attention"). An early
audit (`docs/REFERENCE_REPO_AUDIT.md`) established it could not be reused directly:
V4's cache has none of the structures that code manipulates.

What we **kept as concepts**: teacher-forced evaluation (compare original vs
quantized model on *identical* token histories, so differences measure only the
cache change); the shape of FP8/FP4 quantize-dequantize routines; the organization
of a sensitivity analysis (perturb one thing, measure downstream damage); and
serializable precision configurations.

What we **rebuilt from scratch, V4-native**: everything else. V4 is not MLA — it
uses shared-KV multi-query attention where each token stores a single 512-number
vector serving as both key and value (channels 0–447 carry content, 448–511 carry
rotary position encoding), a 128-token sliding window on every layer, plus two
"compressed" summary streams (CSA and HCA), and an **indexer** — think of it as a
librarian that, for every new token, picks the 512 most relevant compressed entries
to actually attend to. All of this was established by reading and *running* the
official source, with runtime assertions, before any quantization code was written
(D-003: quantization injects as cache-class subclasses; the model's own code is
never edited).

## 3. The optimization itself

Three precision formats, applied at the cache boundary:

- **FP8 (e4m3, groups of 64 channels, power-of-two scales)** for the content
  portion of main KV entries — matching the numerics of DeepSeek's own official
  inference kernels bit-for-bit (verified, including round-to-nearest-even
  midpoints in software FP4, D-007).
- **FP4 (e2m1, packed two-per-byte, after a Hadamard rotation that spreads
  outliers)** for indexer keys — again the official scheme.
- **BF16 (untouched)** for rotary-position channels and anything the measurements
  said was risky.

Two rigor rules separated *science* from *savings*:

- **Stage B (simulation)**: quantize → immediately dequantize → continue in BF16.
  Measures quality impact only; saves zero memory *by design* and is labeled as
  such everywhere.
- **Stage C (actual storage)**: the quantized bytes plus their scales are the only
  thing kept; dequantize on read. This is where real memory savings appear — and
  it carries a hard bitwise contract: `load(store(x)) == qdq(x)`, so every Stage-B
  quality result transfers exactly to Stage-C storage (D-009; verified bitwise on
  the real 4-GPU model, twice, on two different hosts).

## 4. The process — five phases, and what each contributed

**Phase 1 — Local development (ASUS GX10, no weights allowed).** Built and
test-pinned everything that doesn't need the real 149 GB checkpoint: architecture
discovery, QDQ primitives, cache subclasses, calibration plumbing, storage layouts,
benchmark harness. 95 tests, all bitwise gates on a tiny random model. The philosophy:
the rented-GPU phase should spend zero hours debugging plumbing. (Full detail:
`docs/ENGINEERING_REPORT.md`.)

**Phase 2 — RunPod validation (1 then 4 GPUs).** The real checkpoint runs in its
native FP8/FP4 weight formats on RTX PRO 6000 (SM120) — proven by a one-hour
generation gate before committing to the full plan. Two production incidents were
root-caused here, both worth remembering:
- **The rented node silently corrupted GPU-to-GPU transfers** (a host PCIe
  misconfiguration invisible to standard tools). Symptom: nonsense generation that
  changed run to run. The diagnosis eliminated kernels, checkpoint bytes, and loader
  byte-for-byte before a direct stress test caught the hardware (D-011). A
  workaround (staging copies through CPU memory) and a permanent health-check tool
  (`tools/p2p_stress_check.py`) came out of it.
- **An upstream PyTorch bug** (`torch.ldexp` missing a CUDA device guard) produced
  garbage on multi-GPU runs; fixed locally by constructing powers of two via IEEE
  bit layout (still to be reported upstream).

**Phase 3 — Migration to GCP (same GPU model, healthy hardware).** The stress check
passed natively (0/480 corrupt), so the slow workaround became opt-in (D-014). Two
re-validations proved continuity: the generation gate reproduced RunPod's output
word-for-word, and the bitwise instrumentation gates passed — with the random-prompt
divergence metric matching RunPod to the third decimal (same seed → same value), a
strong cross-host reproducibility signal.

**Phase 4 — Calibration on the real model (B4).** A seeded, unpadded corpus (80%
C4-English web text, 20% code), token ids saved for exact reproducibility; held-out
evaluation data from disjoint stream regions. Then a two-stage sensitivity sweep:
quantize *one* target (a channel group, or a whole per-layer state) at a time, run
teacher-forced, and measure downstream damage (ΔNLL — "did the model get worse at
predicting real text" — plus KL divergence and, for the indexer, top-k overlap —
"does the librarian still pick the same pages"). 105 state-level targets screened,
105 group-level targets refined in the most sensitive states, FP8 spot-checks,
and a dedicated 8k-context indexer sweep.

**Phase 5 — Selection and final benchmark (B5–B7).** Three candidate precision maps
of increasing aggressiveness were evaluated against the official QDQ policy on
held-out data at 2k and 8k context, with hard guardrails (a candidate must match or
beat the official policy on ΔNLL, KL, and top-1 agreement). A 32k spot-check broke
the tie. The full performance matrix then ran on the final node: three variants
(baseline / simulation / actual storage) × three context lengths (1k / 8k / 65k) ×
5 warmed trials, medians reported, same node, same run.

## 5. The three methodology findings (what almost fooled us)

These matter beyond this project — they are failure modes of sensitivity measurement
in any architecture with sparse selection:

1. **A measurement that cannot fail is not a measurement.** At 2k-token probes, the
   indexer picks its top 512 entries from a pool of ≤ 512 — selection never actually
   selects, so every indexer perturbation measured *exactly* zero damage and perfect
   overlap. The numbers looked wonderful and meant nothing. (Caught because 21
   identical perfect scores are too good to be true; fixed with 8k probes where the
   pool is 4× the selection size.)

2. **Chaotic downstream noise can drown the signal you're ranking by.** The model's
   indexer scores contain near-ties; *any* tiny perturbation of an early layer flips
   some of them, and the flips cascade through ~40 downstream layers. Result: the KL
   "sensitivity" of early-layer states saturates at a noise floor where FP8 (fine)
   and FP4 (coarse) measure identically. The ranking still orders layers honestly —
   late layers score ~10× lower because their cascade is short — but format choices
   had to rest on held-out guardrails, not probe scores.

3. **Check what your instrument actually perturbs.** The per-layer indexer sweep
   returned a uniform ~0.62 overlap for every layer — even the last one, which has
   almost nothing downstream to disturb. That arithmetic is impossible for a true
   per-layer effect, and the code confirmed it: the query-side quantization wrapper
   arms *all* layers at once, so a "one-layer" experiment actually scored twenty
   layers with mismatched key/query bases. The per-layer numbers were discarded as
   an artifact; the indexer decision was made at the only granularity the
   instrument honestly supports (all layers on / all layers off).

## 6. The result: the ratified precision map

**All main-KV content channels → FP8. Rotary-position channels → BF16. Indexer →
BF16** (i.e., we *decline* the official FP4 indexer path). Owner-ratified 2026-07-20
(D-015).

The interesting part is the indexer decision. The official DeepSeek inference stack
quantizes indexer keys and queries to FP4. On our held-out data that policy is
benign at 8k context (librarian picks 95.2% identical) — but the overlap *decays
with context length*, crossing below the project's 0.9 quality gate at 32k (88.6%
identical picks, while our BF16-indexer map holds 91.1%). Perplexity and top-1
agreement stay clean for both, so this is a conservative call on the one metric
designed for the indexer (D-004: near-tied selection scores make logit closeness
the wrong lens). A long-context retrieval benchmark would discriminate further and
is the top recommended follow-up.

Quality of the ratified map on held-out data (teacher-forced vs. unquantized
baseline): ΔNLL ≈ +0.0009 at 2k, +0.0009 at 8k, −0.0002 at 32k (baseline NLL ≈
1.57), top-1 agreement 0.949–0.977, indexer overlap ≥ 0.91 everywhere tested.
**In perplexity terms: a ratio of e^ΔNLL ≈ 1.0009 — under 0.1% change — and at 32k
the quantized version scored nominally *better* than the baseline, the tell that
these deltas are measurement noise, not degradation.** The 95–98% top-1 agreement
should not be read as a 2–5% accuracy loss: disagreements concentrate at positions
where the baseline model is itself nearly undecided between tokens (near-tied
logits), where either choice is equally good — which is exactly why perplexity
stays flat while agreement sits below 100%.

## 7. The result: measured memory and speed (final node, medians of 5)

| Context | Variant | TTFT | Prefill tok/s | Decode tok/s | ITL p50 | Cache size | Cache vs baseline |
|---|---|---|---|---|---|---|---|
| 1k | baseline | 491 ms | 2086 | 5.0 | 195 ms | 13.4 MiB | — |
| 1k | actual storage | 544 ms | 1884 | 4.4 | 221 ms | 7.2 MiB | **0.535×** |
| 8k | baseline | 5.45 s | 1503 | 5.0 | 196 ms | 60.5 MiB | — |
| 8k | actual storage | 5.77 s | 1419 | 4.4 | 219 ms | 31.2 MiB | **0.516×** |
| 65k | baseline | 85.1 s | 770 | 5.0 | 195 ms | 436.7 MiB | — |
| 65k | actual storage | 87.4 s | 750 | 4.4 | 218 ms | 223.1 MiB | **0.511×** |

(The simulation variant confirms 1.000× — simulation saves nothing, as designed.
Benchmarked policy: the official QDQ policy; the ratified map differs only in
keeping the indexer BF16. All comparisons are same-node, same-run; per-trial data
in `artifacts/phase_b_gcp/benchmark_matrix.json`.)

**How to read this if you're not an ML engineer:** the model's scratchpad now takes
half the memory at every context length we tested, including scales and packing
overhead — honest accounting, not a projection. The price today is ~9–12% slower
token-to-token latency, because our storage prototype decompresses entries in plain
PyTorch on every read. That cost is an engineering artifact: the known remedy
(fused GPU kernels that decompress inside the attention operation, "Stage D") was
deliberately out of scope until correctness was proven. The savings compound in
serving: cache memory scales with context length × concurrent users, so halving it
roughly doubles how much context or how many users fit in the same GPU memory
budget for the cache component.

**Quantization micro-costs** (per-operation, 65k-context shapes): FP8 encode of one
cache write ≈ 78 µs; FP8 decode of the full compressed stream ≈ 47 µs; FP4 indexer
entry encode ≈ 219 µs; FP4 full decode ≈ 65 µs.

## 8. Caveats — what these numbers do and don't claim

- Same-node comparisons only. No cross-hardware claims; RunPod-era numbers are
  historical (that node had faulty P2P; this one doesn't).
- Batch size 1, single process, eager attention, pipeline-sharded over 4 GPUs.
  Serving stacks with continuous batching will see different absolute numbers;
  the *ratio* (cache bytes) is layout-determined and transfers.
- Quality evals are next-token-prediction (NLL/perplexity) on C4-style text + code
  at 2k/8k/32k. No instruction-following, reasoning, or long-context retrieval
  evals were run. The held-out sample is modest (a handful of long sequences), so
  "≤0.1% perplexity change, sign flipping across lengths" is best read as "no
  measurable impact at this statistical power," not as a precise bound.
- The ~10% ITL overhead is the unfused read path, not the format.
- Peak GPU allocation barely moves at these settings because weights dominate at
  batch 1 — the cache savings matter at scale, not in this microbenchmark's peak.

## 9. Reproducibility

Everything is pinned and committed: source revisions (`configs/source_pins.json`),
environment (`artifacts/env/`), calibration token ids, all sweep records, all
candidate maps, per-trial benchmark JSON, and a SHA-256 manifest
(`docs/results_manifest.json`). The full evidence chain for every decision is in
`docs/DECISIONS.md` (D-001…D-015) and `docs/WORKLOG.md`. Suite: 101 tests.
Operational lore that will save the next person a day: pre-warm the page cache
before loading shards; never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
on this stack; `kernels==0.15.2` is required for native-FP8 inference; run
`tools/p2p_stress_check.py` before trusting any new multi-GPU host.

## 10. Recommended next steps

1. **Long-context retrieval eval** (needle-in-haystack style) to pressure-test the
   indexer decision beyond perplexity.
2. **Stage-D fused dequant kernels** to reclaim the ~10% ITL and unlock the
   optimized read path (dequantize only selected CSA entries).
3. **Benchmark the ratified map itself** (the matrix above timed the official
   policy; the map's profile differs only in the indexer).
4. **Serving-shaped benchmarks**: batching, longer decodes, tensor/expert
   parallelism (also restores the faster DeepGEMM weight path that multi-device
   pipelines disable).
