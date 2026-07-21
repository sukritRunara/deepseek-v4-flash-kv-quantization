# Future Work — FP8/FP4 Mixture Across the KV Cache

*Planning record, 2026-07-20, agreed with owner in-session (post-Phase-B). Nothing
here has been executed. Purpose: survive context compaction so the discussion can
resume exactly here. Background: `docs/FINAL_REPORT.md`, D-015, WORKLOG 2026-07-20.*

## Goal

Extend the ratified all-FP8 map toward FP4: measure the full tradeoff curve of
**cache bytes vs quality** from all-FP8 (measured: 0.51× baseline bytes) toward
all-FP4 main-KV (projected ~0.35×: nope channels 0.25× + RoPE BF16 + scales), and
pick a mixture on that curve. Same Stage-B→Stage-C discipline as Phase B (bitwise
storage contract means quality results transfer to real storage).

## Step 0 — cheap first datapoint (OWNER-APPROVED; run this, then HARD STOP)

Evaluate **all-FP4 main-KV** (`main_fp4_nonrope_rope_bf16` named policy — already
exists; indexer stays BF16 per the D-015 finding that the indexer choice is binary
and FP4 indexer failed the 32k overlap gate) on the existing held-out suite
(2k/8k + 32k spot). One model load, ~30 min. This tells us whether the far end of
the curve is even alive before investing in the ladder.
**Stop point: owner discussion of the result before any mixture work.**

### Step 0 RESULT (run 2026-07-20, `--heldout-policies main_fp4_nonrope_rope_bf16`;
same-run comparators; full rows in `heldout_eval.json` / `heldout32k_eval.json`)

| variant | 2k dNLL / top1 | 8k dNLL / top1 | 32k dNLL / top1 | 32k idx-overlap |
|---|---|---|---|---|
| ratified FP8 map | +0.0009 / 0.9591 | +0.0009 / 0.9491 | −0.0002 / 0.9771 | 0.911 |
| official | +0.0010 / 0.9558 | +0.0008 / 0.9485 | −0.0003 / 0.9767 | 0.886 |
| **all-FP4 main-KV** | **+0.0096 / 0.9452** | **+0.0080 / 0.9390** | **+0.0086 / 0.9703** | **0.890** |

**Reading:** the far end of the curve is ALIVE but measurably degraded — ΔNLL
≈ +0.008–0.010 (≈ **+0.9–1.0% perplexity**), consistent in sign across all three
lengths (a real signal, ~10× the FP8 map's noise-level deltas, so current
held-out power resolves it fine at this magnitude). Top-1 drops ~1–1.5 points.
Notably, at 32k the indexer overlap falls to 0.890 — below the 0.9 gate — even
though the indexer itself is BF16: FP4 damage to the compressed cache content the
indexer scores over drags selection with it. Implication for the ladder: the
interesting region is the middle — how much FP4 (ordered least-sensitive-first
per route 1) fits before ΔNLL exceeds noise and/or 32k overlap crosses 0.9.
Endpoints now anchored: 0.51× bytes at ~0% quality cost (FP8) vs ~0.35× bytes at
~1% perplexity + gate failure (all-FP4).
**Status: HARD STOP honored — awaiting owner discussion before ladder work.**

## Agreed approach for the mixture search (owner decision, 2026-07-20)

**Route 3 (brute-force candidate ladder) with Route 1 as the ordering heuristic;
Route 2 only if the ladder shows the ordering is bad.**

- **Route 3 — candidate ladder:** build maps at fp4_fraction ≈ 0, 0.2, 0.4, 0.6,
  0.8, 1.0 over the ranked main-KV pool (FP8 for the rest), evaluate ALL of them
  on held-out end-metrics in one or two model loads (the multi-map heldout stage
  already supports this via `precision_map*.json` glob). Deliverable: Pareto curve
  of cache-bytes (computable per map via `v4_kv_quant.memory`) vs ΔNLL/overlap
  (+ retrieval if added). Rationale: end-metrics average out indexer-flip noise;
  this is the honest measurement even though it costs GPU time.
- **Route 1 — ordering heuristic (zero GPU):** rank groups for "FP4 first" by the
  per-group QDQ RMS error / amax already collected in
  `results/calibration_full/activation_stats.json` (stats pass covered the FULL
  calibration set). Validity unproven — the ladder itself tests it.
- **Route 2 — gradient-weighted score (in reserve):** the reference PoC's concept
  (`reference/v2_mla_poc/ops/sensitivity.py`), deferred by D-004: quantization
  error × gradient magnitude per group; needs backward passes; must be validated
  against empirical perturbation before trusting. Port ONLY if the route-1
  ordering produces a visibly bad ladder (e.g., non-monotone quality vs fraction).

## The measurement crux (why per-target KL cannot pick FP4 vs FP8)

Established 2026-07-20 (WORKLOG "two methodology findings"): V4's indexer selects
top-512 entries via scores containing many near-ties. ANY perturbation of an early
layer — FP8-fine or FP4-coarse — flips some near-tie selections in the ~40
downstream indexer layers, and those flips alone produce KL ≈ 1.5e-2. The
instrument is pinned at that floor: FP8 and FP4 perturbations of early-layer
states read identically. Late layers (short cascade) read ~10× lower and remain
discriminating. Consequences: (a) per-target KL orders layers but cannot choose
formats in early layers; (b) mixture decisions must rest on end-to-end held-out
metrics (NLL/perplexity, retrieval), where unbiased flip noise averages out;
(c) ΔNLL per target is noise-level at current probe sizes — statistical power is
the binding constraint (next section).

## Open question #3 — held-out statistical power (+ retrieval eval)

NOT yet decided with owner. The issue: current held-out set is small (evaluated:
4×2k, 1×8k, 1×32k). Ladder points will differ by ~1e-3 NLL or less; at current
sample sizes the error bars swamp that. Proposal: grow held-out to ~32×2k + 8×8k +
4×32k (≈8× tokens → ~2.8× tighter error bars; report per-sequence variance so
every Pareto point carries an error bar). Additionally: a long-context retrieval
eval (needle-in-haystack style) — C4 perplexity mostly tests short-range
prediction; cache damage plausibly shows first as degraded long-range recall,
which is the KV cache's actual job. Retrieval would also pressure-test the D-015
indexer decision. Cost: corpus regen is cheap; eval adds model-load time per
ladder point; retrieval harness is NEW code (does not exist yet).

## Discussion outcomes (owner + agent, 2026-07-20, post step 0)

1. **Verdict on the all-FP4 delta:** ~1% PPL is within community norms for 4-bit KV
   caches and not alarming by itself; the real concern is the LENGTH TREND of
   selection overlap (0.961 @ 8k → 0.890 @ 32k, gate-failing), with behavior
   beyond 32k unknown on a 1M-context model. PPL delta is roughly flat with
   length; the failure mode that grows is selection drift — which perplexity
   barely sees.
2. **Coupling finding:** FP4 main-KV damage degrades indexer selection EVEN WITH a
   BF16 indexer — queries/keys are computed from hidden states shaped by attention
   over the FP4-stored cache; near-tied scores then flip (more candidates at long
   context → more flips). Consequence: every FP4-containing map must be gated at
   long context on overlap, not just on perplexity.
3. **Characterization:** all-FP4 is mildly-but-really degraded (no degeneration,
   94% token agreement, +1% PPL) — neither "serious" nor "free." Unknowns: task-
   level impact, retrieval impact, behavior past 32k.
4. **Held-out growth scope confirmed:** needed for the ladder's middle rungs
   (deltas ~1e-3 ≲ current error bars), NOT needed to interpret step 0 (effect was
   ~10× noise). Retrieval eval is NOT mixture-specific — it is the correct
   instrument for the selection-drift failure mode and also pressure-tests the
   ratified map.
5. **Agreed sequencing:** retrieval harness (new code) → held-out growth (cheap)
   → ONE ladder pass, gated per rung on (ΔNLL within noise) AND (32k overlap
   ≥ 0.9) AND (retrieval parity); ratify a mixture only if a plateau appears.
   Ladder is self-terminating (stop at the first sloping rung). Prize is bounded:
   the whole mixture space is ~16 points of baseline bytes (0.51× → 0.35×), the
   2× win being already banked — hence modest expectations.
6. **Portfolio note:** Stage-D fused kernels (recover ~10% ITL on the ratified
   map, benefits every deployment) likely beat the ladder on value per
   engineering-hour; they are a parallel track (engineering-heavy) while the
   ladder is GPU-cheap.

## Long-context testing (owner question, 2026-07-20 — agreed: wise, do it)

- Motivation: the failing trend is in the length dimension; 32k is ~3% of the
  model's advertised context. Decay curve shape (saturating vs sliding) decides
  FP4 viability — and matters for the RATIFIED map too (0.969 @ 8k → 0.911 @ 32k
  is not comfortably above the 0.9 gate; a 65k measurement is wanted regardless
  of the ladder).
- Feasibility: benchmark already runs 65k on this node, but the teacher-forced
  harness keeps all logits on-GPU (~17 GB at 65k next to 34 GiB weights on GPU 0
  → OOM). Small required change: stream logits (or per-chunk metrics) to CPU —
  metrics are already position-chunked, so this is cheap. NOT yet implemented.
- Proposed rungs: add 65k (and if the harness change proves comfortable, ~128k)
  to the held-out suite for baseline, ratified map, official, and any ladder
  candidates.
- Corpus caveat: held-out windows are packed short documents — long sequences,
  not long-range dependencies. 65k perplexity/overlap is still informative (the
  overlap metric does not need natural long-range structure), but the retrieval
  eval remains the sharper long-context instrument.

## Standing constraints

- Indexer precision remains a BINARY (all-layers) choice — per-layer indexer
  measurement is impossible without new per-layer scorer wrappers (finding #3,
  WORKLOG 2026-07-20); FP4 indexer already fails the 0.9 overlap gate at 32k.
- Guardrail discipline as in D-015: candidates compared against both the baseline
  and the ratified FP8 map; memory claims include scales; same-node comparisons.
- Stop points: after step 0 (owner discussion); after the ladder, before any
  ratification of a new map.

## Kimi context (owner note)

Owner knows of equivalent FP8/FP4-mixture work on a Kimi model. Kimi is
MLA-family — architecturally closer to `reference/v2_mla_poc` than to V4. V4
differences that change the problem: shared-KV MQA (one 512-vec is both K and V),
per-layer window/compressed/indexer streams (mixture is over heterogeneous
states, not one latent), and the selective indexer (source of the measurement
noise floor above, which MLA-based studies would not have faced).
