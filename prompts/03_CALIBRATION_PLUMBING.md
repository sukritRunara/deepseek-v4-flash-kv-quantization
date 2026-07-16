# Task 03 — Calibration and Precision-Policy Plumbing (DGX plan Phase 4)

## Objective

Build the calibration pipeline: enumerate quantizable targets, collect activation ranges and
quantization error, measure one-group empirical perturbation with the Task-02 teacher-forced
harness, rank targets, and emit a versioned per-group **precision map** that a mapped cache
consumes. A tiny-model smoke calibration must produce a valid map and a held-out evaluation.

Design unification: a single-entry precision map IS the perturbation experiment; the full map
IS the final mixed-precision policy. The map consumer (`MappedQDQCache`) is built once.

## Scope

1. `src/v4_kv_quant/targets.py` — target taxonomy `(layer_idx, layer_type, state, group)`:
   - `window_kv` / `compressed_kv`: contiguous channel groups within the NON-RoPE width
     (production groups of 64; RoPE slice is never a target);
   - `indexer_kv`: ONE state-level target per CSA layer (rotation mixes channels, so
     per-group granularity in the rotated basis has no deployable meaning — the official
     policy is uniform whole-vector FP4);
   - compressor buffers / overlap / gates are NOT targets (stay BF16, injection plan §5).
2. `src/v4_kv_quant/precision_map.py` — versioned `PrecisionMap` (entries: layer/state/
   group channel range/kind/scale-group), JSON round trip, validation against a config
   (state-per-layer-type legality, bounds, non-overlap, full indexer coverage), provenance.
3. `src/v4_kv_quant/mapped_cache.py` — cache layers applying per-group QDQ from a map;
   `indexer_query_policy()` synthesizes the symmetric query-QDQ context.
4. `src/v4_kv_quant/stats.py` — pass-through stats collector cache: per (layer, state)
   amax / mean|x| / RMS, per-group amax, RMS QDQ error for FP8 (nope) and rotated FP4
   (indexer). Must not change any value (bit-exact logits, test-pinned).
5. `src/v4_kv_quant/sensitivity.py` — one-target perturbation runner (metrics vs baseline:
   ΔNLL, KL, max|Δlogit|, top-1; + top-k overlap for indexer targets), full sweep,
   ranking, `build_map_from_sweep(...)` with explicit thresholds/fractions.
6. `tools/run_calibration_smoke.py` — end-to-end smoke: stats pass → sweep on CALIBRATION
   ids → map → HELD-OUT evaluation; saves exact token ids, seeds, thresholds, and all
   results under `results/calibration_smoke/`.
7. `tests/test_calibration.py` — gates below.

## Acceptance gates

1. Empty map ⇒ bit-exact vs stock `DynamicCache` (logits and states).
2. A single-group entry perturbs ONLY its channel slice at its layer/state (other groups,
   RoPE slice, other layers' first-write bitwise clean at layer level).
3. Full-coverage map reproduces the Task-02 policy cache bitwise (same group geometry).
4. Stats collector is bit-exact pass-through; recorded amax matches manual computation.
5. Sweep is deterministic; indexer targets report top-k overlap.
6. Smoke calibration produces a schema-valid map + held-out metrics with provenance
   (seeds, token-id file, thresholds); calibration and held-out ids are disjoint sets.
7. Gradient-weighted ranking: deferred (optional per CLAUDE.md; empirical perturbation is
   the primary truth) — documented, not implemented.

## Constraints

Tiny random model: synthetic deterministic token ids stand in for C4 (the real-corpus
loader is a RunPod-phase port; see REFERENCE_PORT_MAP §"calibration data"). Absolute
sensitivity numbers are machinery validation only. Unpadded inputs; seeds everywhere;
no vendor/reference edits; simulation only — no memory-saving claims.
