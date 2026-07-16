# Task 02 — Official-Policy QDQ Simulation (Stage B)

## Objective

Reproduce DeepSeek-V4-Flash's official QAT-aligned quantize-dequantize numerics at the
Transformers cache write boundary, selected entirely through a serializable precision policy.
Simulation only: values remain BF16/FP32 storage — **no memory-saving claims**. This builds and
validates the machinery (QDQ primitives, policy object, cache subclasses, metrics harness)
that calibration (Phase 4) and actual-storage (Stage C) will reuse.

## Scope

1. `src/v4_kv_quant/qdq.py` — pure-PyTorch primitives matching `inference/kernel.py` semantics:
   - FP8 e4m3 QDQ: contiguous groups along the last dim, absmax scale with 1e-4 floor,
     optional ue8m0 power-of-2 round-up scale (exact via frexp), clamp ±448, native
     `float8_e4m3fn` cast round trip;
   - FP4 e2m1 QDQ: groups of 32, power-of-2 round-up scale (`amax/6`, floor `6·2⁻¹²⁶`),
     software e2m1 grid with round-to-nearest-even ties (no native cast in torch 2.13);
   - exact Walsh–Hadamard transform (orthonormal FWHT, power-of-2 dims).
2. `src/v4_kv_quant/policy.py` — versioned `KVQuantPolicy` (window_kv / compressed_kv /
   indexer_kv / indexer_q state policies), JSON round trip, named presets:
   `baseline_bf16`, `reference_official_qdq`, `main_fp8_nonrope_rope_bf16`,
   `main_fp4_nonrope_rope_bf16`, `indexer_reference_qdq`.
3. `src/v4_kv_quant/qdq_cache.py` — cache-layer subclasses (window + compressed + indexer
   write paths), `_layer_type = None` so the global registry is untouched; indexer-query QDQ
   as a scorer-wrapper context manager; `build_qdq_cache(config, policy)`.
4. `src/v4_kv_quant/metrics.py` — max/mean/RMS logit error, KL, NLL delta, top-1 agreement,
   NaN/Inf counts, indexer top-k overlap.
5. `tools/run_qdq_simulation.py` — teacher-forced baseline-vs-policy comparison on the tiny
   model; readable table + JSON to `results/`, explicitly labeled simulation.
6. `tests/test_qdq_simulation.py` — gates below.

## Acceptance gates

1. Policy disabled ⇒ **bit-identical** logits and cache states vs stock `DynamicCache`.
2. RoPE slice (trailing `qk_rope_head_dim`) bit-untouched under every policy.
3. QDQ applied exactly once per value (window write / entry emission); earlier entries
   bitwise-stable across later appends and decode steps.
4. Chunked prefill / token-by-token decode match one-shot under QDQ (dense-indexer config,
   Task-01 tolerances).
5. Entry counts and shapes preserve all Task-01 invariants.
6. Metrics harness runs end to end; results labeled "simulation — no memory savings".
7. No edits to `vendor/` or `reference/`; production cache classes untouched.

## Constraints

Tiny model only; unpadded inputs; deterministic seeds; group sizes read from policy with the
documented tiny-model fallback (group = state width when width < group and divisibility
holds); production group sizes (64/32) exercised in unit tests at real widths (448/128).
