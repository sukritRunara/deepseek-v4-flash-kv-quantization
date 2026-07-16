# Quantization Injection Plan (DeepSeek-V4, Transformers implementation)

Derived from `docs/V4_CACHE_ARCHITECTURE.md` (V4ARCH). All line references:
`vendor/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py` (`modular:`)
and `vendor/DeepSeek-V4-Flash/inference/model.py` (`official:`).

**Strategy in one paragraph.** Every cache state is written through exactly one of three
cache-layer methods — `update()` (window KV), `update_compressor_states()` (compressed
entries, main + indexer), `store_compression_weights()` (buffers) — all owned by the
per-layer cache classes. We therefore implement quantization entirely as **cache-layer
subclasses** (`DeepseekV4CSACache` / `DeepseekV4HCACache` / `DynamicSlidingWindowLayer`
descendants) constructed from a serializable precision policy, and never touch
`DeepseekV4Attention.forward` or the generated modeling file. `DynamicCache(config=…)`
dispatch is replaced by explicitly building the layer list (same registry pattern,
`cache_utils.py:1709`). Indexer *query* QDQ (needed for exact official parity) is the one
exception that cannot live in a cache layer; it wraps `DeepseekV4Indexer.forward` externally
(hook or subclass) and is deferred to the end of Task 02.

Stage labels: **B** = QDQ simulation (BF16 storage, numerics only), **C** = actual
low-precision storage, **D** = fused/optimized read path (deferred; not on GX10).

## Per-state insertion points

### 1. Sliding-window main KV (all 43 layers)

State: `layer.keys` (= `values`), `[B, 1, ≤127, 512]`, written post-`kv_norm`+RoPE by
`update()` (`modular:167` on CSA/HCA; `cache_utils.py:208` on sliding-only layers).

| Stage | Insertion | Detail |
|---|---|---|
| B (QDQ) | subclass `update()`: QDQ `key_states[..., :448]` *before* concat/window-trim, return values untouched otherwise | mirrors `official:506` (`act_quant(kv[..., :-64], 64, ue8m0)`); the trailing 64 RoPE dims are the **precise slice** — bit-untouched (test-enforced) |
| C (storage) | same subclass: store `keys_nope_fp8 [B,1,W,448] float8_e4m3fn` + `scales [B,1,W,7] e8m0/fp32` + `keys_rope_bf16 [B,1,W,64]` | replaces the single BF16 tensor; window-trim logic operates on all three tensors in lockstep |
| C read | in `update()` return path: dequantize `full` before handing to attention | pure PyTorch `q.to(bf16) * scale.repeat_interleave(64)`; slower than baseline — expected pre-fusion |
| D (fused) | dequant fused into attention K/V load; window ring-buffer layout like `official:473` | RunPod/SM120 work; per `official:527` comment the GEMM could consume FP8 directly |

Bonus on sliding-only layers (stage C): stop duplicating V (stock `DynamicSlidingWindowLayer`
stores separate `values`; V4ARCH §2 note) — alias K=V like the CSA/HCA layers do; free 2×
saving on 3 layers before any quantization.

### 2. CSA main compressed KV (20 layers)

State: `compressed_kv["compressor"]`, `[B, T, 512]`, append-only; written by
`update_compressor_states("compressor", …)` (`modular:204`) after kv_norm + compress-RoPE
(`modular:609–616`).

| Stage | Insertion | Detail |
|---|---|---|
| B | subclass `update_compressor_states()`: if `name == "compressor"`, QDQ `compressed[..., :448]` before append | mirrors `official:372`; entries are QDQ'd **once at emission**, never re-quantized (append-only makes this natural — test: entry bytes stable across later steps) |
| C | store per-entry `nope_fp8 [B,T,448] + scales [B,T,7] + rope_bf16 [B,T,64]`; append to all three | read = dequant + concat onto window KV before attention (`modular:725` consumes the return value) |
| C read | in `update_compressor_states()` return (it returns the running tensor) | keep a BF16 "materialized view" only per forward call, never persisted |
| D | dequantize **only indexer-selected entries** (top-k=512 of T) before attention | this is the big long-context win: decode reads ≤512 compressed entries regardless of T; requires gather-then-dequant fused kernel |

### 3. HCA main compressed KV (20 layers)

Same write/read sites as CSA (class inherits; `name == "compressor"` only). Differences that
matter for policy, not plumbing: entries pool 128 tokens each (highest information density per
entry → likely most sensitive; calibrate separately), and HCA attends **densely** to all T
entries (no top-k), so stage D selective dequant does not apply — fused dequant-attention does.

### 4. CSA indexer KV / state (20 layers)

State: `compressed_kv["indexer"]`, `[B, T, 128]`, written by
`Indexer.forward → update_compressor_states("indexer", …)` (`modular:496`).

| Stage | Insertion | Detail |
|---|---|---|
| B | same subclassed `update_compressor_states()`, branch `name == "indexer"`: Hadamard-rotate then FP4 QDQ groups of 32 over the **full 128 dims** (incl. rope slice) | mirrors `official:368–370`; reproduce official QAT-aligned path *first* (CLAUDE.md hypothesis); pure-PyTorch Hadamard (128 = 2⁷, power of two — exact H₁₂₈ exists) |
| B (queries) | wrap `DeepseekV4Indexer.forward` (external subclass swap or module hook): QDQ `q` after RoPE (`modular:501`) with the same Hadamard+FP4 | `official:414–416` quantizes q and kv symmetrically — keys-only QDQ would skew `q·k`; validate necessity by measuring both |
| C | packed FP4: `[B, T, 64] uint8` (2 nibbles/byte) + `[B, T, 4] e8m0` scales | native `float4_e2m1fn_x2` viewing verified available (hardware_smoke) |
| C read | dequant inside `IndexerScorer` input (`modular:391`) | scores are fp32 there already |
| D | keep indexer keys permanently packed; fused ReLU(q·k) scorer | candidate for biggest *bandwidth* win at 1M context (T = 262k entries at m=4) |

Metric gate for anything touching the indexer: **top-k overlap/recall vs BF16 baseline**, not
logit closeness (V4ARCH §6.8 — selection is tie-unstable; logit deltas explode when the set
flips, overlap tells you *how often* it flips).

### 5. Incomplete compressor buffers, overlap state, gates

States: `buffer_kv/buffer_gate[name]` (< rate tokens), `overlap_kv/overlap_gate[name]`
(one window's Ca slice, CSA only), written by `store_compression_weights` (`modular:181`) and
`update_overlap_state` (`modular:249`).

**Stay BF16/full precision at every stage of this project.** Rationale: (a) bounded size —
per layer ≤ `(rate−1)·2d + m·d` elements ≈ **3.5 KB/seq** (CSA) vs a compressed stream that
grows ~1 entry/4 tokens: quantizing them saves nothing material; (b) they feed the softmax-gated
pooling — errors here multiply into *every* future compressed entry (official keeps analogous
`kv_state`/`score_state` in **fp32**, `official:303–304` — strictly higher precision than HF's
BF16, another reason not to go lower); (c) gate values pass through `softmax(dim=…)` where
relative errors distort mixing weights nonlinearly.

### 6. Scales and metadata

- Scale representation (stage B+C): follow official — **power-of-2 scales, round up**
  (`fast_round_scale`, `official kernel.py:36`), stored as `float8_e8m0fnu` (1 byte) with an
  fp32 fallback config flag; group sizes: 64 (FP8 main), 32 (FP4 indexer). Both dtypes verified
  on this machine.
- Scale storage layout (C): sibling tensor per quantized state, same leading dims
  (`[B, W, 7]` window / `[B, T, 7]` compressed / `[B, T, 4]` indexer), appended in lockstep —
  keeps append semantics trivial and makes memory accounting explicit.
- `entry_count`, `cumulative_length`: ints, untouched.
- Memory accounting (all stages): count values + scales + padding + buffers + overlap +
  the K=V alias (once) via real tensor storage bytes; report allocator peaks separately.

## What stays BF16 initially (Task 02 baseline policy), and why

| Component | Why BF16 initially |
|---|---|
| RoPE slice of every main KV state (window + compressed), 64/512 dims | official never quantizes it ("positional precision", `official:505`); it is the *precise slice* our tests pin |
| Compressor buffers / overlap / gates | §5 above — negligible size, error-amplifying position |
| Indexer *scores* path (weights_proj output, softmax scale) | fp32 in both implementations; not cache state |
| Attention compute (post-dequant GEMMs) | QAT was done for cache values, not for accumulation; official computes BF16 (`official:527` comment) |
| Sliding-only layers in the *first* QDQ experiment | isolate one variable: start with CSA/HCA main KV (the dominant state), enable window-KV QDQ as the second config |

## Policy configuration surface (Task 02 deliverable)

```
policy:
  version: 1
  default: bf16
  window_kv:     {nope: fp8_e4m3_g64_ue8m0 | bf16, rope: bf16}
  compressed_kv: {nope: fp8_e4m3_g64_ue8m0 | bf16, rope: bf16}
  indexer_kv:    {full: fp4_e2m1_g32_e8m0_hadamard | bf16}
  indexer_q:     {full: same-as-indexer_kv}          # symmetric QAT pairing
  per_layer_overrides: {layer_idx | layer_type: ...} # contiguous groups only
```

Candidate named policies (DGX_PHASE_PLAN Phase 3): `baseline_bf16`,
`reference_official_qdq` (FP8 main nope + FP4 indexer), `main_fp8_nonrope_rope_bf16`,
`main_fp4_nonrope_rope_bf16` (experimental), `mixed_group_policy`, `indexer_reference_qdq`
(indexer only).

## Test obligations attached to each injection (from CLAUDE.md / DGX plan)

1. Policy disabled ⇒ **bit-identical** to baseline (identity test).
2. RoPE slice bit-untouched under every policy.
3. QDQ applied exactly once per value (window write / entry emission), incl. across chunked
   prefill and decode boundaries.
4. Entry counts/shapes unchanged vs `tests/test_v4_cache_semantics.py` invariants.
5. Stage B explicitly reports "BF16 storage — no memory saving" (CLAUDE.md hard constraint 7).
6. Stage C memory accounting includes scales/padding/buffers before any savings claim.
