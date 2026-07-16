# Reference Repository Port Map (V2/V3 MLA PoC → DeepSeek-V4)

Source: `reference/v2_mla_poc/` (read-only; from `kv-cache-quantization-kimi-k27-main.zip`).
Classifications: **reuse unchanged** / **reuse with modification** / **concept only** /
**not applicable to V4**. Anchors are `file:line` in the reference repo. V4 facts cite
`docs/V4_CACHE_ARCHITECTURE.md` (V4ARCH).

## Summary table

| Component | Anchor | Classification | One-line reason |
|---|---|---|---|
| `KVLatentCache` (old cache inheritance) | `ops/kv_latent_cache.py:56` | **not applicable to V4** | V4 caches are per-layer `CacheLayerMixin` subclasses with compressor state; no `key_cache`/`value_cache` list model |
| `kv_patch.py` attention monkey-patch | `kv_patch.py:140,285` | **not applicable to V4** | keys off `kv_a_proj_with_mqa`/`kv_b_proj` attributes V4 doesn't have; V4 needs cache-layer subclassing, not forward replacement |
| FP8 QDQ function | `ops/kv_latent_cache_quantized.py:86` | **reuse with modification** | correct e4m3 QDQ skeleton, but scales are per-channel absmax; official V4 policy is groups of 64 along features with power-of-2 (ue8m0) round-up scales |
| Software FP4 (e2m1) QDQ function | `ops/kv_latent_cache_quantized.py:114` | **reuse with modification** | correct e2m1 nearest-value grid; needs group-32 power-of-2 scales + Hadamard rotation to match the official indexer path |
| FP3/INT2 grids | `ops/kv_latent_cache_quantized.py:200,205` | **concept only** | not in any official V4 policy; keep as optional extreme-compression experiments |
| Precision config (per-channel uint8 map) | `ops/kv_latent_cache_quantized.py:294–309` | **concept only** | keying must become (layer, state, contiguous group) — V4 has 4+ distinct cache states per layer, not one latent |
| Running-max scale tracking | `ops/kv_latent_cache_quantized.py:329–337` | **concept only** | append-friendly scale idea is right; V4 needs per-group scales stored alongside entries, not a running per-channel max |
| Sensitivity score `mean|c_kv|·‖kv_b_proj col‖` | `ops/sensitivity.py:186` | **not applicable to V4** | depends on `kv_b_proj` reconstruction which V4 does not have; V4 primary truth = empirical downstream perturbation (NLL/KL deltas) |
| Sensitivity analyzer *structure* (hooks, save/load, ranking→config) | `ops/sensitivity.py:64,328` | **reuse with modification** | hook-based activation collection and score serialization carry over; layer discovery must use `layer_idx` + layer_type, not module-path string parsing |
| Calibration data loader (C4 streaming, fixed windows) | `src/calibration_data.py:33` | **reuse with modification** | sound structure (seeded, non-overlapping windows, saved token ids); must add length-bucketing/no-left-pad guarantees for compressor boundaries (V4ARCH §6.2) |
| Teacher-forced logit harness | `tests/verify_quantized_kv.py:83` | **reuse with modification** | the fixed-`reference_ids` teacher-forcing idea is exactly right; port to V4 cache API and add per-position metrics |
| Logit comparison metrics | `tests/verify_quantized_kv.py:192` | **reuse with modification** | max/mean |Δlogit| + top-1 agreement carry over; add KL, RMS, NaN/Inf counts, and indexer top-k overlap (V4ARCH §6.8) |
| Perplexity evaluation | `tests/verify_quantized_kv.py:160` | **reuse with modification** | shifted-CE structure fine; must run through V4 cache path and respect unpadded inputs |
| Memory reporting | `ops/kv_latent_cache.py:146` / `…_quantized.py:363` | **concept only** | reports *theoretical* bytes-per-element of BF16-stored QDQ values; V4 accounting must count actual tensors incl. scales, buffers, overlap, and both K=V aliasing and the sliding-layer double-copy |
| `requirements.txt` | `requirements.txt` | **concept only** | version floor hints only; known malformed line (`nvidia-ml-py>=12.560.30tiktoken`); our env is pinned separately (REPRODUCIBILITY.md) |
| `scripts/setup_runtime.sh`, `scripts/download_model.sh` | `scripts/` | **not applicable to V4 (never run)** | target x86 CUDA 12.8 / torch 2.7.0 and download 160 GB weights; both violate GX10 constraints; also reference a missing `tests/verify_kv_relation.py` |

## Detailed rationale (required items)

### 1. Old cache inheritance — not applicable

`KVLatentCache(DynamicCache)` stores `(kv_a_norm, k_pe)` in `key_cache[i]`/`value_cache[i]`
lists and relies on `kv_b_proj` reconstruction at attention time (`ops/kv_latent_cache.py:56–128`).
V4 has no latent+reconstruction path at all: the cache holds the *final* K=V vector (window +
compressed entries) in per-layer `CacheLayerMixin` objects with dict-keyed compressor state
(V4ARCH §2). A V4 quantized cache must subclass `DeepseekV4CSACache`/`DeepseekV4HCACache`/
`DynamicSlidingWindowLayer` (transformers ≥5 API), not `DynamicCache`. Upstream confirms the
generic `QuantizedCache` is incompatible (V4ARCH §6.3).

### 2. Old attention patch — not applicable

`_patch_attention_forward` (`kv_patch.py:140`) rebuilds MLA attention manually: old-style
`rotary_emb(seq_len=...)` contract, `(attn_output, None, cache)` 3-tuple return, manual softmax.
V4's attention (a) uses the new position-embeddings-dict interface, (b) returns 2-tuples,
(c) applies inverse RoPE to outputs, (d) concatenates compressed KV mid-block. None of the
patch survives. The *replacement strategy itself* is also wrong for V4: QDQ belongs at the
cache write boundary (cache-layer subclass), leaving `DeepseekV4Attention.forward` untouched.

### 3. FP8 function — reuse with modification

`_quantize_fp8` (`ops/kv_latent_cache_quantized.py:86`): per-channel absmax → scale =
`ch_max/448` → cast `float8_e4m3fn` → back to BF16. Correct QDQ skeleton and correct use of
native e4m3. Required modifications to match the official V4 policy (V4ARCH §3):
group size **64 along the feature dim** (not per-channel-over-batch), scale =
`2^ceil(log2(amax/448))` (**round-up power of 2**, `ue8m0`), amax floor `1e-4`, and
application **only to the non-RoPE 448 dims** of main KV states.

### 4. Software FP4 function — reuse with modification

`_quantize_fp4` (`…:114`) rounds to the e2m1 value grid `{0,.5,1,1.5,2,3,4,6}` — a valid
software emulation and a useful CPU reference. Official V4 indexer policy differs: groups of
**32**, power-of-2 scale (`amax/6`, round up, floor `6·2⁻¹²⁶`), **Hadamard rotation before
quantization**, applied to the whole 128-dim indexer vector *and* to indexer queries
symmetrically. Native `torch.float4_e2m1fn_x2` + `float8_e8m0fnu` exist in our torch build
(hardware_smoke PASS) for the actual-storage stage.

### 5. Precision config — concept only

Per-channel `uint8` masks keyed by layer (`…:294–309`) don't map onto V4's state space.
The V4 taxonomy (DGX_PHASE_PLAN Phase 4) is
`(layer_index, layer_type, state ∈ {window_kv, compressed_kv, indexer_kv, buffers}, group_index)`
with contiguous groups of 32/64. Keep: the idea of a serializable, versioned policy object
that the cache consumes at construction.

### 6. Sensitivity score — not applicable (formula) / reuse with modification (harness)

The score `mean|c_kv[:,:,i]| × ‖kv_b_proj.weight[:,i]‖₂` (`ops/sensitivity.py:186`) measures
how latent channel error amplifies through `kv_b_proj` — a projection that does not exist in
V4. V4 primary sensitivity truth = empirical perturbation (quantize one group/state, measure
ΔNLL / logit KL); indexer states additionally need top-k overlap/recall (V4ARCH §6.8 shows
selection flips are the actual failure mode). The *harness* (forward hooks, activation stats,
`save_scores`/`load_scores`, ranked→policy) ports with the layer-discovery fix
(use `module.layer_idx`, not `"layers.<i>"` string parsing — the two disagree on wrapped models).

### 7. Teacher-forced logit harness — reuse with modification

`collect_logits` (`tests/verify_quantized_kv.py:83`) prefills the prompt then feeds a fixed
`reference_ids` sequence so baseline and quantized runs consume identical token histories —
exactly the CLAUDE.md-mandated methodology. Port: V4 cache construction (`DynamicCache(config=…)`
with policy-carrying layer subclasses), unpadded prompts only, and record per-position metrics
(the V4 indexer makes divergence position-dependent).

### 8. Perplexity evaluation — reuse with modification

`compute_perplexity` (`…:160`): shifted cross-entropy over held-out samples, `exp(mean NLL)` —
structure carries over unchanged; modifications are only plumbing (V4 model/cache API, length
bucketing, calibration/held-out separation already present in spirit).

### 9. Memory reporting — concept only

`cache_size_bytes` (`ops/kv_latent_cache.py:146`, quantized variant `…:363`) multiplies element
counts by theoretical bytes/element (FP8=1.0, FP4=0.5) while the storage is actually BF16 —
useful only as a *projection*. V4 accounting must (per CLAUDE.md): measure actual
`tensor.untyped_storage().nbytes()` per state, include scale tensors, padding, compressor
buffers/overlap state, count K=V aliasing once, count the stock sliding layer's V duplication,
and report `torch.cuda` allocator peaks separately. Simulation (Stage B) must always be labeled
"no memory saving".

## Repo defects to avoid importing

- missing `tests/verify_kv_relation.py` and `ops/kv_relation_module.py` (referenced by scripts
  and comments; absent from the archive);
- malformed `requirements.txt` line 37 (`nvidia-ml-py>=12.560.30tiktoken`) + duplicated `tiktoken`;
- V2/V3 code labeled "DeepSeek-V4" in scripts/docstrings (naming confusion);
- `--model-path` help text still says DeepSeek-V2-Lite.
