# Task 04 — Stage-C Actual-Storage Prototype (DGX plan Phase 5)

## Objective

Store cache states in genuinely low-precision form — `float8_e4m3fn` codes with explicit
`float8_e8m0fnu`/fp32 scale tensors for main KV (non-RoPE slice), byte-packed e2m1 nibbles
for the indexer (rotated basis) — and dequantize on read in pure PyTorch. Demonstrate real
storage reduction on a tiny test with honest accounting. Correctness prototype only:
expected to be SLOWER than baseline before kernel fusion; no speed claims on GX10.

## Scope

1. `src/v4_kv_quant/storage.py` — storage primitives with a hard numerical contract:
   `load(store(x)) == qdq(x)` **bitwise** for every kind (FP8 g64 ue8m0, FP4 g32,
   rotated FP4). Scales stored as `float8_e8m0fnu` when power-of-2 (1 byte), fp32 otherwise.
   FP4 codes: sign bit + 3-bit magnitude index, two per uint8.
2. `src/v4_kv_quant/storage_cache.py` — `QuantizedStorageCache(config, policy)` reusing the
   Task-02 `KVQuantPolicy` (same kinds now mean actual storage): window store (append +
   window-trim in lockstep across codes/scales/rope, contiguous after trim so no hidden
   BF16 history is retained), append-only compressed store, rotated packed indexer store.
   `keys`/`values` stay empty placeholders — quantized tensors are the only storage.
   `entry_count` bookkeeping preserved (compressor RoPE positions depend on it).
   Buffers/overlap stay BF16 stock. Sliding-only layers stop duplicating V (K=V return).
3. `src/v4_kv_quant/memory.py` — per-state logical and storage bytes
   (`numel*element_size` and `untyped_storage().nbytes()`), K=V alias counted once,
   stock sliding-layer V duplication flagged, scales/rope/buffers/overlap all itemized.
4. `tools/measure_cache_memory.py` — same token stream through stock BF16 cache, Stage-B
   QDQ cache, and Stage-C storage cache; per-layer byte comparison + JSON. Must show the
   QDQ cache saves NOTHING and the storage cache shows real reduction.
5. `tests/test_actual_storage.py` — gates below.

## Acceptance gates

1. Primitive round trips: `load(store(x)) == qdq(x)` bitwise (fp32 and bf16 pipelines,
   FP8 and FP4, rotated FP4; e8m0 scale storage exact for power-of-2 scales).
2. **Model-level: `QuantizedStorageCache(policy)` logits bitwise-identical to
   `QDQCache(policy)`** for `main_fp8_nonrope_rope_bf16` and `reference_official_qdq`.
3. All-BF16 policy through the storage cache: logits bitwise-identical to stock cache.
4. Codes written once (bitwise-stable across appends/decode); window trim keeps exactly
   the last `window-1` rows and drops old storage (no silent history retention).
5. Task-01 count invariants hold (`cumulative_length`, `entry_count`, buffer lengths).
6. Memory report shows quantized total bytes strictly below baseline on the same tokens,
   with scales and rope slices included in the quantized total.
7. No vendor/reference edits; `_layer_type = None` (registry untouched).

## Constraints

Tiny model, deterministic seeds, unpadded inputs. Dequantize-on-read is per-forward and
never persisted. Report bytes for the actual run dtype (fp32 tiny model) and label that
real-model ratios differ (BF16 baseline). GX10 timings are meaningless — do not record them.
