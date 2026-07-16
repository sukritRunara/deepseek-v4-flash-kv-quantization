# Audit of the Supplied Reference Repository

Repository source: `kv-cache-quantization-kimi-k27-main.zip`

## What it actually implements

The code implements a custom latent-cache path for older DeepSeek-V2/V3-family MLA modules. `kv_patch.py` searches for attention objects with architecture-specific members including:

```text
kv_a_proj_with_mqa
kv_a_layernorm
kv_b_proj
kv_lora_rank
```

It stores the normalized latent and RoPE-key component in a `DynamicCache`-compatible object, then reconstructs full non-RoPE keys and values through `kv_b_proj` when attention runs.

This is valuable prior work, but it is not a DeepSeek-V4 cache implementation.

## Reusable ideas

### Strongly reusable

- teacher-forced baseline-versus-quantized logit comparison;
- held-out perplexity structure;
- deterministic experiment configuration;
- FP8 e4m3 QDQ implementation as a numerical reference;
- software FP4 e2m1 nearest-value QDQ implementation;
- saved precision-map and sensitivity-score concepts;
- activation collection and result serialization patterns.

### Reusable only after redesign

- precision assignment: use V4 layer/state/group keys rather than one per old latent channel;
- calibration: use downstream perturbation for V4 and separate indexer metrics;
- cache class: rebuild around V4 cache state ownership and append semantics;
- memory report: count real stored values, scales, padding, buffers, and allocator usage.

### Not portable as written

- `kv_patch.py` attention replacement;
- `KVLatentCache` state semantics;
- the old `mean(abs(c_kv)) × norm(kv_b_proj column)` sensitivity score;
- assumptions that `kv_b_proj` exists after the cached representation;
- reconstructing full historical K/V with `kv_b_proj` on each step.

## Concrete repository issues

1. `requirements.txt` contains a malformed dependency line:

   ```text
   nvidia-ml-py>=12.560.30tiktoken
   ```

2. `tiktoken` is duplicated adjacent to that malformed line.

3. `scripts/setup_runtime.sh` and `scripts/download_model.sh` refer to:

   ```text
   tests/verify_kv_relation.py
   ```

   but that file is absent from the archive.

4. Setup/download comments label the project as DeepSeek-V4, while the executable patch and verification code still target DeepSeek-V2/V3-family MLA.

5. `tests/verify_quantized_kv.py` describes its `--model-path` as a DeepSeek-V2-Lite path.

6. The FP8/FP4 functions quantize and then convert back to BF16. The cache therefore remains BF16 storage; reported FP8/FP4 memory is theoretical.

7. Scale treatment is not a deployable append-friendly storage design. Per-channel absmax is computed from the current tensor/update, and real scale-storage overhead is not represented in actual cache allocation.

8. Arbitrary per-channel precision masks are awkward for efficient packed storage and kernels. V4 experiments should start with contiguous groups.

9. The patched attention reconstructs full accumulated K/V from the latent at each decode step. This is suitable for a proof-of-concept quality study, not a final serving implementation.

10. The repository lacks a root README and a reproducible distinction between simulation, real storage, and serving performance.

## Policy for this project

Keep the repository unchanged under `reference/v2_mla_poc/`. Copy individual ideas into new V4-native modules only after adding tests and recording provenance in `docs/REFERENCE_PORT_MAP.md`.
