# Task 01 — DeepSeek-V4 Cache Architecture Discovery and Tiny-Model Baseline

## Objective

Determine the exact DeepSeek-V4 cache architecture and establish deterministic tiny-model tests before implementing cache quantization.

This task is **source inspection and baseline correctness only**. Do not download the full checkpoint and do not implement the production quantized cache yet.

## Inputs

1. `reference/v2_mla_poc/`
   - An older DeepSeek-V2/V3-family MLA KV-cache quantization proof of concept.
   - Read-only reference material.
2. `vendor/DeepSeek-V4-Flash/`
   - Official model repository cloned with weight download disabled.
3. `vendor/transformers/`
   - Pinned Hugging Face Transformers checkout containing DeepSeek-V4 support.
4. `artifacts/env/`
   - Environment report from `scripts/capture_environment.sh`.

If a source checkout is absent, clone source only:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash vendor/DeepSeek-V4-Flash
git clone https://github.com/huggingface/transformers.git vendor/transformers
```

Record exact revisions immediately.

## Inspect at minimum

Official model repository:

```text
config.json
inference/README.md
inference/model.py
inference/convert.py
inference/generate.py
```

Transformers:

```text
src/transformers/models/deepseek_v4/configuration_deepseek_v4.py
src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
src/transformers/models/deepseek_v4/modular_deepseek_v4.py, if present
src/transformers/models/deepseek_v4/*
tests/models/deepseek_v4/*
src/transformers/cache_utils.py
```

Reference repository:

```text
kv_patch.py
ops/kv_latent_cache.py
ops/kv_latent_cache_quantized.py
ops/sensitivity.py
src/calibration_data.py
tests/verify_quantized_kv.py
requirements.txt
scripts/setup_runtime.sh
scripts/download_model.sh
```

Search for and trace:

```text
DeepseekV4Attention
DeepseekV4CSACache
DeepseekV4HCACache
CSA compressor
HCA compressor
Indexer
update_compressor_states
cache construction
cache_position
sliding window
compression ratio
RoPE split
FP8 / FP4 QDQ
scale group size
```

Names may differ. Follow actual call graphs.

## Constraints

- Do not edit `reference/v2_mla_poc/`.
- Do not run its setup or model-download scripts.
- Do not download model weights.
- Do not alter system-wide packages without stopping and documenting why.
- Do not edit generated Transformers source directly without identifying the canonical source and regeneration process.
- Do not make claims based only on comments; confirm through executable code and runtime assertions.
- Do not use left-padded inputs in compressor-boundary tests.
- Use deterministic seeds.
- Prefer a tiny randomly initialized model; no real weights are needed.

## Deliverables

### 1. `docs/V4_CACHE_ARCHITECTURE.md`

For every attention/cache path, document:

- layer/attention type;
- source class and method;
- cache owner and state name;
- exact tensor shape convention;
- dtype at write and read;
- persistent versus bounded temporary state;
- write cadence;
- read point;
- compression ratio/window behavior;
- main KV versus indexer state;
- RoPE and non-RoPE dimensions and order;
- scale group and QDQ behavior in official reference code;
- candidate QDQ injection point;
- candidate real-storage injection point;
- uncertainties requiring full-model validation.

Include a compact architecture table and a call-flow diagram in Mermaid or ASCII.

### 2. `docs/REFERENCE_PORT_MAP.md`

Map each relevant reference component to one of:

```text
reuse unchanged
reuse with modification
concept only
not applicable to V4
```

Explain each classification. Explicitly address:

- old cache inheritance;
- old attention patch;
- FP8 function;
- software FP4 function;
- precision config;
- sensitivity score;
- teacher-forced logit harness;
- perplexity evaluation;
- memory reporting.

### 3. `docs/REPRODUCIBILITY.md`

Record:

- host architecture and GPU;
- Python, PyTorch, CUDA, driver, Triton, Transformers versions;
- DeepSeek repository revision;
- Transformers revision;
- commands used to clone and run;
- environment limitations or unsupported features.

### 4. `tools/inspect_v4_cache.py`

Build a tiny random DeepSeek-V4 causal language model configuration that exercises every available V4 attention type. Run baseline forward/cache operations and emit both a readable report and JSON containing:

```text
layer index
layer type
module class
cache class
state name
shape
dtype
device
entry count
persistent/buffer classification
```

The tool must not require model weights or internet after source checkout.

### 5. `tests/test_v4_cache_semantics.py`

Create deterministic tests for all applicable paths:

- full-sequence forward versus token-by-token cached forward;
- one-shot versus multi-chunk prefill;
- prefill -> decode -> later prefill chunk;
- cache reset/reuse;
- exact cache entry counts;
- exact cache state dimensions;
- shared key/value behavior where applicable;
- sliding-window rollover around its boundary;
- CSA compression immediately before, at, and after its boundary;
- HCA compression immediately before, at, and after its boundary;
- runtime verification of RoPE/non-RoPE ordering and widths;
- no unexpected cache mutation when `use_cache=False`;
- deterministic behavior across repeated runs.

Use the actual configured boundary values rather than hardcoding assumptions when possible. If tests cannot cover a path because the upstream tiny configuration disallows it, document the reason and create the closest isolated unit test.

### 6. `tools/hardware_smoke.py`

Report and test:

- CPU architecture;
- Python, PyTorch, CUDA runtime, CUDA toolkit, driver, and Triton versions;
- GPU name and compute capability;
- BF16 support;
- `torch.float8_e4m3fn` availability and a small cast round trip;
- any available FP4 dtype/API and a small operation;
- whether `torch.compile` and Triton can compile a trivial GPU function;
- available unified memory as observed through supported APIs.

Unsupported functionality must be reported clearly, not silently skipped.

### 7. `docs/QUANTIZATION_INJECTION_PLAN.md`

Propose separate insertion points for:

- sliding-window main KV;
- CSA main compressed KV;
- HCA main compressed KV;
- CSA indexer KV/state;
- incomplete compressor buffers;
- scales and metadata.

For each, distinguish:

```text
QDQ simulation insertion
actual low-precision storage insertion
read-side dequantization point
future fused-kernel opportunity
```

State which components should remain BF16 initially and why.

### 8. `PROJECT_STATUS.md` and `docs/WORKLOG.md`

Track tasks, commands, findings, test results, blockers, and decisions. Do not leave important reasoning only in terminal output or chat history.

## Acceptance criteria

This task is complete only when:

1. all source revisions are pinned and recorded;
2. the cache architecture map is supported by source paths and runtime evidence;
3. tiny-model semantic tests pass, or any upstream blocker has a minimal reproduction and documented explanation;
4. the inspection and hardware-smoke tools run from a clean checkout;
5. no full model weights were downloaded;
6. no quantization code was added to production cache paths;
7. the proposed next task is a narrow **official-policy QDQ simulation** implementation.

## Final response format

At completion, report:

```text
Summary
Files created/changed
Commands run
Tests and outcomes
Confirmed architecture findings
Open questions
Environment limitations
Recommended Task 02 scope
```
