# CLAUDE.md — DeepSeek-V4-Flash KV-Cache Quantization

## Mission

Build a **DeepSeek-V4-native KV-cache quantization experiment**. Reuse the attached DeepSeek-V2/V3 MLA repository only as reference for quantization functions, calibration concepts, teacher-forced evaluation, and reporting. Do not attempt to plug its cache class or attention patch directly into DeepSeek-V4.

The intended final target is the official Hugging Face model repository:

```text
deepseek-ai/DeepSeek-V4-Flash
```

The final hardware is one RunPod node with four RTX PRO 6000 Blackwell 96 GB GPUs. The current development hardware is an ASUS Ascent GX10 / DGX Spark-class system:

```text
CPU architecture: aarch64
GPU: NVIDIA GB10, compute capability 12.1
Memory: approximately 128 GB unified system/GPU memory
OS: Ubuntu 24.04
CUDA reported by driver: 13.0
```

## Current phase: local GX10 development

This phase must finish the parts that do not require the full checkpoint. Do not download or load the full DeepSeek-V4-Flash weights. Do not attempt to produce final latency, throughput, or multi-GPU claims locally.

### Work permitted locally

- inspect official DeepSeek and Transformers source;
- map all V4 cache states and attention paths;
- construct tiny randomly initialized V4 configurations;
- write deterministic cache-semantics tests;
- implement FP8/FP4 quantize-dequantize simulation;
- implement policy selection and calibration plumbing;
- build quality and benchmark harnesses;
- prototype actual storage layouts in pure PyTorch;
- prepare the RunPod handoff.

### Work deferred to RunPod

- full-checkpoint loading;
- real V4 activation collection and sensitivity ranking;
- final precision-map selection;
- four-GPU correctness and tensor/model parallel validation;
- final memory, prefill, decode, and latency benchmarks;
- target-specific SM120 kernel optimization.

## Sources of truth

Use executable source, runtime assertions, and tests. Do not infer architecture from names alone.

Inspect at minimum:

```text
vendor/DeepSeek-V4-Flash/inference/
vendor/DeepSeek-V4-Flash/config.json
vendor/transformers/src/transformers/models/deepseek_v4/
vendor/transformers/tests/models/deepseek_v4/
reference/v2_mla_poc/
```

Official upstream locations:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/deepseek_v4
- https://github.com/huggingface/transformers/tree/main/tests/models/deepseek_v4

Pin and record exact revisions before relying on the source.

## Hard safety and cost constraints

1. **Never modify `reference/v2_mla_poc/`.** Treat it as read-only evidence.
2. **Never run its setup or download scripts unchanged.** They are not a trustworthy V4 environment bootstrap.
3. **Do not download full model weights on the GX10.** Clone model source with `GIT_LFS_SKIP_SMUDGE=1`.
4. **Do not replace system-wide CUDA or PyTorch during discovery.** First capture the environment and explain any proposed environment change.
5. **Do not copy ARM64 virtual environments or compiled extensions to RunPod.** The cloud environment will be rebuilt for x86-64 and SM120.
6. **Do not edit generated Transformers files blindly.** Prefer external wrappers/subclasses. If modifying Transformers, identify the canonical modular source and regeneration workflow.
7. **Do not claim memory savings from QDQ simulation.** BF16 values after quantize-dequantize still consume BF16 storage.
8. **Do not claim speedups from tiny-model or GX10 timing.** Final performance is measured on the RTX PRO 6000 node.
9. **Do not optimize before correctness.** Pure PyTorch is preferred until semantic and numerical tests pass.
10. **Do not silently skip unsupported dtypes or kernels.** Fail with a clear explanation and record the limitation.

## Architecture hypothesis to verify

DeepSeek-V4 replaces the older V2/V3 MLA cache path with a hybrid design containing sliding-window attention, Compressed Sparse Attention (CSA), Heavily Compressed Attention (HCA), compressor state, and a CSA indexer. The project must identify the precise executable cache write/read paths and not rely on this summary as proof.

The working quantization hypothesis is:

```text
Main KV entries:
    non-RoPE dimensions -> official-style FP8 baseline, experimental FP4/mixed precision
    RoPE dimensions     -> BF16 initially

CSA indexer cache:
    reproduce the official low-precision/QAT-aligned numerical path first

Incomplete compressor buffers, gates, and metadata:
    BF16/full precision initially
```

Verify dimension order, widths, grouping, scale representation, and storage dtype from source and runtime assertions.

## Reference repository: what it is and is not

The supplied code is an older MLA proof of concept. It expects attributes such as:

```text
kv_a_proj_with_mqa
kv_a_layernorm
kv_b_proj
kv_lora_rank
```

It is useful for:

- teacher-forced logit comparison;
- perplexity evaluation structure;
- FP8 and software FP4 QDQ functions;
- sensitivity-analysis organization;
- precision-config serialization concepts.

It is not a V4 implementation. Do not preserve its architecture-specific cache class, patching closure, or sensitivity formula without a fresh justification.

Known issues are documented in `docs/REFERENCE_REPO_AUDIT.md`.

## Required engineering behavior

### Before editing

- inspect relevant files completely;
- identify call sites and state ownership;
- explain the intended change and tests;
- create or update a short task checklist in `PROJECT_STATUS.md`.

### For every meaningful change

- add or update tests;
- run the narrow test first, then the relevant suite;
- record commands and outcomes in `docs/WORKLOG.md`;
- keep commits small and descriptive;
- update documentation when an architectural assumption changes.

### Code quality

- type annotate public functions;
- use deterministic seeds in tests and experiments;
- avoid hidden global state;
- use explicit configuration objects for precision policies;
- separate QDQ simulation from packed/native storage;
- separate calibration data from held-out evaluation data;
- save machine-readable run metadata and results;
- include scale, padding, metadata, and buffer overhead in memory accounting.

## Required project structure

Maintain or evolve this structure:

```text
reference/v2_mla_poc/       # read-only attached repository
vendor/DeepSeek-V4-Flash/   # source-only clone; no weights locally
vendor/transformers/        # pinned source checkout
src/v4_kv_quant/            # project implementation
tests/                      # unit and tiny-model semantic tests
tools/                      # inspection, calibration, evaluation, benchmark CLIs
configs/                    # YAML/JSON experiment configurations
docs/                       # architecture, decisions, worklog, handoff
results/                    # generated experiment outputs; generally gitignored
artifacts/env/              # captured environment reports
```

## Experiment layers

Keep these stages distinct:

### Stage A — baseline semantics

No quantization. Prove full-forward, incremental decode, and chunked prefill agree as expected.

### Stage B — numerical simulation

Quantize and dequantize at the cache boundary, then continue with ordinary computation. This measures quality impact but does not save memory.

### Stage C — actual low-precision storage

Store FP8 or packed FP4 values plus scales and precise slices. Dequantize on read. This measures real memory use and may be slower before kernel fusion.

### Stage D — optimized read path

Dequantize only selected CSA entries or use fused target-specific kernels. This is later work and must not block initial correctness.

## Calibration principles

- Analyze **contiguous groups**, not arbitrary scattered channels, unless there is a compelling implementation reason.
- Begin with groups of 32 or 64 in the non-RoPE portion.
- Use empirical downstream perturbation as the primary sensitivity truth.
- Suitable primary metrics include change in negative log-likelihood and logit KL divergence when one group/state is quantized.
- A gradient-weighted quantization-error score may be used for ranking, but must be validated against actual perturbation.
- For the CSA indexer, measure top-k overlap/recall and downstream loss/logit effects.
- Use unpadded or carefully length-bucketed inputs. Compression boundaries make casual left padding unsafe.
- Save calibration token IDs and use separate held-out data.

## Quality metrics

At minimum collect:

```text
max/mean/RMS logit error
KL divergence
next-token NLL / perplexity
Top-1 agreement
Top-k overlap where relevant
NaN/Inf counts
long-context retrieval accuracy
```

Use teacher forcing for baseline-versus-quantized decode comparisons so that both paths consume identical token histories.

## Benchmark rules

The final RunPod benchmark must compare baseline and quantized variants on the same node, software stack, prompts, batch sizes, context lengths, warmups, and parallel configuration.

Record:

```text
actual cache bytes including scales and padding
peak allocated and reserved memory per GPU
prefill throughput
TTFT
decode throughput
inter-token latency
cache write quantization overhead
cache read/dequantization overhead
```

Report medians over repeated synchronized runs. Never compare GX10 throughput to RTX PRO 6000 throughput as evidence of improvement.

## Local completion gate

Do not declare the DGX phase complete until all requirements in `docs/DGX_PHASE_PLAN.md` pass. When complete:

1. freeze dependency and source revisions;
2. generate a local results manifest;
3. commit all work;
4. tag the repository `dgx-phase-complete-v1`;
5. follow `docs/RUNPOD_HANDOFF_CHECKLIST.md`.

## First task

Execute `prompts/01_ARCHITECTURE_DISCOVERY.md`. Do not start the quantization implementation until that task's required semantic tests and documentation are complete.
