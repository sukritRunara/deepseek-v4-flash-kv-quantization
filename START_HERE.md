# Start Here — DeepSeek-V4-Flash KV-Cache Quantization on the GX10

This pack is the starting point for the **local DGX/GX10 phase**. The objective is to finish the architecture discovery, tiny-model correctness work, quantization simulation, and experiment plumbing **before** renting the four RTX PRO 6000 GPUs.

## What to copy to the GX10

Copy this entire directory and the supplied reference archive:

```text
kv-cache-quantization-kimi-k27-main.zip
```

A sensible project location is:

```bash
mkdir -p ~/projects
cp -r deepseek-v4-kv-dgx-starter ~/projects/deepseek-v4-kv-quant
cd ~/projects/deepseek-v4-kv-quant
```

Then unpack the reference repository without modifying it:

```bash
bash scripts/bootstrap_workspace.sh /path/to/kv-cache-quantization-kimi-k27-main.zip
```

Capture the machine and software state:

```bash
bash scripts/capture_environment.sh
```

Initialize version control:

```bash
git init
git add .
git commit -m "Initialize DeepSeek V4 KV quantization project"
```

## Start Claude Code

From the project root:

```bash
claude
```

Give Claude this instruction:

```text
Read CLAUDE.md in full. Then execute prompts/01_ARCHITECTURE_DISCOVERY.md.
Begin by presenting a concise execution plan. Do not download model weights, do not
run the reference repository's setup or download scripts, and do not alter system-wide
packages during this first task.
```

`CLAUDE.md` is the persistent project instruction file. `prompts/01_ARCHITECTURE_DISCOVERY.md` is the first bounded task.

## Local-phase rule

Do **not** download or try to load the full DeepSeek-V4-Flash checkpoint on the GX10. Clone source repositories with Git LFS weight downloads disabled. This phase is for:

- source inspection;
- architecture mapping;
- deterministic tiny-model tests;
- FP8/FP4 quantize-dequantize simulation;
- calibration and evaluation harness development;
- storage-layout prototypes;
- RunPod scripts and handoff preparation.

The real model calibration, full-model quality testing, four-GPU execution, and final performance benchmarks belong on RunPod.

## Recommended source-only clones

Claude should do this after inspecting the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash vendor/DeepSeek-V4-Flash
git clone https://github.com/huggingface/transformers.git vendor/transformers
```

Record the exact revisions in `docs/REPRODUCIBILITY.md`. Do not casually update either repository after experiments begin.

## What “done locally” means

The GX10 phase is complete only when:

1. V4 cache states and source paths are documented.
2. Tiny-model full-forward, cached decode, and chunked-prefill tests pass.
3. Official-style FP8/BF16 and indexer FP4 simulation paths work.
4. The calibration pipeline produces a precision policy from a tiny-model smoke run.
5. The benchmark harness runs end to end on a tiny model.
6. RunPod commands, environment assumptions, and output formats are documented.
7. The repository is committed and tagged for handoff.

See `docs/DGX_PHASE_PLAN.md` for the complete gate.
