# Reproducibility — GX10 Local Phase (Task 01)

Snapshot date: 2026-07-16. Raw machine reports: `artifacts/env/environment-20260716T222305Z.txt`
(pre-venv) and `…20260716T222905Z…` (post-venv); regenerate any time with
`bash scripts/capture_environment.sh` (gitignored outputs — this file records the durable facts).

## Host

| Item | Value |
|---|---|
| Machine | ASUS Ascent GX10 (DGX Spark class), aarch64 |
| OS | Ubuntu 24.04.4 LTS, kernel `6.17.0-1018-nvidia` |
| GPU | NVIDIA GB10, compute capability **12.1**, 121.6 GiB unified memory |
| Driver | 580.159.03 |
| CUDA toolkit (system) | 13.0, V13.0.88 (`nvcc` on PATH) |
| Compilers | gcc/g++ 13.3.0, cmake 3.28.3; **ninja absent**; **git-lfs absent** (harmless: clones keep LFS pointers) |

## Python environment (project venv, no system packages touched)

Created with `python3 -m venv .venv` (system Python 3.12.3). Key packages:

| Package | Version | Source |
|---|---|---|
| torch | **2.13.0+cu130** | `pip install torch --index-url https://download.pytorch.org/whl/cu130` (aarch64/SBSA wheel; CUDA works on GB10, verified) |
| transformers | **5.15.0.dev0** | `pip install -e vendor/transformers` (pinned checkout below — the *only* transformers on the path) |
| triton | 3.7.1 | bundled with torch wheel |
| numpy / safetensors / tokenizers | 2.5.1 / 0.8.0 / 0.22.2 | PyPI |
| accelerate / sentencepiece / pytest | 1.14.0 / 0.2.2 / 9.1.1 | PyPI |
| v4-kv-quant | 0.1.0 (editable) | `pip install -e .` (this repo) |

## Pinned source revisions

| Repo | Revision | How obtained |
|---|---|---|
| `vendor/DeepSeek-V4-Flash` | `60d8d70770c6776ff598c94bb586a859a38244f1` | `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash vendor/DeepSeek-V4-Flash` |
| `vendor/transformers` | `150eb7c9ed4091294c829fa0e9466b090cb0f87f` (main, 2026-07-16, v5.15.0.dev0) | `git clone --depth 50 https://github.com/huggingface/transformers.git vendor/transformers` |
| `reference/v2_mla_poc` | unpacked from `kv-cache-quantization-kimi-k27-main.zip` (no git history in archive), `chmod -R a-w` | supplied archive |

`vendor/` is gitignored; to reconstruct, clone as above and `git checkout <SHA>`.
**No model weights were downloaded**: `vendor/DeepSeek-V4-Flash` is 14 MB; every
`model-*.safetensors` is a 130-byte LFS pointer (spot-verified; largest real file is
`tokenizer.json`). Do not run `git lfs pull` / `git lfs install` in that clone.

## Commands (chronological, condensed)

```bash
# workspace normalization (starter promoted to repo root; reference read-only)
mv deepseek-v4-kv-dgx-starter/* . && rmdir deepseek-v4-kv-dgx-starter
mv kv-cache-quantization-kimi-k27-main reference/v2_mla_poc && chmod -R a-w reference/v2_mla_poc
bash scripts/capture_environment.sh

# source-only clones + pinning
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash vendor/DeepSeek-V4-Flash
git clone --depth 50 https://github.com/huggingface/transformers.git vendor/transformers
git -C vendor/DeepSeek-V4-Flash rev-parse HEAD; git -C vendor/transformers rev-parse HEAD

# project environment
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install numpy safetensors pytest accelerate sentencepiece
.venv/bin/pip install -e vendor/transformers
.venv/bin/pip install -e .
PATH="$PWD/.venv/bin:$PATH" bash scripts/capture_environment.sh

# Task-01 tooling and tests
.venv/bin/python tools/inspect_v4_cache.py          # cache inventory + runtime assertions
.venv/bin/python tools/hardware_smoke.py            # dtype/hardware capability report
.venv/bin/python -m pytest tests/test_v4_cache_semantics.py -q   # 27 passed
```

## Environment limitations (explicit)

1. **Triton kernels and `torch.compile` are unavailable**: both fail at the gcc step because
   `/usr/include/python3.12/Python.h` is missing (`python3.12-dev` system package not
   installed). Installing it is a system-wide change — deliberately not done during discovery
   (CLAUDE.md constraint 4). Remedy when authorized: `sudo apt install python3.12-dev`.
   Impact: none for Task 01–02 (pure PyTorch by design); revisit before any Triton prototype.
2. **`fast_hadamard_transform` and `tilelang` not installed**: the official `inference/` code
   imports them; we read it as reference only and never execute it. The FP4 indexer QDQ will
   use an exact pure-PyTorch H₁₂₈ Hadamard instead (128 = 2⁷).
3. **`torch._scaled_mm` FP8 GEMM works** on GB10 (verified) — available for Stage C/D
   prototypes; `float8_e4m3fn`, `float8_e8m0fnu`, `float4_e2m1fn_x2` all present.
4. GX10 numbers are **development-only**: no latency/throughput/memory-saving claims from this
   machine transfer to the RTX PRO 6000 target (CLAUDE.md constraints 7–8).
5. Transformers is a **dev snapshot** (5.15.0.dev0, main). Do not `git pull` vendor checkouts
   mid-experiment; any deliberate re-pin must be recorded here and in `docs/DECISIONS.md`.
6. No git-lfs on the machine — protects against accidental weight smudge; if it is ever
   installed, re-verify the no-weights invariant (`du -sh vendor/DeepSeek-V4-Flash` ≈ 14 MB).

## Determinism notes

- Tiny-model work: fp32 CPU, `torch.manual_seed` everywhere; repeated runs bit-identical
  (asserted in tests).
- Known non-transferable nondeterminism: CSA indexer top-k tie flips under float noise —
  see `docs/V4_CACHE_ARCHITECTURE.md` §6.8.
