# Worklog

## 2026-07-16

### Goal

Execute Task 01 (`prompts/01_ARCHITECTURE_DISCOVERY.md`): normalize the workspace, pin
sources, map the DeepSeek-V4 cache architecture from executable code, and protect the
baseline semantics with deterministic tiny-model tests. No weights, no quantization code.

### Commands run

```bash
# workspace (user-approved layout: starter promoted to repo root)
mv deepseek-v4-kv-dgx-starter/* . && rmdir deepseek-v4-kv-dgx-starter
mv kv-cache-quantization-kimi-k27-main reference/v2_mla_poc && chmod -R a-w reference/v2_mla_poc
bash scripts/capture_environment.sh                      # pre-venv snapshot

# sources (SHAs in docs/REPRODUCIBILITY.md; weights = LFS pointers only, verified)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash vendor/DeepSeek-V4-Flash
git clone --depth 50 https://github.com/huggingface/transformers.git vendor/transformers

# environment (project venv only; system untouched)
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130   # 2.13.0+cu130, CUDA OK on GB10
.venv/bin/pip install numpy safetensors pytest accelerate sentencepiece
.venv/bin/pip install -e vendor/transformers && .venv/bin/pip install -e .
PATH="$PWD/.venv/bin:$PATH" bash scripts/capture_environment.sh   # post-venv snapshot

# deliverables
.venv/bin/python tools/inspect_v4_cache.py               # all runtime assertions PASS
.venv/bin/python tools/hardware_smoke.py                 # required PASS; triton/compile UNSUPPORTED
.venv/bin/python -m pytest tests/test_v4_cache_semantics.py -q    # 27 passed in ~5s
```

### Files changed

- `docs/V4_CACHE_ARCHITECTURE.md`, `docs/REFERENCE_PORT_MAP.md`,
  `docs/QUANTIZATION_INJECTION_PLAN.md`, `docs/REPRODUCIBILITY.md` (new)
- `src/v4_kv_quant/{__init__,tiny_model}.py`, `pyproject.toml` (new package)
- `tools/inspect_v4_cache.py`, `tools/hardware_smoke.py` (new)
- `tests/test_v4_cache_semantics.py` (new, 27 tests)
- `PROJECT_STATUS.md`, `docs/DECISIONS.md`, this file
- `.gitignore` (+`vendor/`), `README.md` (starter version at root)

### Tests and results

- `pytest tests/test_v4_cache_semantics.py`: **27 passed** (fp32 CPU, deterministic seeds).
- `tools/inspect_v4_cache.py`: 29/29 runtime assertions PASS (counts, widths, K=V sharing).
- `tools/hardware_smoke.py`: BF16 / FP8 e4m3 / e8m0 / FP4-x2 dtypes PASS on CPU+GB10;
  `torch._scaled_mm` PASS; `torch.compile` + Triton UNSUPPORTED (missing python3.12-dev —
  root-caused via manual gcc repro; system package deliberately not installed).

### Findings

1. V4 replaces MLA entirely: shared-KV MQA (K=V, 512 = 448 nope + 64 rope, inverse RoPE on
   output), per-layer sliding window 128, CSA (m=4 + top-k-512 indexer) / HCA (m'=128)
   compressed entries appended on the KV axis inside attention.
2. HF cache states per compressed layer: window `keys`(=values), `compressed_kv`,
   `buffer_kv/gate` (< rate tokens), CSA `overlap_kv/gate` (one window Ca slice),
   `entry_count`; all writes flow through `update` / `store_compression_weights` /
   `update_compressor_states` → quantization injects as cache-layer subclasses.
3. Official QDQ policy extracted from `inference/{model,kernel}.py`: FP8 e4m3 g64, ue8m0
   round-up power-of-2 scales, non-rope dims only; indexer: Hadamard + FP4 e2m1 g32 on keys
   AND queries; buffers fp32; rope dims BF16. HF impl has zero QDQ (that gap = Task 02).
4. **Measured**: batched-prefill vs token-by-token decode logits diverge (up to ~1e-1) with
   the selective indexer purely from top-k tie flips under 1e-7 float noise; agree to 2.4e-7
   with a non-selective indexer; MoE routing ruled out. Equality tests therefore use a
   dense-indexer fixture; indexer quantization metric = top-k overlap/recall.
5. `use_cache=True` vs `False` prefill logits are bit-identical; chunked prefill of any
   tested chunking matches one-shot within 2.4e-7 (dense indexer).
6. Stock `DynamicSlidingWindowLayer` (sliding-only layers) duplicates V storage; V4's own
   cache layers alias K=V — a free 2x saving on 3 layers for the Stage-C custom layer.
7. Upstream-documented constraints confirmed in source: eager-attention only (FA head-dim
   cap 256 < 512), left padding unsupported by design, `QuantizedCache` incompatible,
   compressor state non-rewindable (`_is_stateful`).

### Blockers / open questions

- No blockers for Task 02.
- Open (deferred to RunPod): HF<->official numerical parity on real weights; e8m0-vs-fp32
  scale quality; exact-Hadamard vs `fast_hadamard_transform`; YaRN at real context lengths.

### Next step

Task 02: official-policy QDQ simulation as policy-configurable cache-layer subclasses
(scope in `docs/QUANTIZATION_INJECTION_PLAN.md`, gate list in `PROJECT_STATUS.md`).
