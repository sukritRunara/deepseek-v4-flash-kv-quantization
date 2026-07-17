# Worklog

## 2026-07-17 (RunPod Phase B — B0 environment rebuild, 4-GPU pod)

### Goal

Execute plan step B0 (`docs/RUNPOD_PHASE_B_PLAN.md`): full environment rebuild on the
Phase-B pod — this pod is NOT the frozen Phase-A image (`.venv/`, `vendor/`, weights all
absent; only the repo was on this volume).

### Environment

- Pod: 4× RTX PRO 6000 Blackwell Server Edition 96 GB (CC 12.0/SM120), x86_64,
  2× Xeon 6952P (384 threads), 1.5 TiB RAM, driver 580.126.20, no NVLink (GPU0 on NUMA0,
  GPUs 1–3 on NUMA3), /workspace = MooseFS network volume (~94 TB free).
- Installed by bootstrap: torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.0.dev0
  (editable at pin `150eb7c9ed40`), model source at `60d8d70770c6` (13 MiB LFS pointers).
  Capture: `artifacts/env/*-20260717T163605Z.*`.

### Commands run

```bash
bash scripts/runpod/setup_env.sh   # full run: venv+torch+vendor pins+capture+smoke+landing+suite
```

### Tests and results

- Hardware smoke: all required + all optional PASS (BF16/FP8/FP4 dtypes CPU+CUDA,
  torch.compile, hand-written Triton kernel, `torch._scaled_mm` FP8 GEMM).
- Landing test vs `configs/expectations_runpod.json`: **all 15 checks PASS**, suite
  **90 passed** (99.6 s) → `ALL CHECKS PASSED`, exit 0. Report: `results/landing_test.json`.
- No incompatibilities; no code changes needed.

### Next step

B1: install `hf` CLI (absent on this pod — pre-flight deviation 2), guarded weight
download (~160 GB, pinned revision), then baseline generation sanity = the GO/NO-GO
gate for native FP8/FP4 on SM120.

## 2026-07-17 (RunPod Phase A — 1-GPU landing pod, environment + portability gate)

### Goal

Execute `RUNPOD_START_HERE.md` Phase A on the landing pod: rebuild the environment for
x86-64/SM120, pass hardware smoke + landing test + full suite, confirm the benchmark
measurement path on CUDA, record incompatibilities.

### Environment

- Pod: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB (compute capability 12.0 /
  SM120), x86_64, 256 vCPU / 1.5 TB RAM, driver 580.126.09, network volume on /workspace.
- Image ships CUDA toolkit 12.8 (nvcc) — irrelevant at runtime: the cu130 torch wheel
  bundles the CUDA 13.0 runtime and triton bundles its own ptxas. Default
  `TORCH_INDEX_URL` (cu130) worked; no override needed.
- Python 3.12.3 **with python3.12-dev preinstalled** — the GX10 blocker (Triton-backed
  `torch.bmm` needs `Python.h`) does not exist here; CUDA model execution works.
- Installed: torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.0.dev0 (editable at pin
  `150eb7c9ed40`), model source at `60d8d70770c6` (13 MiB LFS pointers, no weights).
  Capture: `artifacts/env/*-20260717T075537Z.*`.
- Repo state: clean checkout 4 doc-only commits after `dgx-phase-complete-v1`.

### Commands run

```bash
bash scripts/runpod/setup_env.sh   # venv+torch+pins+smoke+landing(+suite); first run: suite 2 failed, 88 passed
.venv/bin/python -m pytest tests/test_benchmark.py -q   # after test fix: 8 passed
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json --run-suite
                                   # ALL CHECKS PASSED (exit 0), suite 90 passed
.venv/bin/python tools/benchmark_cache.py --config configs/bench_tiny_local.json --device cuda   # exit 0
```

### Files changed

- `tests/test_benchmark.py` — the only incompatibility found (see Findings 1)
- `docs/RUNPOD_HANDOFF_CHECKLIST.md` (landing-test boxes ticked), `PROJECT_STATUS.md`,
  this file

### Tests and results

- Hardware smoke: **all required + all optional capabilities PASS** — BF16/FP8
  e4m3/e8m0/FP4-x2 dtypes and round-trips on SM120, `torch._scaled_mm` FP8 GEMM,
  hand-written Triton kernel, `torch.compile` (inductor/triton) on CUDA.
- Landing test: all 15 checks PASS (platform, GPU identity, CC ≥ 12.0, dtypes, vendor
  pins, no weights materialized, python dev headers, tiny forward CPU+CUDA, Stage-C
  bitwise gate). Report: `results/landing_test.json`.
- Suite: first run **2 failed, 88 passed** (host-identity tests, below); after fix
  **90 passed** under `--run-suite` → `ALL CHECKS PASSED`, exit 0.
- Tiny-model CUDA benchmark (batch 2, prompts 64/256, 32 decode tokens, 5 trials,
  medians; fp32, `reference_official_qdq` policy) — machinery validation only,
  NON-TRANSFERABLE numbers: baseline/qdq/storage all ran end to end on CUDA; storage
  cache bytes 9.8 vs 22.0 KiB (prompt 64) and 18.6 vs 46.0 KiB (prompt 256) with QDQ
  == baseline bytes as designed; QDQ/storage decode ~10–20% slower than baseline
  (expected pre-fusion). Micro-overheads recorded (fp8 encode window step ~74 µs,
  fp4 indexer entry encode ~209 µs). Output: `results/benchmark_cache.json`.

### Findings

1. **Only incompatibility: the two landing-check identity tests hardcoded the GX10 as
   "this machine"** (`test_landing_checks_pass_with_local_expectations` asserted the
   GX10 expectations pass; `..._fail_cleanly_with_runpod_expectations` asserted the
   RunPod expectations fail). On the pod both inverted. Fixed by selecting the
   expectations file from `platform.machine()` — "this host's expectations PASS, the
   other host's FAIL cleanly" — so the suite is green on both hosts with unchanged
   intent. No product-code change was needed anywhere.
2. Everything Phase A is allowed to prove about SM120 is proven: FP8/FP4 dtypes and
   casts, FP8 GEMM, Triton compile path, full tiny V4 architecture execution on GPU
   (eager attention, compressors, indexer, all three cache variants, Triton-backed
   `torch.bmm`), Stage-C bitwise gate on this dtype stack. Real-checkpoint quantized
   kernels remain unprovable on 96 GB (Phase B step 1 is the definitive check).
3. cu130 wheel on a CUDA-12.8 image is fine (driver 580.x provides the 13.0 runtime
   surface); worth knowing when picking Phase B images.
4. Compute capability reports **12.0** here vs 12.1 on the GX10 GB10 — the
   `min_compute_capability: [12, 0]` expectation is correct as written.

### Blockers / open questions

- None for Phase B. Remaining Phase A checklist items are operator actions:
  freeze the pod image, stop the pod.

### Next step

Phase B (4× GPU pod): reuse frozen image; budget the first hour for step 1 baseline
generation sanity — native FP8/FP4 inference on SM120 — before committing to the full
experiment plan (fallback: BF16 dequant on 8×96 GB).

## 2026-07-17 (Local completion gate — DGX/GX10 phase closed)

### Goal

Walk the completion gate in docs/DGX_PHASE_PLAN.md and the handoff preflight, then tag.

### Commands run

```bash
.venv/bin/pip freeze > docs/PIP_FREEZE_GX10.txt
cp results/calibration_smoke/*.json artifacts/calibration_smoke/
.venv/bin/python tools/generate_results_manifest.py        # 23 files -> docs/results_manifest.json
.venv/bin/python -m pytest tests/ -q                       # 90 passed
# fresh-worktree validation (catches dependence on uncommitted files):
git worktree add /tmp/dgx-gate-check HEAD
PYTHONPATH=/tmp/dgx-gate-check/src .venv/bin/python -m pytest /tmp/dgx-gate-check/tests -q
git worktree remove /tmp/dgx-gate-check
git tag -a dgx-phase-complete-v1
```

### Files changed

- `docs/PIP_FREEZE_GX10.txt` (frozen GX10 versions; reference only)
- `docs/results_manifest.json` + `tools/generate_results_manifest.py`
- `artifacts/calibration_smoke/` (committed calibration fixtures: token ids, precision
  map, sensitivity records, held-out metrics, stats)
- `docs/DGX_PHASE_PLAN.md` (all boxes checked; gradient-weighted ranking left open as
  explicitly optional/deferred), `docs/RUNPOD_HANDOFF_CHECKLIST.md` (preflight complete),
  `PROJECT_STATUS.md`

### Tests and results

Full suite 90 passed on the working tree AND from a clean git worktree of the final
commit. No weights (vendor model tree 14 MB pointers). Working tree clean at tag time.

### Findings

Gate closed with one explicitly-deferred optional item (gradient-weighted ranking) and one
documented environment limitation (GX10 CUDA blocked pending python3.12-dev — D-010).

### Next step

RunPod one-GPU landing pod per docs/RUNPOD_HANDOFF_CHECKLIST.md.

## 2026-07-17 (Task 05 — benchmark harness + RunPod tooling, Phase 6)

### Goal

Config-driven benchmark CLIs (baseline/QDQ/storage), source-only landing test, guarded
RunPod launch scripts — one command structure for tiny-local and full-model-RunPod.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_benchmark.py -q                      # 8 passed
.venv/bin/python -m pytest tests/ -q                                       # 90 passed
.venv/bin/python tools/benchmark_cache.py --config configs/bench_tiny_local.json
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_gx10.json    # exit 0
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json  # exit 1 here (correct)
bash scripts/runpod/setup_env.sh                    # REFUSING (aarch64 guard) - correct
RUNPOD_ALLOW_WEIGHTS=1 bash scripts/runpod/download_model.sh   # REFUSING - correct
```

### Files changed

- `prompts/05_BENCHMARK_RUNPOD_TOOLING.md`
- `src/v4_kv_quant/bench.py` (BenchSettings from JSON; identical token streams across
  variants; warmup+trials; TTFT/prefill/decode/ITL; per-GPU peaks; cache bytes; QDQ
  overhead micro-bench), `src/v4_kv_quant/landing.py` (expectation-driven checks)
- `tools/benchmark_cache.py`, `tools/runpod_landing_test.py`
- `configs/{bench_tiny_local,bench_runpod_4gpu,expectations_gx10,expectations_runpod,source_pins}.json`
- `scripts/runpod/{setup_env,download_model,launch_4gpu_bench}.sh` (x86_64-guarded;
  weights additionally behind RUNPOD_ALLOW_WEIGHTS=1 + free-disk check + pinned revision)
- `tests/test_benchmark.py`; `docs/REPRODUCIBILITY.md` limitation 1 upgraded

### Tests and results

90/90 pass. Local tiny benchmark (CPU, fp32, non-transferable): storage cache 9.8/18.6 KiB
vs baseline 22/46 KiB at prompt 64/256; QDQ==baseline bytes; quantized variants slower as
expected for pure-PyTorch (no claims). Landing test: GX10 expectations all PASS
(2 truthful WARNs), RunPod expectations fail on platform/GPU/dev-headers here — proving
the expectation mechanism.

### Findings

1. **GX10 CUDA model execution is blocked by the missing python3.12-dev**: torch 2.13
   routes CUDA `torch.bmm` through a Triton-backed `torch._native` kernel, and V4's
   grouped output projection is bmm-based — first CUDA forward of the tiny model
   surfaced it. CPU unaffected. Landing test now carries an expectation-driven
   `cuda_model_forward` check (WARN on GX10, FAIL on RunPod); local bench config pins CPU.
2. CUDA peak-stats APIs error if the CUDA context is uninitialized — benchmark guards all
   `torch.cuda` calls on the *benchmark device*, not on `is_available()`.

### Blockers / open questions

- `python3.12-dev` install (system package) needed before ANY GPU-side validation on the
  GX10; decision deferred to the owner (D-005/D-010). Not required for handoff — RunPod
  images ship dev headers and the landing test enforces it there.

### Next step

Local completion gate: freeze, manifest, tag `dgx-phase-complete-v1`, handoff preflight.

## 2026-07-17 (Task 04 — Stage-C actual-storage prototype, Phase 5)

### Goal

Real low-precision cache storage (FP8 codes + e8m0 scales; packed FP4 nibbles) with
pure-PyTorch dequantize-on-read, bitwise-faithful to the Stage-B QDQ simulation, plus
honest memory accounting demonstrating actual byte reduction.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_actual_storage.py -q   # 15 passed (first run)
.venv/bin/python -m pytest tests/ -q                         # 82 passed
.venv/bin/python tools/measure_cache_memory.py               # results/cache_memory.json
```

### Files changed

- `prompts/04_ACTUAL_STORAGE_PROTOTYPE.md` (spec + gates)
- `src/v4_kv_quant/storage.py` (fp8_store/fp4_store/load with bitwise QDQ parity;
  e8m0 1-byte scales for power-of-2, fp32 fallback; sign+3-bit e2m1 codes, 2 per uint8)
- `src/v4_kv_quant/storage_cache.py` (QuantStore append/trim/decode; storage layers for
  sliding/HCA/CSA; QuantizedStorageCache; keys/values stay empty placeholders)
- `src/v4_kv_quant/memory.py` (logical + allocator storage bytes; K=V alias once;
  stock sliding V-duplication flagged; per-state itemization)
- `src/v4_kv_quant/harness.py` (`storage=True` runs the Stage-C cache for a policy)
- `tools/measure_cache_memory.py`, `tests/test_actual_storage.py`

### Tests and results

82/82 pass. Memory comparison (tiny fp32 model, reference_official_qdq, 112 tokens):

| cache | logical bytes | vs baseline |
|---|---|---|
| baseline_bf16_stock | 24,576 | 1.000x |
| stage_b_qdq_sim | 24,576 | 1.000x (simulation saves nothing — proven, not claimed) |
| stage_c_storage | 10,758 | **0.438x** (per-layer 0.223 / 0.485 / 0.445) |

Itemization confirms: fp8 codes 1 B, e8m0 scales 1 B/group, raw fp32 rope slices, packed
uint8 indexer nibbles (2, 28, 8) for 16 logical channels, fp32 buffers/overlap intact.
Sliding layer gains extra from removing the stock duplicated V copy.

### Findings

1. **Stage-C == Stage-B bitwise at model level** (logits and indexer picks) for both
   fp8-main and full official policies — every Stage-B quality result transfers to real
   storage unchanged. This is the load-bearing Stage-C correctness gate (D-009).
2. e8m0/fp8 tensor dtypes support cat/slice/contiguous on CPU in torch 2.13 — no uint8
   view workarounds needed for storage bookkeeping.
3. Window trim must re-contiguate: the stock dynamic layers keep views into concatenated
   history (storage_bytes > logical_bytes, visible in the report as 35,840 vs 24,576);
   Stage-C stores are trimmed contiguous, so allocator bytes track logical bytes.
4. fp32 tiny-model ratios overstate real savings (fp8 vs fp32 = 4x on quantized slices);
   BF16 checkpoint gives 2x there. Accounting validated; final ratios come from RunPod.

### Blockers / open questions

None. Decode-path latency is expectedly worse than baseline (pure-PyTorch dequant per
forward); not measured on GX10 by design — Stage-D fusion is target-hardware work.

### Next step

Task 05: benchmark harness + RunPod tooling (Phase 6), then the local completion gate.

## 2026-07-16 (Task 03 — calibration and precision-policy plumbing, Phase 4)

### Goal

Target taxonomy, per-group precision map + consumer cache, activation stats, one-target
empirical perturbation sweep, map builder, smoke calibration with held-out evaluation.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_calibration.py -q   # 13 passed (first run)
.venv/bin/python -m pytest tests/ -q                      # 67 passed
.venv/bin/python tools/run_calibration_smoke.py           # results/calibration_smoke/
```

### Files changed

- `prompts/03_CALIBRATION_PLUMBING.md` (spec + gates)
- `src/v4_kv_quant/targets.py` (QuantTarget, enumerate_targets; indexer = state-level target)
- `src/v4_kv_quant/precision_map.py` (MapEntry/PrecisionMap v1, validation, JSON)
- `src/v4_kv_quant/mapped_cache.py` (MappedQDQCache + per-group write-boundary QDQ +
  indexer query context from map)
- `src/v4_kv_quant/stats.py` (pass-through StatsCollectorCache: amax/mean|x|/RMS,
  per-group amax, FP8/rotated-FP4 QDQ-error RMS)
- `src/v4_kv_quant/sensitivity.py` (measure_target, run_sensitivity_sweep,
  build_map_from_sweep with fp8/fp4 fractions + indexer overlap threshold)
- `src/v4_kv_quant/harness.py` (accepts `precision_map=` alongside `policy=`)
- `tools/run_calibration_smoke.py`, `tests/test_calibration.py`

### Tests and results

67/67 pass. Smoke run (tiny random model, 16 targets, group_size_main=8, S=48/prefill 24):

- stats: indexer rotated-FP4 QDQ error RMS 1.1e-1 vs main-KV FP8 2.5–2.8e-2 (~4x) — expected;
- sensitivity ranking: indexer target most sensitive (score 1.6e-3, ~4x above the top main
  target); layer-0 window groups next (earliest layer, most downstream amplification);
  compressed-KV groups least sensitive on this random model;
- built map: 11x fp8_e4m3 + 1x indexer fp4_hadamard, 4 most-sensitive groups left BF16;
- held-out eval: top-1 agreement 0.9375, KL 6.3e-5, dNLL -4.4e-4, no NaN/Inf.

All numbers are random-weight machinery validation — not transferable to the checkpoint.

### Findings

1. Single-entry precision map == perturbation experiment; full map == deployable policy —
   one consumer (`MappedQDQCache`) serves calibration and deployment, cross-validated
   bitwise against the Task-02 policy cache on full coverage.
2. Indexer calibrated as ONE state-level target: Hadamard rotation mixes all channels, so
   per-group granularity in the original basis has no deployable meaning (D-008).
3. Gradient-weighted ranking deferred (optional per CLAUDE.md); empirical perturbation is
   the primary truth and the harness makes each target measurement ~1 s on the tiny model.

### Blockers / open questions

None. Real-corpus calibration data (C4 port) and real-weight sensitivity are RunPod work.

### Next step

Task 04: Stage-C actual-storage prototype (Phase 5) — see PROJECT_STATUS.md.

## 2026-07-16 (Task 02 — official-policy QDQ simulation, Stage B)

### Goal

Reproduce the official QAT-aligned QDQ numerics at the Transformers cache write boundary,
selected purely by a serializable policy. Simulation only — no memory savings.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_qdq_simulation.py -q   # 27 passed
.venv/bin/python -m pytest tests/ -q                         # 54 passed
.venv/bin/python tools/run_qdq_simulation.py                 # results/qdq_simulation.json
```

### Files changed

- `prompts/02_OFFICIAL_POLICY_QDQ_SIMULATION.md` (task spec + acceptance gates)
- `src/v4_kv_quant/qdq.py` (FP8 e4m3 QDQ g64 + exact ue8m0 round-up scales via frexp;
  software FP4 e2m1 RNE g32; orthonormal FWHT; effective_group_size with no silent fallback)
- `src/v4_kv_quant/policy.py` (StatePolicy/KVQuantPolicy v1, JSON round trip, 5 presets)
- `src/v4_kv_quant/qdq_cache.py` (QDQSlidingWindowLayer / QDQHCACacheLayer / QDQCSACacheLayer,
  QDQCache container, indexer-query scorer wrapper context manager)
- `src/v4_kv_quant/{metrics,harness}.py` (logit/KL/NLL/top-1 + indexer top-k overlap;
  teacher-forced runner)
- `tools/run_qdq_simulation.py`, `tests/test_qdq_simulation.py`

### Tests and results

54/54 pass. Tool output (tiny RANDOM-weight model, fp32 CPU, S=48, prefill 24 — machinery
validation and relative ordering only):

| policy | max abs dlogit | KL mean | top-1 agree | indexer overlap |
|---|---|---|---|---|
| baseline_bf16 | 0.0 (bit-exact) | 0.0 | 1.0000 | 1.0000 |
| main_fp8_nonrope_rope_bf16 | 3.8e-2 | 7.9e-6 | 0.9688 | 0.9944 |
| reference_official_qdq | 1.7e-1 | 5.3e-5 | 0.9479 | 0.9444 |
| main_fp4_nonrope_rope_bf16 (exp) | 1.8e-1 | 1.4e-4 | 0.8958 | 0.9389 |
| indexer_reference_qdq | 1.2e-1 | 4.7e-5 | 0.9583 | 0.9333 |

Ordering is sensible: FP8 mildest; FP4-on-main worst; indexer FP4 shows up mainly as top-k
overlap loss (0.93) — consistent with the Task-01 tie-instability finding.

### Findings

1. torch 2.13 cannot cast to `float4_e2m1fn_x2` (storage-only; "copy_kernel not implemented")
   → software e2m1 grid with RNE ties (tie table 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2,
   3.5→4, 5→4), verified idempotent and grid-exact.
2. `CacheLayerMixin.__init_subclass__` re-registers subclasses by inherited `_layer_type` —
   QDQ subclasses MUST set `_layer_type = None` or they silently hijack every
   `DynamicCache(config=…)` in the process (test-pinned).
3. Indexer entries are stored in the Hadamard-ROTATED basis and queries rotated symmetrically
   via a scorer wrapper — matches official model.py:414-420; FWHT dot-product preservation
   verified to 1e-4.
4. QDQ at the write boundary is chunking-invariant: chunked/one-shot/token-by-token agree to
   Task-01 tolerances under FP8 policy (per-token row groups + per-entry emission).

### Blockers / open questions

None for local work. Real-checkpoint validation of all quality numbers is a RunPod gate.

### Next step

Task 03: calibration plumbing (Phase 4) — see PROJECT_STATUS.md.

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
