# RunPod — Start Here

Single runbook for continuing this project on RunPod. The GX10 phase is closed at tag
`dgx-phase-complete-v1`; everything you need is in this repository — `vendor/` sources and
the Python environment are rebuilt from pinned revisions by the scripts below (never copy
them from the GX10; they are ARM64).

Full background: `docs/RUNPOD_HANDOFF_CHECKLIST.md` (authoritative checklist),
`PROJECT_STATUS.md` (what exists), `docs/REPRODUCIBILITY.md` (pins and limitations).

---

## Phase A — cheap one-GPU landing pod (do this before renting 4 GPUs)

**Provision:** 1× RTX PRO 6000 Blackwell 96 GB, x86-64, ≥100 GB disk (no weights in this
phase). Base image: a CUDA 13.x devel image with Python 3.10+ **including python3-dev
headers** and git. Attach the persistent volume you intend to keep.

```bash
# 1. clone at the handoff tag
git clone --branch dgx-phase-complete-v1 https://github.com/sukritRunara/deepseek-v4-flash-kv-quantization.git
cd deepseek-v4-flash-kv-quantization

# 2. one-shot bootstrap (refuses on non-x86_64):
#    venv + torch (cu130 wheel; override TORCH_INDEX_URL if the image needs another CUDA line)
#    + vendor clones at pinned SHAs (source-only, no weights) + editable installs
#    + environment capture + hardware smoke + landing test + full 90-test suite
bash scripts/runpod/setup_env.sh
```

**Success looks like:** hardware smoke "All required capabilities available", landing test
`ALL CHECKS PASSED` (exit 0), suite `90 passed`. The landing test hard-fails if: wrong
platform/GPU, missing FP8/FP4 dtypes, vendor SHAs not matching `configs/source_pins.json`,
weights unexpectedly present, or **missing python dev headers** (on this torch build,
CUDA `torch.bmm` runs through a Triton kernel that compiles against `Python.h` — a pod
without dev headers cannot run the model on GPU at all; fix: `apt install python3-dev`
for the image's Python, then rerun).

```bash
# 3. re-run pieces individually if anything failed
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json --run-suite
.venv/bin/python tools/hardware_smoke.py

# 4. optional: tiny-model benchmark to confirm the measurement path on this hardware
.venv/bin/python tools/benchmark_cache.py --config configs/bench_tiny_local.json --device cuda

# 5. record any incompatibilities + fixes as commits, freeze the image, stop the pod
```

**Driving this with Claude Code:** install/launch `claude` on the pod from the repo root
(it auto-loads `CLAUDE.md`) and tell it: *"Read RUNPOD_START_HERE.md and execute Phase A;
record outcomes in docs/WORKLOG.md."* The guards and the landing test make this safe to
run hands-off; anything it may not do (weight download) is gated behind an env var it
won't set on its own.

### What Phase A proves about SM120 — and what it deliberately can't

Proven on the 1-GPU pod (all from the steps above):
- FP8 e4m3 / e8m0 / FP4 dtype support and cast round-trips on SM120 silicon;
- `torch._scaled_mm` FP8 GEMM, a hand-written Triton kernel, and `torch.compile`
  (this also implicitly verifies the pod has python dev headers);
- full V4 *architecture* execution on GPU — the tiny-model suite and CUDA benchmark
  exercise eager attention, compressors, indexer, all three cache variants, and the
  Triton-backed `torch.bmm` path end to end;
- the Stage-C bitwise gate (storage == QDQ simulation) on this hardware/dtype stack.

NOT provable here: the **quantized-weight kernels of the real checkpoint** (FP4 expert
grouped-GEMM / DeepGEMM / fp8_linear dispatch). Those need the actual weights, and
~160 GB does not fit one 96 GB card — this is why the checklist calls the landing pod an
architecture-portability test only. The definitive native-format check is **Phase B
step 1** (baseline generation on the 4-GPU pod): budget one hour for it before committing
to the full experiment plan — if native FP8/FP4 inference misbehaves on SM120 there, the
fallback is BF16 dequantization (~600 GB), which needs an 8×96 GB pod instead of 4.

---

## Phase B — four-GPU pod (full model)

**Provision:** 4× RTX PRO 6000 Blackwell 96 GB, one node, persistent volume ≥500 GB
(original weights ~160 GB + outputs + headroom). Reuse the frozen Phase-A image.

```bash
cd deepseek-v4-flash-kv-quantization
bash scripts/runpod/setup_env.sh                      # no-op-ish if image was frozen
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json

# weights (guarded: requires explicit opt-in; pinned to the SHA in configs/source_pins.json)
RUNPOD_ALLOW_WEIGHTS=1 MODEL_DIR=/workspace/models/DeepSeek-V4-Flash \
  bash scripts/runpod/download_model.sh
# if you used a custom MODEL_DIR, update "model_path" in configs/bench_runpod_4gpu.json
```

### Execution order (maps checklist steps → existing tools)

1. **Baseline generation sanity** (untouched model, device_map=auto over 4 GPUs):
   ```bash
   .venv/bin/python - <<'PY'
   from transformers import AutoModelForCausalLM, AutoTokenizer
   path = "/workspace/models/DeepSeek-V4-Flash"
   tok = AutoTokenizer.from_pretrained(path)
   model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="auto",
                                                attn_implementation="eager")
   ids = tok("Pipeline parallelism in ai is ", return_tensors="pt").to(model.device)
   print(tok.decode(model.generate(**ids, max_new_tokens=32, do_sample=False)[0]))
   PY
   ```
   (Expected continuation snapshot: upstream integration test in
   `vendor/transformers/tests/models/deepseek_v4/test_modeling_deepseek_v4.py`.)

2. **Baseline memory/timing** — same CLI as local, config-driven:
   ```bash
   .venv/bin/python tools/benchmark_cache.py --config configs/bench_runpod_4gpu.json --variants baseline
   # or the wrapper (runs landing test first): bash scripts/runpod/launch_4gpu_bench.sh
   ```

3. **Cache instrumentation without quantization** (identity gate on the real model —
   QDQ cache with the all-BF16 policy must be bit-identical to baseline):
   ```bash
   .venv/bin/python tools/benchmark_cache.py --config configs/bench_runpod_4gpu.json \
     --variants qdq --policy baseline_bf16
   ```

4. **Real-model calibration** — the APIs are model-agnostic
   (`v4_kv_quant.sensitivity.run_sensitivity_sweep`, `enumerate_targets(config,
   group_size_main=64)`, `v4_kv_quant.stats.StatsCollectorCache`), but two small pieces
   are deliberately RunPod-side work (see `docs/REFERENCE_PORT_MAP.md`):
   * a real-corpus loader (port the C4 streaming pattern from
     `reference/v2_mla_poc/src/calibration_data.py`; unpadded, seeded, token ids saved);
   * a full-model calibration CLI wiring the loader to the sweep (mirror
     `tools/run_calibration_smoke.py`, which shows the exact sequence on the tiny model).
   Rank with ΔNLL/KL, indexer targets by top-k overlap; build the map with
   `build_map_from_sweep(...)` and validate it against the real config.

5. **QDQ quality on held-out data** — `v4_kv_quant.harness.run_teacher_forced(model, ids,
   prefill_len, policy=...)` or `precision_map=...`; metrics via `v4_kv_quant.metrics`.

6. **Actual low-precision storage** — `storage=True` in the harness or the `storage`
   benchmark variant. Re-verify the Stage-C bitwise gate on CUDA/BF16 first
   (`tools/runpod_landing_test.py` runs a CPU version; D-009 asks for a CUDA re-check).

7. **Final benchmark matrix** — all variants, several context lengths, medians:
   ```bash
   .venv/bin/python tools/benchmark_cache.py --config configs/bench_runpod_4gpu.json
   ```
   Compare only within one node/run (CLAUDE.md benchmark rules). Capture outputs:
   ```bash
   .venv/bin/python tools/generate_results_manifest.py && git add -A && git commit
   ```

---

## Rules that still apply on RunPod

- Never run anything from `reference/v2_mla_poc/scripts/` (read-only reference).
- Never `git lfs pull` inside `vendor/DeepSeek-V4-Flash` — weights go to `MODEL_DIR` via
  the guarded download script only.
- Don't update `vendor/` checkouts mid-experiment; any re-pin goes through
  `configs/source_pins.json` + `docs/DECISIONS.md`.
- Memory claims must include scales/buffers (use `v4_kv_quant.memory`); simulation (QDQ)
  variants save nothing by design.
- Record commands/results in `docs/WORKLOG.md`; decisions in `docs/DECISIONS.md`.

## Environment variables (scripts)

| Variable | Script | Default | Meaning |
|---|---|---|---|
| `TORCH_INDEX_URL` | setup_env.sh | cu130 wheel index | torch build for the image's CUDA |
| `RUNPOD_ALLOW_WEIGHTS` | download_model.sh | unset (refuses) | must be `1` to download |
| `MODEL_DIR` | download_model.sh | /workspace/models/DeepSeek-V4-Flash | weight location |
| `MIN_FREE_GB` | download_model.sh | 200 | free-disk guard |
| `BENCH_CONFIG` | launch_4gpu_bench.sh | configs/bench_runpod_4gpu.json | benchmark config |
| `EXPECTED_GPUS` | launch_4gpu_bench.sh | 4 | GPU-count guard |
