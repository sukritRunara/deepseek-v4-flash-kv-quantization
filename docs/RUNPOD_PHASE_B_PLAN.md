# RunPod Phase B — Execution Plan (4-GPU pod)

*Written 2026-07-17 on the Phase-B pod, branch `runpod-phase-b`. This is the working
checklist for the full-model phase: it binds the execution order in
`RUNPOD_START_HERE.md` (Phase B) and `docs/RUNPOD_HANDOFF_CHECKLIST.md` (steps 1–13) to
this specific pod, records the pre-flight survey, and marks the agreed stop points.
Commands/results go to `docs/WORKLOG.md` as steps complete; decisions to
`docs/DECISIONS.md`.*

## Pod survey (pre-flight, 2026-07-17)

- 4× NVIDIA RTX PRO 6000 Blackwell **Server Edition** 96 GB (97,887 MiB each; CC 12.0 /
  SM120 — same card model as the Phase-A pod), driver 580.126.20 (CUDA 13.0 driver
  surface), all idle at survey time.
- **No NVLink.** All inter-GPU traffic is PCIe. Asymmetric topology: GPU0 on NUMA node 0,
  GPUs 1–3 on NUMA node 3 (GPU0↔others cross the socket interconnect, `SYS`); GPU2↔GPU3
  share a PCIe bridge (`PIX`). Relevant to `device_map=auto` sharding and any
  inter-GPU-transfer interpretation in benchmarks; not a blocker for the single-process
  pipeline-sharded plan.
- 2× Intel Xeon 6952P (384 threads), 1.5 TiB RAM, no swap.
- `/workspace` is a persistent MooseFS network volume (~94 TB free — far above the 500 GB
  budget). Container root disk 150 GB (ephemeral). Expect first-load of weight shards to
  be network-I/O-bound.
- Python 3.12.3 **with dev headers present** (`/usr/include/python3.12/Python.h` — the
  D-010 gate is satisfied). x86-64 confirmed.
- System torch 2.8.0+cu128 exists but is irrelevant: the project venv rebuilds
  torch 2.13.0+cu130 per pins.

### Deviations from runbook assumptions

1. **This pod is not the frozen Phase-A image.** `.venv/`, `vendor/`, and
   `/workspace/models` do not exist here (only the repo is on this volume), so step B0
   is a full `setup_env.sh` rebuild, not the "no-op-ish" rerun the runbook anticipated.
2. **`hf` CLI is not installed** — `scripts/runpod/download_model.sh` checks
   `command -v hf` and will refuse. Fix at step B1: `pip install -U huggingface_hub`
   somewhere on `PATH` (or expose `.venv/bin`), before invoking the download script.
   (`git-lfs` is also absent; not needed — source clones use `GIT_LFS_SKIP_SMUDGE=1` and
   weights come via `hf download` only.)

## Execution checklist

Each step ends with: results recorded in `docs/WORKLOG.md`, artifacts under `results/`
or `artifacts/`, and a commit on `runpod-phase-b`.

- [x] **B0 — environment rebuild + landing gate.** *(2026-07-17: smoke all PASS, landing
      ALL CHECKS PASSED, suite 90 passed — see WORKLOG "Phase B — B0".)*
      `bash scripts/runpod/setup_env.sh` (full run: venv + torch cu130 + vendor clones at
      `configs/source_pins.json` pins + editable installs + env capture + hardware smoke +
      landing test + 90-test suite).
      *Pass:* smoke all required PASS; landing `ALL CHECKS PASSED` (exit 0) against
      `configs/expectations_runpod.json`; suite 90 passed.
- [x] **B1 — weights + baseline generation sanity (GO/NO-GO).** *(2026-07-17: **GO** —
      but only after root-causing faulty pod P2P (D-011, workaround required for ALL
      multi-GPU runs) and adding `kernels==0.15.2`. Weights 149 GB @ pin; coherent
      native-FP8 generation; ~55 GiB headroom/GPU. See WORKLOG "Phase B — B1".)*
      Install `hf` CLI, then
      `RUNPOD_ALLOW_WEIGHTS=1 MODEL_DIR=/workspace/models/DeepSeek-V4-Flash bash
      scripts/runpod/download_model.sh` (~160 GB at the pinned revision), then the
      untouched-model generation snippet from `RUNPOD_START_HERE.md` step 1
      (`device_map=auto`, eager attention, greedy 32 tokens).
      *Pass:* coherent greedy continuation consistent with the upstream integration-test
      snapshot — this is the definitive proof that the checkpoint's native FP8/FP4 weight
      kernels run on SM120.
      *Fail:* stop; fallback decision (BF16 dequant ≈ 600 GB → 8×96 GB pod) goes to
      `docs/DECISIONS.md` before any further spend.
- [x] **B2 — baseline memory/timing.** *(2026-07-17: medians over 5 warmed trials —
      TTFT 0.55/5.7/88.4 s, prefill 1874/1444/742 tok/s, decode ~3 tok/s (ITL ~300 ms;
      host-staged-P2P + Triton-fallback caveats per D-011), cache 13.4/60.5/437 MiB,
      peak alloc 41/47/68 GiB per hottest GPU at 1k/8k/65k. Bring-up findings (chunked
      prefill, expandable_segments) in WORKLOG. `results/benchmark_cache.json`.)*
      `tools/benchmark_cache.py --config configs/bench_runpod_4gpu.json --variants baseline`
      (or `scripts/runpod/launch_4gpu_bench.sh`). Prompt lens 1024/8192/65536, 128 decode
      tokens, 5 trials, medians; per-GPU peaks and cache bytes captured.
- [x] **B3 — instrumentation identity gate on the real model.** *(2026-07-17: ALL
      GATES PASSED bitwise (identity + D-009 storage==qdq) on natural + random prompts,
      after fixing an upstream torch.ldexp missing-device-guard bug found via this gate
      (see WORKLOG). Official-QDQ divergence signal consistent with D-004.)*
      Same CLI, `--variants qdq --policy baseline_bf16`.
      *Pass:* bit-identical logits/picks vs baseline (the Stage-B identity gate holds on
      the real checkpoint). Also re-verify the Stage-C bitwise gate on CUDA/BF16 here
      (D-009 follow-up), before any storage runs.
- [ ] **B4 — real-model calibration. ⚠ STOP FIRST: discuss design with owner before
      running** (corpus choice/size, sequence lengths vs compression boundaries, sweep
      breadth, ranking thresholds). Then build the two deliberately-deferred pieces
      (`docs/REFERENCE_PORT_MAP.md`): a real-corpus streaming loader (C4 pattern from
      `reference/v2_mla_poc/src/calibration_data.py`; unpadded, seeded, token ids saved)
      and a full-model calibration CLI mirroring `tools/run_calibration_smoke.py`.
      Run stats collection + sensitivity sweep (`run_sensitivity_sweep`,
      `enumerate_targets(config, group_size_main=64)`); rank by ΔNLL/KL, indexer by
      top-k overlap (D-004/D-008); build + validate the precision map
      (`build_map_from_sweep`).
- [ ] **B5 — QDQ quality on held-out data.**
      `run_teacher_forced(...)` with the official policy and the B4 candidate map(s);
      metrics via `v4_kv_quant.metrics`; held-out data disjoint from calibration tokens.
- [ ] **B6 — actual low-precision storage.**
      Storage variant runs (harness `storage=True` / benchmark `storage` variant), after
      the CUDA/BF16 bitwise gate from B3 is green.
- [ ] **B7 — final benchmark matrix + capture.**
      All variants × context lengths from `configs/bench_runpod_4gpu.json`, medians,
      same-node-same-run comparisons only; then
      `tools/generate_results_manifest.py`, commit, and record everything in the worklog.

## Working agreements for this phase

- All work on branch **`runpod-phase-b`**; small descriptive commits per step (tests/docs
  updated in the same commit); `main` stays at the Phase-A state until the phase closes.
- Stop points: **B1 fail** (fallback decision with owner) and **B4 start** (calibration
  design discussion with owner) — both explicit above.
- Standing rules from `RUNPOD_START_HERE.md` remain in force: never run
  `reference/v2_mla_poc/scripts/`; never `git lfs pull` in `vendor/DeepSeek-V4-Flash`;
  no vendor re-pins mid-experiment; memory claims include scales/buffers; simulation
  variants save nothing by design and are labeled as such.
