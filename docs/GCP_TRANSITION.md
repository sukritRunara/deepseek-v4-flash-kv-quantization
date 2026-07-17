# GCP Transition — Phase B continues on a G4 instance (D-013)

*Owner-approved 2026-07-17. RunPod Phase-B pod sealed at this commit; work resumes on a
GCP `g4-standard-48` (4× RTX PRO 6000 Blackwell Server Edition 96 GB — same silicon as
the RunPod pod, so all SM120 validation carries over). Reason: GCP credits + the RunPod
host's faulty PCIe P2P (D-011).*

## State at handoff (what is DONE)

- B0 ✓ (env gates), B1 ✓ GO (native FP8/FP4 on SM120), B2 ✓ (baseline matrix —
  RunPod-node numbers, NOT comparable to GCP; per-trial data in
  `artifacts/phase_b_runpod/`), B3 ✓ (all bitwise gates on the real model).
- B4 design approved (D-012), corpus loader + staged CLI committed and unit-tested;
  **no B4 outputs exist yet** — run 1 (corpus,stats,screening) starts fresh on GCP.
- Hard-won environment facts: see D-011 (P2P), WORKLOG B2 (expandable_segments,
  chunked prefill, cache pre-warm), WORKLOG B3 (torch.ldexp upstream bug, fixed in
  qdq.py), WORKLOG B1 (`kernels==0.15.2`).

## Bring-up checklist on the GCP instance

Provisioning: `g4-standard-48`, boot disk ≥ 500 GB, Blackwell-capable NVIDIA driver
(GCP G4 default image), `python3-dev` + `git` present, GitHub credentials configured.

1. `git clone -b runpod-phase-b https://github.com/sukritRunara/deepseek-v4-flash-kv-quantization.git`
2. `bash scripts/runpod/setup_env.sh` — venv + pins + kernels + datasets + smoke +
   landing test (`configs/expectations_runpod.json` should pass as-is: same GPU
   name/CC; if the reported name differs, add an `expectations_gcp.json`) + full suite
   (95 expected).
3. **`python tools/p2p_stress_check.py` FIRST** (both phases must be OK). If it passes
   natively, make `ensure_host_staged_p2p()` opt-in (env var) and record in DECISIONS;
   if it fails, keep the D-011 workaround exactly as wired.
4. Weights: `pip install -U huggingface_hub`, then
   `RUNPOD_ALLOW_WEIGHTS=1 MODEL_DIR=<path> bash scripts/runpod/download_model.sh`
   (update `model_path` in `configs/bench_runpod_4gpu.json` if not
   `/workspace/models/DeepSeek-V4-Flash`).
5. Re-validation (cheap insurance, one model load each):
   a. B1 generation sanity (scriptlet in `RUNPOD_START_HERE.md` step 1 + p2p decision
      from step 3) — expect the coherent pipeline-parallelism continuation;
   b. B3 gates rerun (identity + storage==qdq bitwise; driver preserved at
      `artifacts/phase_b_runpod/b3_gates_run2.log` for reference — reconstruct from
      WORKLOG B3 or the harness API).
6. Resume **B4 run 1**: `tools/run_calibration_full.py --stages corpus,stats,screening`
   (background; ~1 h), then owner reviews rankings → refine/fp8spot → map → heldout
   (D-012 fractions discussion).
7. B5–B7 per `docs/RUNPOD_PHASE_B_PLAN.md`. B7 includes the baseline variant, which
   regenerates the node-local baseline (B2-GCP) automatically — never compare against
   the RunPod numbers.

## Notes

- Weight loading: if the model dir is on a network/PD volume, pre-warm the page cache
  (parallel `dd` of the shards) before loads — see WORKLOG B1 finding 1.
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on this stack
  (WORKLOG B2 finding 2).
- RunPod volume retention: keep until step 5 passes on GCP, then release (weights
  re-download freely; everything else is in git).
