# RunPod Handoff Checklist

The intended transition is one-way: GX10 development first, then RunPod becomes the authoritative execution environment for the full model.

## Preflight on the GX10 — completed 2026-07-17

- [x] Repository is clean and tagged `dgx-phase-complete-v1`.
- [x] `docs/REPRODUCIBILITY.md` contains all source revisions (machine-readable copy:
      `configs/source_pins.json`, cross-checked by `tests/test_benchmark.py`; GX10 package
      versions frozen in `docs/PIP_FREEZE_GX10.txt` — reference only, RunPod rebuilds).
- [x] Tiny-model test suite passes from a fresh checkout (90 passed; verified against a
      clean `git worktree` of the tagged commit — no dependence on uncommitted files).
- [x] Source-only smoke commands are documented (`docs/REPRODUCIBILITY.md`,
      `docs/WORKLOG.md`; single entry point: `tools/runpod_landing_test.py --run-suite`).
- [x] Full-model commands accept paths/configuration rather than hardcoded local values
      (`configs/bench_runpod_4gpu.json`, env-overridable `scripts/runpod/*.sh`).
- [x] Calibration inputs or manifests are committed or stored in a reproducible location
      (`artifacts/calibration_smoke/` incl. exact token ids + seeds; regeneration:
      `tools/run_calibration_smoke.py`; hashes in `docs/results_manifest.json`).
- [x] No virtual environment, Triton cache, compiled `.so`, or architecture-specific build
      artifact is included (`.gitignore` covers `.venv/`, caches, `*.so`; verified with
      `git ls-files`).
- [x] No full model weights are included in Git (`vendor/` untracked; model tree is 14 MB
      of LFS pointers; enforced by the landing test's `no_weights_materialized` check).

## Artifacts to transfer

Transfer:

```text
source code
Git history and tags
configuration files
small test fixtures
calibration/evaluation manifests
saved token IDs where licensing permits
documentation
container definition or environment specification
```

Do not transfer:

```text
GX10 Python virtual environment
ARM64 wheels or binaries
Triton cache
CUDA extensions compiled for SM121
host-specific absolute paths
temporary model caches
```

## Cheap one-GPU landing test

Before paying for four GPUs:

- [x] Launch one RTX PRO 6000 Blackwell 96 GB pod. (2026-07-17; CC 12.0, driver 580.126.09)
- [x] Use the intended persistent/network volume. (network volume on /workspace)
- [x] Clone the tagged repository. (checkout 4 doc-only commits after the tag)
- [x] Rebuild the environment for x86-64 and SM120. (`scripts/runpod/setup_env.sh`, default cu130)
- [x] Capture environment metadata. (`artifacts/env/*-20260717T075537Z.*`)
- [x] Run hardware smoke tests. (all required + optional PASS, incl. triton/compile on CUDA)
- [x] Run the complete tiny-model test suite. (90 passed; landing test ALL CHECKS PASSED)
- [x] Compile any Triton/CUDA extensions on the target. (triton kernels JIT-compile; python3.12-dev present)
- [x] Run tiny QDQ and storage smoke experiments. (`benchmark_cache.py --config configs/bench_tiny_local.json --device cuda`)
- [x] Record all incompatibilities and fixes in Git. (one: GX10-hardcoded host-identity tests; see WORKLOG 2026-07-17 Phase A)
- [ ] Freeze the resulting image/environment.
- [ ] Stop the one-GPU pod.

The full model is not expected to fit on this one-GPU landing pod. This step is solely for architecture and environment portability.

## Four-GPU pod

Provision:

```text
4 × RTX PRO 6000 Blackwell, 96 GB each
one node/pod
persistent storage sized for original weights, converted checkpoint, environments,
and experiment outputs
```

Suggested storage budget: at least 500 GB, with more if retaining both original and converted checkpoints.

## Full-model execution order

1. [ ] Download/pin the official model checkpoint on persistent storage.
2. [ ] Convert weights for the selected official reference runtime if required.
3. [ ] Establish untouched baseline generation.
4. [ ] Establish baseline four-GPU memory and timing.
5. [ ] Verify source-level cache instrumentation without quantization.
6. [ ] Collect real-model calibration statistics.
7. [ ] Run empirical group perturbations or ranked subsets.
8. [ ] Generate the candidate precision policy.
9. [ ] Evaluate QDQ quality on held-out data.
10. [ ] Implement/enable actual low-precision storage.
11. [ ] Re-run quality checks.
12. [ ] Run final memory and performance benchmarks.
13. [ ] Save results, environment, and exact command manifests.

## Final benchmark matrix

At minimum compare:

```text
untouched/reference baseline
official-style QDQ simulation
candidate mixed QDQ simulation
actual FP8 storage
actual mixed FP4/FP8 storage, if implemented
```

At several context lengths, record:

```text
quality metrics
actual cache bytes including scale overhead
peak allocated/reserved memory per GPU
prefill throughput
TTFT
decode throughput
inter-token latency
quantization/dequantization overhead
```

Use the same node, prompts, seeds, parallel configuration, batch sizes, and software stack for every comparison.
