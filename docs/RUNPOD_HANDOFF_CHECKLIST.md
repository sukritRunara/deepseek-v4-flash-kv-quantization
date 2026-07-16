# RunPod Handoff Checklist

The intended transition is one-way: GX10 development first, then RunPod becomes the authoritative execution environment for the full model.

## Preflight on the GX10

- [ ] Repository is clean and tagged `dgx-phase-complete-v1`.
- [ ] `docs/REPRODUCIBILITY.md` contains all source revisions.
- [ ] Tiny-model test suite passes from a fresh checkout.
- [ ] Source-only smoke commands are documented.
- [ ] Full-model commands accept paths/configuration rather than hardcoded local values.
- [ ] Calibration inputs or manifests are committed or stored in a reproducible location.
- [ ] No virtual environment, Triton cache, compiled `.so`, or architecture-specific build artifact is included.
- [ ] No full model weights are included in Git.

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

- [ ] Launch one RTX PRO 6000 Blackwell 96 GB pod.
- [ ] Use the intended persistent/network volume.
- [ ] Clone the tagged repository.
- [ ] Rebuild the environment for x86-64 and SM120.
- [ ] Capture environment metadata.
- [ ] Run hardware smoke tests.
- [ ] Run the complete tiny-model test suite.
- [ ] Compile any Triton/CUDA extensions on the target.
- [ ] Run tiny QDQ and storage smoke experiments.
- [ ] Record all incompatibilities and fixes in Git.
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
