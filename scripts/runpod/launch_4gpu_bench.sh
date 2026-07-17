#!/usr/bin/env bash
# Full-model benchmark launch on the four-GPU RunPod pod.
#
# Single-process multi-GPU via transformers device_map=auto (the configuration upstream's
# own V4 integration tests use). All paths/context lengths/batch sizes come from
# configs/bench_runpod_4gpu.json — edit the config, not this script.
#
# For expert-parallel generation experiments (torchrun EP), see the distributed worker
# pattern in vendor/transformers/tests/models/deepseek_v4/test_modeling_deepseek_v4.py —
# benchmark comparisons should still be produced by tools/benchmark_cache.py so every
# variant shares one measurement path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "REFUSING: full-model benchmarks run on the RunPod pod, not the GX10 (got $(uname -m))." >&2
  exit 1
fi

: "${VENV:=.venv}"
: "${BENCH_CONFIG:=configs/bench_runpod_4gpu.json}"
: "${EXPECTED_GPUS:=4}"

gpus=$(nvidia-smi --list-gpus | wc -l)
if (( gpus < EXPECTED_GPUS )); then
  echo "REFUSING: expected ${EXPECTED_GPUS} GPUs, found ${gpus}." >&2
  exit 1
fi

"$VENV/bin/python" tools/runpod_landing_test.py --expect configs/expectations_runpod.json
"$VENV/bin/python" tools/benchmark_cache.py --config "$BENCH_CONFIG" \
  --json-out "results/benchmark_runpod_$(date -u +%Y%m%dT%H%M%SZ).json"
