#!/usr/bin/env bash
# Full-checkpoint download for the RunPod four-GPU pod. HEAVILY GUARDED:
#   * refuses on non-x86_64 hosts (the GX10 must never hold weights — CLAUDE.md constraint 3);
#   * requires explicit RUNPOD_ALLOW_WEIGHTS=1 opt-in;
#   * pins the revision from configs/source_pins.json;
#   * checks free disk before starting (~160 GB of shards; budget 500 GB total).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "REFUSING: weight download is only permitted on x86-64 RunPod pods (got $(uname -m))." >&2
  exit 1
fi
if [[ "${RUNPOD_ALLOW_WEIGHTS:-0}" != "1" ]]; then
  echo "REFUSING: set RUNPOD_ALLOW_WEIGHTS=1 explicitly to download ~160 GB of weights." >&2
  exit 1
fi

: "${MODEL_ID:=deepseek-ai/DeepSeek-V4-Flash}"
: "${MODEL_DIR:=/home/sukrit/models/DeepSeek-V4-Flash}"  # GCP G4 boot disk (D-013/D-014)
: "${MIN_FREE_GB:=200}"

revision=$(python3 -c "import json;print(json.load(open('configs/source_pins.json'))['model_sha'])")

mkdir -p "$MODEL_DIR"
free_gb=$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  echo "REFUSING: only ${free_gb} GB free at $MODEL_DIR (need >= ${MIN_FREE_GB} GB)." >&2
  exit 1
fi

command -v hf >/dev/null || { echo "hf CLI missing: pip install -U huggingface_hub" >&2; exit 1; }
# hub 1.x dropped hf_transfer (WORKLOG B1); the default xet backend is used instead.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "downloading $MODEL_ID @ $revision -> $MODEL_DIR"
hf download "$MODEL_ID" --revision "$revision" --local-dir "$MODEL_DIR"

echo "done. Update configs/bench_runpod_4gpu.json model_path if you used a custom MODEL_DIR."
