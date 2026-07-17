#!/usr/bin/env bash
# RunPod environment bootstrap (x86-64, SM120). Source-only: never downloads weights.
# Rebuilds everything from the pinned revisions in configs/source_pins.json — never copy
# the GX10's ARM64 venv or compiled artifacts (CLAUDE.md constraint 5).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "REFUSING: this bootstrap targets x86-64 RunPod pods (got $(uname -m))." >&2
  echo "On the GX10, the environment already exists; see docs/REPRODUCIBILITY.md." >&2
  exit 1
fi

# Overridables (keep hardware/path assumptions out of the script body)
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu130}"
: "${VENV:=.venv}"

PINS="configs/source_pins.json"
model_repo=$(python3 -c "import json;print(json.load(open('$PINS'))['model_repo'])")
model_sha=$(python3 -c "import json;print(json.load(open('$PINS'))['model_sha'])")
tf_repo=$(python3 -c "import json;print(json.load(open('$PINS'))['transformers_repo'])")
tf_sha=$(python3 -c "import json;print(json.load(open('$PINS'))['transformers_sha'])")

export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -d vendor/DeepSeek-V4-Flash ]]; then
  git clone "$model_repo" vendor/DeepSeek-V4-Flash
fi
git -C vendor/DeepSeek-V4-Flash checkout "$model_sha"
if [[ ! -d vendor/transformers ]]; then
  git clone "$tf_repo" vendor/transformers
fi
git -C vendor/transformers fetch --depth 1 origin "$tf_sha" || true
git -C vendor/transformers checkout "$tf_sha"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install torch --index-url "$TORCH_INDEX_URL"
"$VENV/bin/pip" install numpy safetensors pytest accelerate sentencepiece
# kernels: required at first forward of the native-FP8 checkpoint (finegrained-fp8 hub
# kernel); version pin from transformers' own compatibility check (WORKLOG 2026-07-17 B1).
"$VENV/bin/pip" install "kernels==0.15.2"
# datasets: calibration corpus streaming (D-012)
"$VENV/bin/pip" install datasets
"$VENV/bin/pip" install -e vendor/transformers
"$VENV/bin/pip" install -e .

bash scripts/capture_environment.sh || true
"$VENV/bin/python" tools/hardware_smoke.py
"$VENV/bin/python" tools/runpod_landing_test.py --expect configs/expectations_runpod.json --run-suite

echo
echo "Environment ready. Next (four-GPU pod only):"
echo "  RUNPOD_ALLOW_WEIGHTS=1 bash scripts/runpod/download_model.sh"
echo "  bash scripts/runpod/launch_4gpu_bench.sh"
