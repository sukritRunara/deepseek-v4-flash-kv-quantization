#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/artifacts/env"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TXT="$OUT_DIR/environment-$STAMP.txt"
JSON="$OUT_DIR/python-environment-$STAMP.json"

{
  echo "timestamp_utc=$STAMP"
  echo "hostname=$(hostname)"
  echo "pwd=$ROOT"
  echo
  echo "== uname =="
  uname -a || true
  echo
  echo "== os-release =="
  cat /etc/os-release || true
  echo
  echo "== architecture =="
  uname -m || true
  lscpu || true
  echo
  echo "== memory =="
  free -h || true
  echo
  echo "== storage =="
  df -h "$ROOT" || true
  echo
  echo "== NVIDIA =="
  nvidia-smi || true
  echo
  echo "== NVIDIA query =="
  nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu,power.draw --format=csv,noheader || true
  echo
  echo "== CUDA toolkit =="
  command -v nvcc || true
  nvcc --version || true
  echo
  echo "== compilers =="
  gcc --version | head -1 || true
  g++ --version | head -1 || true
  cmake --version | head -1 || true
  ninja --version || true
  echo
  echo "== Git =="
  git --version || true
  git lfs version || true
  echo
  echo "== Python executables =="
  command -v python || true
  python --version || true
  command -v python3 || true
  python3 --version || true
} | tee "$TXT"

python3 - "$JSON" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
report = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
}

try:
    import torch
    report["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "float8_e4m3fn_available": hasattr(torch, "float8_e4m3fn"),
        "float4_e2m1fn_x2_available": hasattr(torch, "float4_e2m1fn_x2"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["torch"]["device"] = {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        }
except Exception as exc:
    report["torch_error"] = repr(exc)

for module in ("transformers", "triton", "numpy", "safetensors", "accelerate"):
    try:
        mod = __import__(module)
        report[module] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        report[f"{module}_error"] = repr(exc)

out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(f"Wrote {out}")
PY

echo
printf 'Environment reports written to:\n  %s\n  %s\n' "$TXT" "$JSON"
