#!/usr/bin/env python
"""Hardware / dtype smoke test for the KV-quantization project.

Reports and functionally exercises everything the quantization experiments
depend on. Every unsupported capability is reported explicitly as
UNSUPPORTED with the failing exception — never silently skipped.

Checks:
  * CPU architecture, Python / PyTorch / CUDA runtime / CUDA toolkit / driver /
    Triton / Transformers versions;
  * GPU name and compute capability;
  * BF16 tensor ops (CPU and GPU);
  * torch.float8_e4m3fn availability + cast round trip (CPU and GPU);
  * torch.float8_e8m0fnu (scale dtype used by the official ue8m0 policy);
  * FP4 dtypes (float4_e2m1fn_x2) + a small pack/unpack-style operation;
  * torch.compile of a trivial function (CPU inductor and CUDA/Triton);
  * a hand-written Triton kernel (if triton importable);
  * unified/total memory as observed through supported APIs.

Usage:
    python tools/hardware_smoke.py [--json-out results/hardware_smoke.json]

Exit code 0 if the *required* capabilities pass (BF16 + FP8 round trip on the
available device); 1 otherwise. Optional capabilities (FP4, Triton, compile)
only affect the report, matching the phased plan in docs/DGX_PHASE_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

RESULTS: list[dict[str, Any]] = []


def record(name: str, required: bool, fn: Callable[[], Any]) -> None:
    try:
        detail = fn()
        RESULTS.append({"check": name, "status": "PASS", "required": required, "detail": detail})
    except Exception as exc:  # noqa: BLE001 - the whole point is reporting any failure
        RESULTS.append(
            {
                "check": name,
                "status": "UNSUPPORTED",
                "required": required,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-out", default="results/hardware_smoke.json")
    args = parser.parse_args()

    import torch

    info: dict[str, Any] = {
        "cpu_architecture": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    nvcc = shutil.which("nvcc")
    if nvcc:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False).stdout
        info["cuda_toolkit"] = next((line.strip() for line in out.splitlines() if "release" in line), "unknown")
    else:
        info["cuda_toolkit"] = "nvcc not on PATH"
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        info["nvidia_smi"] = out
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton"] = f"unavailable ({type(exc).__name__}: {exc})"
    # Triton / inductor JIT compile launcher stubs against Python.h; without the
    # python3.X-dev system package every torch.compile / triton kernel fails at gcc.
    python_h = Path(f"/usr/include/python{sys.version_info.major}.{sys.version_info.minor}/Python.h")
    info["python_dev_headers"] = (
        str(python_h) if python_h.exists() else f"MISSING ({python_h}) - triton/torch.compile will fail at gcc"
    )
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        info["transformers"] = f"unavailable ({type(exc).__name__}: {exc})"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gib": round(props.total_memory / 2**30, 1),
            "multi_processor_count": props.multi_processor_count,
        }
        free, total = torch.cuda.mem_get_info()
        info["cuda_mem_get_info_gib"] = {"free": round(free / 2**30, 1), "total": round(total / 2**30, 1)}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    # ---- functional checks -------------------------------------------------
    for device in devices:
        def bf16_check(device=device):
            a = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
            b = (a @ a).float()
            assert torch.isfinite(b).all()
            return f"{device}: bf16 matmul finite, mean|x|={b.abs().mean().item():.3f}"

        record(f"bf16_matmul[{device}]", required=True, fn=bf16_check)

        def fp8_roundtrip(device=device):
            assert hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn missing"
            x = torch.linspace(-440, 440, 448, device=device, dtype=torch.float32)
            q = x.to(torch.float8_e4m3fn)
            back = q.to(torch.float32)
            err = (x - back).abs().max().item()
            assert err < 32, f"round-trip error too large: {err}"  # e4m3 step near 448 is 32
            return f"{device}: e4m3 cast round trip max|err|={err:.3f} (within format step)"

        record(f"fp8_e4m3_roundtrip[{device}]", required=True, fn=fp8_roundtrip)

        def fp8_e8m0_scale(device=device):
            assert hasattr(torch, "float8_e8m0fnu"), "torch.float8_e8m0fnu missing"
            scales = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device).to(torch.float8_e8m0fnu)
            back = scales.to(torch.float32)
            assert torch.equal(back, torch.tensor([0.5, 1.0, 2.0, 4.0], device=device))
            # e8m0 is power-of-two only: 3.0 must round to a neighbouring power of 2
            three = torch.tensor([3.0], device=device).to(torch.float8_e8m0fnu).to(torch.float32)
            assert three.item() in (2.0, 4.0), f"unexpected e8m0 rounding: {three.item()}"
            return f"{device}: e8m0 powers-of-two exact; 3.0 -> {three.item()}"

        record(f"fp8_e8m0_scale_dtype[{device}]", required=False, fn=fp8_e8m0_scale)

        def fp4_dtype(device=device):
            assert hasattr(torch, "float4_e2m1fn_x2"), "torch.float4_e2m1fn_x2 missing"
            # float4_e2m1fn_x2 packs two e2m1 values per byte; PyTorch exposes it as a
            # storage dtype. Verify construction + a byte-level view round trip.
            raw = torch.randint(0, 256, (32, 16), device=device, dtype=torch.uint8)
            packed = raw.view(torch.float4_e2m1fn_x2)
            assert packed.dtype == torch.float4_e2m1fn_x2
            assert packed.shape == raw.shape
            back = packed.view(torch.uint8)
            assert torch.equal(back, raw)
            return f"{device}: float4_e2m1fn_x2 storage dtype usable (uint8 view round trip)"

        record(f"fp4_e2m1_dtype[{device}]", required=False, fn=fp4_dtype)

    def compile_cpu():
        def f(x):
            return torch.nn.functional.silu(x) * 2.0

        compiled = torch.compile(f)
        x = torch.randn(128)
        assert torch.allclose(compiled(x), f(x), atol=1e-6)
        return "cpu inductor compile OK"

    record("torch_compile[cpu]", required=False, fn=compile_cpu)

    if torch.cuda.is_available():

        def compile_cuda():
            def f(x):
                return torch.nn.functional.silu(x) * 2.0

            compiled = torch.compile(f)
            x = torch.randn(1024, device="cuda")
            assert torch.allclose(compiled(x), f(x), atol=1e-6)
            return "cuda torch.compile (triton backend) OK"

        record("torch_compile[cuda]", required=False, fn=compile_cuda)

        def triton_kernel():
            import triton
            import triton.language as tl

            @triton.jit
            def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)

            x = torch.randn(1000, device="cuda")
            y = torch.empty_like(x)
            add_one[(triton.cdiv(1000, 256),)](x, y, 1000, BLOCK=256)
            torch.cuda.synchronize()
            assert torch.allclose(y, x + 1)
            return "hand-written triton kernel OK"

        record("triton_kernel[cuda]", required=False, fn=triton_kernel)

        def fp8_scaled_mm():
            # torch._scaled_mm is the canonical FP8 GEMM entry point; row-major x col-major
            a = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn).t()
            scale = torch.tensor(1.0, device="cuda")
            out = torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            assert out.shape == (64, 64) and torch.isfinite(out.float()).all()
            return "torch._scaled_mm fp8 GEMM OK"

        record("fp8_scaled_mm[cuda]", required=False, fn=fp8_scaled_mm)

    # ---- report -------------------------------------------------------------
    print("=" * 88)
    print("Hardware / dtype smoke report")
    print("=" * 88)
    for key, value in info.items():
        print(f"{key:>26}: {value}")
    print("-" * 88)
    required_failed = []
    for r in RESULTS:
        flag = "required" if r["required"] else "optional"
        print(f"[{r['status']:^11}] ({flag:^8}) {r['check']}: {r['detail']}")
        if r["status"] != "PASS" and r["required"]:
            required_failed.append(r["check"])
    print("-" * 88)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"info": info, "checks": RESULTS}, indent=2) + "\n")
    print(f"JSON report written to {json_path}")

    if required_failed:
        print(f"REQUIRED CAPABILITIES MISSING: {required_failed}")
        return 1
    print("All required capabilities available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
