"""Source-only landing checks for a new execution environment (GX10 or RunPod pod).

Driven entirely by an expectations file (configs/expectations_*.json):

    {
      "machine": "x86_64",
      "require_cuda": true,
      "gpu_name_contains": "RTX PRO 6000",
      "min_compute_capability": [12, 0],
      "source_pins": "configs/source_pins.json",
      "max_vendor_model_mb": 100
    }

Every check returns PASS/FAIL/WARN with detail; nothing is silently skipped.
The same checks pass on the GX10 with its own expectations file, satisfying the
"same command structure works locally" Phase-6 exit condition.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch


def _check(name: str, ok: bool, detail: str, warn_only: bool = False) -> dict:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    return {"check": name, "status": status, "detail": detail}


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def run_landing_checks(expect: dict, repo_root: Path, tiny_gate: bool = True) -> list[dict]:
    checks: list[dict] = []

    machine = platform.machine()
    checks.append(_check("platform_machine", machine == expect["machine"],
                         f"expected {expect['machine']}, got {machine}"))

    cuda_ok = torch.cuda.is_available()
    if expect.get("require_cuda", True):
        checks.append(_check("cuda_available", cuda_ok, f"torch.cuda.is_available()={cuda_ok}"))
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        wanted = expect.get("gpu_name_contains", "")
        checks.append(_check("gpu_name", wanted.lower() in props.name.lower(),
                             f"expected name containing {wanted!r}, got {props.name!r}"))
        min_cc = tuple(expect.get("min_compute_capability", [0, 0]))
        checks.append(_check("compute_capability", (props.major, props.minor) >= min_cc,
                             f"expected >= {min_cc}, got ({props.major}, {props.minor})"))

    for dtype_name in ("float8_e4m3fn", "float8_e8m0fnu", "float4_e2m1fn_x2"):
        checks.append(_check(f"dtype_{dtype_name}", hasattr(torch, dtype_name), "torch attribute presence"))
    try:
        x = torch.linspace(-440, 440, 448)
        err = (x - x.to(torch.float8_e4m3fn).float()).abs().max().item()
        checks.append(_check("fp8_roundtrip", err < 32, f"max cast error {err:.3f}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("fp8_roundtrip", False, f"{type(exc).__name__}: {exc}"))

    pins_path = repo_root / expect["source_pins"]
    if pins_path.exists():
        pins = json.loads(pins_path.read_text())
        for name, subdir, sha_key in (
            ("vendor_model", "vendor/DeepSeek-V4-Flash", "model_sha"),
            ("vendor_transformers", "vendor/transformers", "transformers_sha"),
        ):
            head = _git_head(repo_root / subdir)
            checks.append(_check(f"{name}_pinned", head == pins[sha_key],
                                 f"expected {pins[sha_key][:12]}, got {str(head)[:12]}"))
    else:
        checks.append(_check("source_pins_file", False, f"missing {pins_path}"))

    model_dir = repo_root / "vendor/DeepSeek-V4-Flash"
    if model_dir.exists():
        size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 2**20
        limit = expect.get("max_vendor_model_mb", 100)
        checks.append(_check("no_weights_materialized", size_mb < limit,
                             f"vendor model tree {size_mb:.0f} MiB (limit {limit} MiB; LFS pointers only)"))

    python_h = Path(f"/usr/include/python{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}/Python.h")
    checks.append(_check("python_dev_headers", python_h.exists(),
                         f"{python_h} (triton/torch.compile AND torch._native CUDA bmm need it)",
                         warn_only=not expect.get("require_python_dev", False)))

    if tiny_gate:
        try:
            from .harness import run_teacher_forced
            from .policy import reference_official_qdq
            from .tiny_model import build_tiny_model, deterministic_input_ids

            # deterministic portable gate on CPU
            model = build_tiny_model(seed=0, device="cpu")
            ids = deterministic_input_ids(1, 15)
            simulated = run_teacher_forced(model, ids, prefill_len=9, policy=reference_official_qdq())
            actual = run_teacher_forced(model, ids, prefill_len=9, policy=reference_official_qdq(), storage=True)
            checks.append(_check("tiny_forward", bool(torch.isfinite(simulated.logits).all()),
                                 "tiny model forward on cpu"))
            checks.append(_check("stage_c_bitwise_gate", bool(torch.equal(actual.logits, simulated.logits)),
                                 "storage cache == QDQ cache on cpu"))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("tiny_gate", False, f"{type(exc).__name__}: {exc}"))

        if cuda_ok:
            # CUDA model execution: on this torch build, torch.bmm dispatches to a
            # Triton-backed kernel (torch._native bmm_outer_product), so a missing
            # python3.X-dev makes ALL CUDA forwards fail. Expectation-driven severity:
            # a RunPod pod must FAIL here, the GX10 records a WARN with the root cause.
            try:
                from .tiny_model import build_tiny_model, deterministic_input_ids

                cuda_model = build_tiny_model(seed=0, device="cuda")
                with torch.no_grad():
                    out = cuda_model(deterministic_input_ids(1, 12).to("cuda"), use_cache=True)
                checks.append(_check("cuda_model_forward", bool(torch.isfinite(out.logits).all()),
                                     "tiny model forward on cuda"))
            except Exception as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"
                if "Python.h" in detail or "cuda_utils.c" in detail:
                    detail = ("CUDA forward blocked: python dev headers missing "
                              "(torch._native Triton bmm compiles against Python.h; "
                              "remedy: install python3.X-dev) - " + detail[:160])
                checks.append(_check("cuda_model_forward", False, detail[:400],
                                     warn_only=not expect.get("require_cuda_model_forward", True)))

    return checks


def checks_passed(checks: list[dict]) -> bool:
    return all(c["status"] != "FAIL" for c in checks)
