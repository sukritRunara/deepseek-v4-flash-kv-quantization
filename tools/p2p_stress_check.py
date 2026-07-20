#!/usr/bin/env python
"""Pod-health check: stress-test inter-GPU copies for silent corruption.

Validates every ordered GPU pair with large direct copies, plain and under concurrent
compute (the regime that exposed the Phase-B pod's faulty PCIe P2P — see D-011 and
``v4_kv_quant.p2p_workaround``). Exit 0 = all transfers bit-exact; exit 1 = corruption.

Usage:
    python tools/p2p_stress_check.py [--iters 20] [--mib 64] [--workaround]

``--workaround`` applies ``ensure_host_staged_p2p()`` first, to verify the mitigation
on a faulty pod (should then pass).
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=20, help="iterations per pair per phase")
    ap.add_argument("--mib", type=int, default=64, help="transfer size in MiB")
    ap.add_argument("--workaround", action="store_true",
                    help="disable peer access first (validate the mitigation)")
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n < 2:
        print("fewer than 2 CUDA devices; nothing to check")
        return 0
    if args.workaround:
        import os

        from v4_kv_quant.p2p_workaround import FORCE_ENV_VAR, ensure_host_staged_p2p

        os.environ[FORCE_ENV_VAR] = "1"  # --workaround always arms it (D-014 opt-in gate)
        ensure_host_staged_p2p()

    torch.manual_seed(0)
    numel = args.mib * (1 << 20) // 4
    total_bad = 0

    for phase in ("plain", "concurrent-compute"):
        print(f"== {phase}: {args.mib} MiB x {args.iters} iters per ordered pair ==", flush=True)
        mats = {i: torch.randn(4096, 4096, device=f"cuda:{i}") for i in range(n)}
        for i in range(n):
            src = torch.randn(numel, device=f"cuda:{i}")
            ref = src.cpu()
            for j in range(n):
                if i == j:
                    continue
                bad = 0
                for _ in range(args.iters):
                    if phase == "concurrent-compute":
                        _ = mats[i] @ mats[i]
                        _ = mats[j] @ mats[j]
                    dst = src.to(f"cuda:{j}")
                    got = dst.cpu()
                    torch.cuda.synchronize()
                    if not torch.equal(got, ref):
                        bad += 1
                total_bad += bad
                print(f"  gpu{i}->gpu{j}: {'OK' if bad == 0 else f'CORRUPT {bad}/{args.iters}'}",
                      flush=True)

    print(f"total corrupt transfers: {total_bad}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
