#!/usr/bin/env python
"""Source-only landing test for a new execution environment (RunPod pod or the GX10).

Runs the checks in src/v4_kv_quant/landing.py against an expectations file, optionally
followed by the full tiny-model pytest suite. No weights, no network required after the
source checkout. Exit code 0 iff no check FAILs (WARNs allowed).

Usage:
    python tools/runpod_landing_test.py --expect configs/expectations_gx10.json
    python tools/runpod_landing_test.py --expect configs/expectations_runpod.json --run-suite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from v4_kv_quant.landing import checks_passed, run_landing_checks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect", default="configs/expectations_runpod.json")
    parser.add_argument("--run-suite", action="store_true", help="also run the full pytest suite")
    parser.add_argument("--json-out", default="results/landing_test.json")
    args = parser.parse_args()

    expect = json.loads((REPO_ROOT / args.expect).read_text())
    print(f"landing checks against {args.expect}")
    checks = run_landing_checks(expect, REPO_ROOT)
    for c in checks:
        print(f"  [{c['status']:^4}] {c['check']}: {c['detail']}")

    suite = None
    if args.run_suite:
        print("\nrunning tiny-model test suite ...")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"], cwd=REPO_ROOT, capture_output=True, text=True,
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()[-200:]
        suite = {"returncode": proc.returncode, "summary": tail}
        print(f"  suite: rc={proc.returncode} ({tail})")
        checks.append({"check": "pytest_suite", "status": "PASS" if proc.returncode == 0 else "FAIL",
                       "detail": tail})

    ok = checks_passed(checks)
    json_path = REPO_ROOT / args.json_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"expectations": args.expect, "checks": checks, "suite": suite,
                                     "passed": ok}, indent=2) + "\n")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} - report at {json_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
