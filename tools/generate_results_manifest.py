#!/usr/bin/env python
"""Generate the local results manifest (DGX completion gate item 2).

Hashes every file under results/ and artifacts/ so transferred or regenerated outputs can
be verified against the tagged state. results/ itself is gitignored; the manifest
(docs/results_manifest.json) is committed and lists sha256 + size for each output, plus
the producing repo revision.

Usage:
    python tools/generate_results_manifest.py [--out docs/results_manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("results", "artifacts")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/results_manifest.json")
    args = parser.parse_args()

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    files = []
    for base in SCAN_DIRS:
        root = REPO_ROOT / base
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                files.append({
                    "path": str(path.relative_to(REPO_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })

    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_head": head,
        "note": ("results/ is gitignored; regenerate outputs with the tools listed in "
                 "docs/WORKLOG.md and verify against these hashes. artifacts/ is tracked."),
        "file_count": len(files),
        "files": files,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(files)} files hashed -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
