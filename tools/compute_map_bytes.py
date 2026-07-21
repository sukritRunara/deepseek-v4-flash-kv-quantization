#!/usr/bin/env python
"""Analytic KV-cache bytes for precision maps/policies (D-016 deliverable).

Computes exact storage-layout bytes (codes + scales, K=V counted once, RoPE BF16)
per variant at several context lengths, from the model config and map/policy
definitions — no GPU needed. Validated against the MEASURED B7 benchmark numbers
for the baseline and official-policy storage variants (same accounting rules as
`v4_kv_quant.memory`); compressor working buffers are excluded (small, identical
across variants), which is the expected source of the ~2% validation gap.

Byte layout mirrors src/v4_kv_quant/storage.py exactly:
  FP8 e4m3: 1 B/value + 1 B e8m0 scale per group of 64
  FP4 e2m1: 0.5 B/value (packed) + 1 B e8m0 scale per group of 32
  BF16:     2 B/value
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NOPE, ROPE, IDX_DIM = 448, 64, 128
# measured anchors: results/benchmark_runpod_20260720T083335Z.json (B7, cache KiB @8k)
MEASURED_8K_KIB = {"baseline": 61930.0, "official_storage": 31933.5}


def entry_bytes_main(kinds_by_range: list[tuple[int, int, str]]) -> float:
    """Bytes for one main-KV entry given [(start, end, kind)] nope coverage."""
    total = ROPE * 2.0  # RoPE channels always BF16
    covered = 0
    for start, end, kind in kinds_by_range:
        width = end - start
        covered += width
        if kind == "fp8_e4m3":
            total += width + width / 64
        elif kind == "fp4_e2m1":
            total += width / 2 + width / 32
        else:
            total += width * 2.0
    total += (NOPE - covered) * 2.0  # uncovered nope stays BF16
    return total


def entry_bytes_indexer(kind: str | None) -> float:
    if kind in ("fp4_e2m1_hadamard", "fp4_e2m1"):
        return IDX_DIM / 2 + IDX_DIM / 32
    return IDX_DIM * 2.0


def variant_bytes(cfg: dict, seq_len: int, main_kinds, indexer_kind) -> float:
    """main_kinds(layer, state) -> [(start,end,kind)]; indexer_kind(layer) -> str|None."""
    ratios = cfg["compress_ratios"][: cfg["num_hidden_layers"]]
    window = min(seq_len, cfg["sliding_window"])
    total = 0.0
    for layer, ratio in enumerate(ratios):
        total += window * entry_bytes_main(main_kinds(layer, "window_kv"))
        if ratio > 0:
            n_comp = seq_len // ratio
            total += n_comp * entry_bytes_main(main_kinds(layer, "compressed_kv"))
            if ratio == 4:  # CSA layer: indexer keys, one per compressed entry
                total += n_comp * entry_bytes_indexer(indexer_kind(layer))
    return total


def from_map(map_path: Path):
    entries = json.loads(map_path.read_text())["entries"]
    main: dict = {}
    idx: dict = {}
    for e in entries:
        if e["state"] == "indexer_kv":
            idx[e["layer_idx"]] = e["kind"]
        else:
            main.setdefault((e["layer_idx"], e["state"]), []).append(
                (e["start"], e["end"], e["kind"]))
    return (lambda l, s: main.get((l, s), [])), (lambda l: idx.get(l))


def uniform(main_kind: str | None, indexer_kind: str | None):
    ranges = [(0, NOPE, main_kind)] if main_kind else []
    return (lambda l, s: ranges), (lambda l: indexer_kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="/home/sukrit/models/DeepSeek-V4-Flash/config.json")
    ap.add_argument("--maps-dir", default="results/calibration_full")
    ap.add_argument("--out", default="results/map_bytes.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    maps_dir = Path(args.maps_dir)

    variants = {
        "baseline_bf16": uniform(None, None),
        "official_storage": uniform("fp8_e4m3", "fp4_e2m1_hadamard"),
        "all_fp4": uniform("fp4_e2m1", None),
    }
    for name, path in (("ratified_map", maps_dir / "precision_map.json"),
                       ("ladder20", maps_dir / "precision_map_ladder20.json")):
        if path.exists():
            main_fn, idx_fn = from_map(path)
            variants[name] = (main_fn, idx_fn)
            # hypothetical: same main-KV map but official FP4 indexer — quantifies
            # exactly what the D-015 BF16-indexer decision costs in bytes
            variants[name + "+fp4idx"] = (main_fn, lambda l: "fp4_e2m1_hadamard")

    lens = (8192, 32768, 65536)
    report: dict = {"config": args.config, "lengths": lens, "variants": {}}
    base = {L: variant_bytes(cfg, L, *variants["baseline_bf16"]) for L in lens}
    print(f"{'variant':<18}" + "".join(f"{L:>14}" for L in lens) + f"{'ratio@65k':>11}")
    for name, fns in variants.items():
        row = {L: variant_bytes(cfg, L, *fns) for L in lens}
        report["variants"][name] = {
            str(L): {"bytes": row[L], "ratio_vs_bf16": row[L] / base[L]} for L in lens}
        cells = "".join(f"{row[L] / 2**20:>11.1f} MiB" for L in lens)
        print(f"{name:<18}{cells}{row[65536] / base[65536]:>11.3f}")

    print("\nvalidation vs MEASURED B7 @8k (KiB):")
    for name, measured in MEASURED_8K_KIB.items():
        key = "baseline_bf16" if name == "baseline" else name
        analytic = report["variants"][key]["8192"]["bytes"] / 1024
        delta = analytic / measured - 1
        report.setdefault("validation_8k", {})[name] = {
            "analytic_kib": analytic, "measured_kib": measured, "rel_error": delta}
        print(f"  {name:<18} analytic {analytic:>10.1f}  measured {measured:>10.1f}  "
              f"delta {delta:+.2%}")

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
