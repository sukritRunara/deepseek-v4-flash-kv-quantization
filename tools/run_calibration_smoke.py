#!/usr/bin/env python
"""End-to-end calibration smoke run on the tiny random DeepSeek-V4 model.

Pipeline (DGX plan Phase 4, Stage-B simulation only — NO memory savings):
  1. generate disjoint CALIBRATION and HELD-OUT token sets (separate seeds, both saved);
  2. activation-statistics pass (amax / mean|x| / RMS / per-group amax / QDQ error RMS);
  3. one-target empirical perturbation sweep over all enumerated targets (teacher-forced
     vs BF16 baseline; ΔNLL + KL for main KV, + top-k overlap for indexer targets);
  4. rank targets, build a versioned precision map from explicit fractions/thresholds;
  5. evaluate the built map on the HELD-OUT set vs baseline.

Outputs under --out-dir (default results/calibration_smoke/):
  token_ids.json, activation_stats.json, sensitivity.json, precision_map.json,
  heldout_eval.json, and a combined report.json.

Random-weight caveat: rankings validate the machinery only; real sensitivity comes from
the QAT-trained checkpoint on RunPod.

Usage:
    python tools/run_calibration_smoke.py [--seq-len 48] [--prefill 24] [--batch 2]
        [--model-seed 0] [--calib-seed 11] [--heldout-seed 12] [--group-size-main 8]
        [--fp8-fraction 0.75] [--fp4-fraction 0.0] [--indexer-min-overlap 0.9]
        [--device cpu] [--out-dir results/calibration_smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.harness import run_teacher_forced  # noqa: E402
from v4_kv_quant.metrics import logit_comparison_metrics, next_token_nll  # noqa: E402
from v4_kv_quant.sensitivity import build_map_from_sweep, run_sensitivity_sweep  # noqa: E402
from v4_kv_quant.stats import StatsCollectorCache, StatsRecorder  # noqa: E402
from v4_kv_quant.targets import enumerate_targets  # noqa: E402
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids  # noqa: E402

BANNER = "Stage B CALIBRATION SMOKE - QDQ simulation only; NO memory savings; random-weight machinery validation"


@torch.no_grad()
def stats_pass(model, input_ids: torch.Tensor, prefill_len: int, group_size_main: int) -> StatsRecorder:
    recorder = StatsRecorder(group_size_main=group_size_main)
    cache = StatsCollectorCache(model.config, recorder)
    model(input_ids[:, :prefill_len], past_key_values=cache, use_cache=True)
    for t in range(prefill_len, input_ids.shape[1]):
        model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    return recorder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--prefill", type=int, default=24)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=11)
    parser.add_argument("--heldout-seed", type=int, default=12)
    parser.add_argument("--group-size-main", type=int, default=8,
                        help="contiguous main-KV group width (tiny default 8; production 64)")
    parser.add_argument("--fp8-fraction", type=float, default=0.75)
    parser.add_argument("--fp4-fraction", type=float, default=0.0)
    parser.add_argument("--indexer-min-overlap", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="results/calibration_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 100)
    print(BANNER)
    print("=" * 100)

    model = build_tiny_model(seed=args.model_seed, device=args.device)
    config = model.config
    calib_ids = deterministic_input_ids(args.batch, args.seq_len, seed=args.calib_seed).to(args.device)
    heldout_ids = deterministic_input_ids(args.batch, args.seq_len, seed=args.heldout_seed).to(args.device)
    assert not torch.equal(calib_ids, heldout_ids), "calibration and held-out sets must differ"
    (out_dir / "token_ids.json").write_text(json.dumps({
        "calibration": {"seed": args.calib_seed, "ids": calib_ids.tolist()},
        "heldout": {"seed": args.heldout_seed, "ids": heldout_ids.tolist()},
    }, indent=2) + "\n")

    print(f"model seed={args.model_seed} device={args.device} | layer_types={config.layer_types}")
    print(f"calib seed={args.calib_seed} heldout seed={args.heldout_seed} "
          f"batch={args.batch} seq_len={args.seq_len} prefill={args.prefill}")

    # 1. activation statistics
    print("\n[1/4] activation statistics pass (pass-through, bit-exact)")
    recorder = stats_pass(model, calib_ids, args.prefill, args.group_size_main)
    stats = recorder.summary()
    (out_dir / "activation_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    for key, s in stats.items():
        print(f"  {key:<24} amax={s['amax']:.4f} mean|x|={s['mean_abs']:.4f} "
              f"rms={s['rms']:.4f} rms_qdq_err={s['rms_qdq_err']:.2e} groups={len(s['per_group_amax'])}")

    # 2. sensitivity sweep
    targets = enumerate_targets(config, group_size_main=args.group_size_main)
    print(f"\n[2/4] one-target perturbation sweep: {len(targets)} targets on calibration set")
    _, baseline_nll, records = run_sensitivity_sweep(
        model, calib_ids, args.prefill, targets, progress=True
    )
    (out_dir / "sensitivity.json").write_text(json.dumps({
        "baseline_nll": baseline_nll,
        "records": [r.as_dict() for r in records],
    }, indent=2) + "\n")
    ranked = sorted(records, key=lambda r: r.score, reverse=True)
    print("  most sensitive:")
    for r in ranked[:3]:
        print(f"    {r.target.key:<28} score={r.score:.3e}")
    print("  least sensitive:")
    for r in ranked[-3:]:
        print(f"    {r.target.key:<28} score={r.score:.3e}")

    # 3. build + save precision map
    print(f"\n[3/4] building precision map (fp8_fraction={args.fp8_fraction}, "
          f"fp4_fraction={args.fp4_fraction}, indexer_min_overlap={args.indexer_min_overlap})")
    precision_map = build_map_from_sweep(
        records,
        name=f"tiny-smoke-{stamp}",
        fp8_fraction=args.fp8_fraction,
        fp4_fraction=args.fp4_fraction,
        indexer_min_overlap=args.indexer_min_overlap,
        provenance={
            "created_utc": stamp,
            "model_seed": args.model_seed,
            "calib_seed": args.calib_seed,
            "heldout_seed": args.heldout_seed,
            "token_ids_file": "token_ids.json",
            "group_size_main": args.group_size_main,
            "batch": args.batch, "seq_len": args.seq_len, "prefill": args.prefill,
            "torch": torch.__version__,
            "note": "random-weight tiny model; machinery validation only",
        },
    )
    precision_map.validate(config)
    map_path = out_dir / "precision_map.json"
    precision_map.to_json(map_path)
    by_kind: dict[str, int] = {}
    for e in precision_map.entries:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    n_bf16 = len(targets) - len(precision_map.entries)
    print(f"  map composition: {by_kind} + {n_bf16} targets left BF16 -> {map_path}")

    # 4. held-out evaluation
    print("\n[4/4] held-out evaluation of the built map")
    baseline_held = run_teacher_forced(model, heldout_ids, args.prefill)
    quant_held = run_teacher_forced(model, heldout_ids, args.prefill, precision_map=precision_map)
    held_metrics = logit_comparison_metrics(baseline_held.logits, quant_held.logits)
    held_metrics["nll_baseline"] = next_token_nll(baseline_held.logits, heldout_ids)
    held_metrics["nll_quantized"] = next_token_nll(quant_held.logits, heldout_ids)
    held_metrics["nll_delta"] = held_metrics["nll_quantized"] - held_metrics["nll_baseline"]
    (out_dir / "heldout_eval.json").write_text(json.dumps(held_metrics, indent=2) + "\n")
    print(f"  held-out: max|dlogit|={held_metrics['max_abs_logit_err']:.3e} "
          f"KL={held_metrics['kl_mean']:.3e} top1={held_metrics['top1_agreement']:.4f} "
          f"dNLL={held_metrics['nll_delta']:+.4e} NaN={held_metrics['nan_count']}")

    report = {
        "label": BANNER,
        "created_utc": stamp,
        "config": vars(args),
        "n_targets": len(targets),
        "baseline_nll_calib": baseline_nll,
        "map_file": str(map_path),
        "map_composition": by_kind,
        "heldout": held_metrics,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nall outputs in {out_dir}/")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
