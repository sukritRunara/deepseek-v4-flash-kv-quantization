#!/usr/bin/env python
"""Stage-B QDQ simulation runner: baseline vs named precision policies, teacher-forced.

Runs the tiny random DeepSeek-V4 model (all three attention layer types) once per policy
with IDENTICAL token histories and reports logit/NLL/top-k metrics against the BF16
baseline. Everything here is SIMULATION: values are stored in BF16/FP32 after
quantize-dequantize — there is NO memory saving to report, by design (Stage B).

Random-weight caveat: absolute numbers do not transfer to the real checkpoint (the real
model was QAT-trained for exactly the official policy; a random model was not). This tool
validates the machinery and gives relative orderings only. Real-model numbers come from
the RunPod phase.

Usage:
    python tools/run_qdq_simulation.py [--policies name1,name2,...] [--seq-len 48]
        [--prefill 24] [--batch 2] [--seed 0] [--device cpu] [--index-topk N]
        [--json-out results/qdq_simulation.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.harness import run_teacher_forced  # noqa: E402
from v4_kv_quant.metrics import (  # noqa: E402
    indexer_topk_overlap,
    logit_comparison_metrics,
    next_token_nll,
)
from v4_kv_quant.policy import NAMED_POLICIES  # noqa: E402
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids  # noqa: E402

BANNER = "Stage B QDQ SIMULATION - BF16/FP32 storage after quantize-dequantize; NO memory savings"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", default=",".join(NAMED_POLICIES))
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--prefill", type=int, default=24)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index-topk", type=int, default=None,
                        help="override tiny-model index_topk (default: production-like selective 2)")
    parser.add_argument("--json-out", default="results/qdq_simulation.json")
    args = parser.parse_args()

    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    unknown = [n for n in names if n not in NAMED_POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}; available: {list(NAMED_POLICIES)}")

    overrides = {} if args.index_topk is None else {"index_topk": args.index_topk}
    model = build_tiny_model(seed=args.seed, device=args.device, **overrides)
    input_ids = deterministic_input_ids(args.batch, args.seq_len).to(args.device)

    print("=" * 100)
    print(BANNER)
    print("=" * 100)
    print(f"tiny model: layer_types={model.config.layer_types} window={model.config.sliding_window} "
          f"rates={model.config.compress_rates} index_topk={model.config.index_topk}")
    print(f"batch={args.batch} seq_len={args.seq_len} prefill={args.prefill} seed={args.seed} "
          f"device={args.device} dtype={model.dtype}")

    baseline = run_teacher_forced(model, input_ids, args.prefill, policy=None)
    nll_baseline = next_token_nll(baseline.logits, input_ids)
    print(f"\nbaseline (stock DynamicCache): next-token NLL = {nll_baseline:.6f}")

    header = (f"{'policy':<28} {'max|dlogit|':>12} {'mean|d|':>10} {'RMS':>10} {'KL mean':>10} "
              f"{'top1 agree':>10} {'dNLL':>10} {'idx overlap':>11}")
    print("\n" + header)
    print("-" * len(header))

    results = []
    for name in names:
        policy = NAMED_POLICIES[name]()
        run = run_teacher_forced(model, input_ids, args.prefill, policy=policy)
        metrics = logit_comparison_metrics(baseline.logits, run.logits)
        nll = next_token_nll(run.logits, input_ids)
        overlap = indexer_topk_overlap(baseline.indexer_picks, run.indexer_picks)
        metrics |= {
            "nll": nll,
            "nll_delta": nll - nll_baseline,
            "indexer_topk_overlap": overlap,
            "policy": policy.to_dict(),
        }
        results.append({"name": name, **metrics})
        print(f"{name:<28} {metrics['max_abs_logit_err']:>12.4e} {metrics['mean_abs_logit_err']:>10.3e} "
              f"{metrics['rms_logit_err']:>10.3e} {metrics['kl_mean']:>10.3e} "
              f"{metrics['top1_agreement']:>10.4f} {metrics['nll_delta']:>+10.4e} "
              f"{overlap['mean_overlap']:>11.4f}")
        if metrics["nan_count"] or metrics["inf_count"]:
            print(f"  !! NaN={metrics['nan_count']} Inf={metrics['inf_count']}")

    report = {
        "label": BANNER,
        "caveat": "tiny RANDOM-weight model: machinery validation and relative ordering only; "
                  "absolute quality numbers do not transfer to the QAT-trained checkpoint",
        "config": {
            "seed": args.seed, "batch": args.batch, "seq_len": args.seq_len,
            "prefill": args.prefill, "device": args.device,
            "index_topk": model.config.index_topk,
            "layer_types": model.config.layer_types,
            "compress_rates": model.config.compress_rates,
        },
        "baseline_nll": nll_baseline,
        "policies": results,
    }
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nJSON written to {json_path}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
