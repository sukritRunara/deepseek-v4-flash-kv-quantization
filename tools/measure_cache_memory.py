#!/usr/bin/env python
"""Actual cache-byte comparison: stock BF16 vs Stage-B QDQ simulation vs Stage-C storage.

Runs the SAME token stream through three caches on the tiny random model and reports
per-layer and total bytes (logical and allocator-storage), demonstrating:
  * the Stage-B QDQ simulation saves exactly nothing (by design);
  * the Stage-C storage cache shows a real reduction, with scale tensors, RoPE slices,
    and BF16 compressor buffers honestly included.

NO SPEED CLAIMS: the Stage-C read path dequantizes in pure PyTorch and is expected to be
slower than baseline until Stage-D kernel fusion; GX10 timings are meaningless for the
target hardware and are deliberately not measured.

Dtype note: the tiny model runs fp32, so FP8 codes show ~4x on quantized slices; the real
checkpoint runs BF16, where the same layout gives ~2x on those slices. Ratios reported
here validate the accounting, not the final savings.

Usage:
    python tools/measure_cache_memory.py [--seq-len 96] [--decode 16] [--batch 2]
        [--seed 0] [--device cpu] [--json-out results/cache_memory.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import DynamicCache  # noqa: E402

from v4_kv_quant.memory import cache_memory_report, compare_reports  # noqa: E402
from v4_kv_quant.policy import reference_official_qdq  # noqa: E402
from v4_kv_quant.qdq_cache import QDQCache  # noqa: E402
from v4_kv_quant.storage_cache import QuantizedStorageCache  # noqa: E402
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids  # noqa: E402

BANNER = ("Stage C ACTUAL-STORAGE memory comparison - tiny fp32 model; correctness prototype; "
          "NO speed claims (pure-PyTorch dequantize-on-read; GX10 timings not representative)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default="results/cache_memory.json")
    args = parser.parse_args()

    model = build_tiny_model(seed=args.seed, device=args.device)
    config = model.config
    ids = deterministic_input_ids(args.batch, args.seq_len + args.decode).to(args.device)
    policy = reference_official_qdq()

    caches = {
        "baseline_bf16_stock": DynamicCache(config=config),
        "stage_b_qdq_sim": QDQCache(config, policy),
        "stage_c_storage": QuantizedStorageCache(config, policy),
    }

    print("=" * 110)
    print(BANNER)
    print("=" * 110)
    print(f"model dtype={model.dtype} | policy={policy.name} | batch={args.batch} "
          f"prefill={args.seq_len} decode={args.decode} seed={args.seed}")

    for name, cache in caches.items():
        with torch.no_grad():
            model(ids[:, : args.seq_len], past_key_values=cache, use_cache=True)
            for t in range(args.seq_len, args.seq_len + args.decode):
                model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)

    reports = {name: cache_memory_report(cache, label=name) for name, cache in caches.items()}

    print(f"\n{'cache':<24} {'logical bytes':>14} {'storage bytes':>14} {'vs baseline':>12}")
    print("-" * 68)
    base = reports["baseline_bf16_stock"]["total_logical_bytes"]
    for name, report in reports.items():
        ratio = report["total_logical_bytes"] / base
        print(f"{name:<24} {report['total_logical_bytes']:>14,} "
              f"{report['total_storage_bytes']:>14,} {ratio:>11.3f}x")

    comparison = compare_reports(reports["baseline_bf16_stock"], reports["stage_c_storage"])
    print(f"\nStage-C vs baseline: {comparison['bytes_saved']:,} bytes saved "
          f"(ratio {comparison['ratio']:.3f}); per-layer ratios "
          f"{['%.3f' % r for r in comparison['per_layer_ratio']]}")

    print("\nper-layer detail (stage_c_storage):")
    for layer in reports["stage_c_storage"]["layers"]:
        print(f"  layer {layer['layer_index']} [{layer['layer_class']}] "
              f"logical={layer['logical_bytes']:,}")
        for state, entry in layer["states"].items():
            print(f"    {state:<28} {str(tuple(entry['shape'])):<20} {entry['dtype']:<16} "
                  f"{entry['logical_bytes']:>8,} B")

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "label": BANNER,
        "config": vars(args),
        "model_dtype": str(model.dtype),
        "policy": policy.to_dict(),
        "reports": reports,
        "comparison": comparison,
    }, indent=2) + "\n")
    print(f"\nJSON written to {json_path}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
