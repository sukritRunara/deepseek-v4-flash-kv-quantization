#!/usr/bin/env python
"""Cache benchmark CLI: baseline vs Stage-B QDQ vs Stage-C storage.

Configuration-driven (Phase 6): the SAME command structure runs the tiny model locally
and the full checkpoint on RunPod — only the JSON config differs:

    python tools/benchmark_cache.py --config configs/bench_tiny_local.json
    python tools/benchmark_cache.py --config configs/bench_runpod_4gpu.json

Records per variant and context length (per-trial + median): TTFT, prefill throughput,
decode throughput, inter-token latency, peak allocated/reserved memory per GPU, actual
cache bytes (scales included), and quantize/dequantize micro-overhead.

NON-TRANSFERABILITY: compare variants only within one run (same node, prompts, seeds,
batch, stack). Tiny-model/GX10 timings do not predict RTX PRO 6000 performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.bench import BenchSettings, run_benchmark  # noqa: E402

BANNER = ("CACHE BENCHMARK - same-node variant comparison only; "
          "tiny/GX10 timings are NON-TRANSFERABLE to target hardware")


def load_model(settings: BenchSettings):
    if settings.model_path:
        from transformers import AutoModelForCausalLM

        if torch.cuda.device_count() > 1:
            # Opt-in since D-014 (env V4_KV_FORCE_HOST_STAGED_P2P=1): required only on
            # hosts where tools/p2p_stress_check.py shows corruption (D-011, RunPod).
            from v4_kv_quant.p2p_workaround import ensure_host_staged_p2p

            ensure_host_staged_p2p()
        print(f"loading full checkpoint from {settings.model_path} "
              f"(device_map={settings.device_map}, dtype={settings.dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            dtype=settings.dtype or "auto",
            device_map=settings.device_map,
            attn_implementation="eager",
        )
        return model.eval()
    from v4_kv_quant.tiny_model import build_tiny_model

    return build_tiny_model(seed=settings.seed, device=settings.resolved_device())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/bench_tiny_local.json")
    parser.add_argument("--variants", default=None, help="comma list override (baseline,qdq,storage)")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--prompt-lens", default=None, help="comma list override")
    parser.add_argument("--decode-tokens", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json-out", default="results/benchmark_cache.json")
    args = parser.parse_args()

    overrides = {
        "variants": args.variants.split(",") if args.variants else None,
        "policy": args.policy,
        "batch": args.batch,
        "prompt_lens": [int(x) for x in args.prompt_lens.split(",")] if args.prompt_lens else None,
        "decode_tokens": args.decode_tokens,
        "trials": args.trials,
        "warmup": args.warmup,
        "device": args.device,
        "seed": args.seed,
    }
    settings = BenchSettings.from_file(args.config, **overrides)

    print("=" * 110)
    print(BANNER)
    print("=" * 110)
    model = load_model(settings)
    device = settings.resolved_device()
    print(f"config={args.config} | device={device} dtype={model.dtype} | policy={settings.policy}")
    print(f"batch={settings.batch} prompt_lens={settings.prompt_lens} decode={settings.decode_tokens} "
          f"trials={settings.trials} warmup={settings.warmup} seed={settings.seed}")

    report = run_benchmark(model, settings)

    header = (f"{'prompt':>7} {'variant':<10} {'TTFT ms':>10} {'prefill tok/s':>14} "
              f"{'decode tok/s':>13} {'ITL p50 ms':>11} {'cache KiB':>10} {'peak alloc MiB':>15}")
    print("\n" + header)
    print("-" * len(header))
    for row in report["results"]:
        median = row["median"]
        peak = row["trials"][-1]["peak_memory"]
        peak_mib = (max(g["peak_allocated_bytes"] for g in peak["per_gpu"]) / 2**20) if peak else float("nan")
        print(f"{row['prompt_len']:>7} {row['variant']:<10} {median['ttft_s'] * 1e3:>10.2f} "
              f"{median['prefill_tokens_per_s']:>14.1f} {median['decode_tokens_per_s']:>13.1f} "
              f"{median['itl_p50_ms']:>11.3f} {median['cache_logical_bytes'] / 1024:>10.1f} "
              f"{peak_mib:>15.1f}")

    print("\nquantize/dequantize micro-overhead (median):")
    for name, entry in report["quantization_overhead_microbench"].items():
        print(f"  {name:<32} {entry['median_us']:>10.1f} us   shape={tuple(entry['shape'])}")

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nJSON (per-trial + medians) written to {json_path}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
