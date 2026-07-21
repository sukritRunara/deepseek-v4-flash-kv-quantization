#!/usr/bin/env python
"""Real-model gate + measurement for MappedStorageCache (D-016).

Per map: (1) D-009-style bitwise gate — MappedStorageCache produces logits AND
indexer picks bitwise identical to MappedQDQCache (natural + random prompts,
prefill ~512 + 8 teacher-forced steps); (2) only if the gate passes, MEASURED
cache bytes: chunked prefill of a real held-out sequence at each length, then
`v4_kv_quant.memory.cache_memory_report` on the filled cache, with a stock-cache
baseline ratio. Cross-check for tools/compute_map_bytes.py's analytic numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.harness import run_teacher_forced  # noqa: E402
from v4_kv_quant.mapped_storage_cache import MappedStorageCache  # noqa: E402
from v4_kv_quant.memory import cache_memory_report  # noqa: E402
from v4_kv_quant.p2p_workaround import ensure_host_staged_p2p  # noqa: E402
from v4_kv_quant.precision_map import PrecisionMap  # noqa: E402

from b3_identity_gates import NATURAL_TEXT, bitwise_equal  # noqa: E402  (same dir)


@torch.no_grad()
def fill_cache(model, cache, ids: torch.Tensor, chunk: int) -> None:
    for start in range(0, ids.shape[1], chunk):
        model(ids[:, start : start + chunk], past_key_values=cache, use_cache=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="/home/sukrit/models/DeepSeek-V4-Flash")
    ap.add_argument("--maps", default="results/calibration_full/precision_map.json,"
                                      "results/calibration_full/precision_map_ladder20.json")
    ap.add_argument("--ids-file", default="results/calibration_full/token_ids_32k.json")
    ap.add_argument("--measure-lens", default="8192,32768")
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default="results/mapped_storage_gate.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    ensure_host_staged_p2p()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", attn_implementation="eager"
    ).eval()
    print("[loaded]", flush=True)
    device = model.device
    seq = args.prefill + args.decode

    natural = tok.encode(NATURAL_TEXT * 8, add_special_tokens=True)[:seq]
    prompts = {
        "natural": torch.tensor([natural], dtype=torch.long).to(device),
        f"random-{args.prefill}": torch.randint(
            0, model.config.vocab_size, (1, seq),
            generator=torch.Generator().manual_seed(0)).to(device),
    }
    long_ids = json.loads(Path(args.ids_file).read_text())["ids"][0]
    lens = [int(x) for x in args.measure_lens.split(",")]

    report: dict = {"maps": {}}
    failures: list[str] = []

    # stock-cache baseline bytes at each length (measured once)
    base_bytes = {}
    for L in lens:
        ids = torch.tensor([long_ids[:L]], dtype=torch.long).to(device)
        cache = DynamicCache(config=model.config)
        fill_cache(model, cache, ids, args.chunk)
        rep = cache_memory_report(cache, label=f"baseline@{L}")
        base_bytes[L] = rep["total_logical_bytes"]
        print(f"[measure baseline@{L}] {base_bytes[L] / 2**20:.1f} MiB", flush=True)
        del cache

    for map_path in [Path(p) for p in args.maps.split(",")]:
        pm = PrecisionMap.from_json(map_path)
        row: dict = {"path": str(map_path)}
        ok = True
        with torch.no_grad():
            for label, ids in prompts.items():
                sim = run_teacher_forced(model, ids, args.prefill, precision_map=pm)
                act = run_teacher_forced(model, ids, args.prefill, precision_map=pm,
                                         storage=True)
                same = bitwise_equal(sim, act)
                ok &= same
                row[f"gate_{label}"] = same
                print(f"[{'PASS' if same else 'FAIL'}] {map_path.stem} storage==qdq "
                      f"({label})", flush=True)
        if not ok:
            failures.append(map_path.stem)
            report["maps"][map_path.stem] = row
            continue  # do NOT measure with an unproven cache
        for L in lens:
            ids = torch.tensor([long_ids[:L]], dtype=torch.long).to(device)
            cache = MappedStorageCache(model.config, pm)
            fill_cache(model, cache, ids, args.chunk)
            rep = cache_memory_report(cache, label=f"{map_path.stem}@{L}")
            bytes_ = rep["total_logical_bytes"]
            row[f"measured_bytes_{L}"] = bytes_
            row[f"ratio_vs_baseline_{L}"] = bytes_ / base_bytes[L]
            print(f"[measure {map_path.stem}@{L}] {bytes_ / 2**20:.1f} MiB "
                  f"({bytes_ / base_bytes[L]:.3f}x baseline)", flush=True)
            del cache
        report["maps"][map_path.stem] = row

    report["baseline_bytes"] = {str(L): b for L, b in base_bytes.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("ALL GATES PASSED" if not failures else f"GATES FAILED: {failures}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
