#!/usr/bin/env python
"""Full-model calibration on the RunPod 4-GPU pod (Phase B step B4, design D-012).

Mirrors tools/run_calibration_smoke.py on the real checkpoint with a real corpus
(C4-en + code slice, seeded/unpadded/non-overlapping; token ids saved; held-out from
disjoint stream regions). Stages (comma list via --stages; each invocation loads the
model once):

  corpus     build & save calibration/held-out token ids (or reuse existing)
  stats      pass-through activation statistics over the FULL calibration set
  screening  FP4 state-level sensitivity sweep (group = whole nope width) on the probe
  refine     group-level (64) FP4 sweep inside the --refine-top worst screening states
  fp8spot    FP8 spot-check on the --fp8-spot worst screening states
  map        build + validate precision map from refine (+ indexer screening) records
  heldout    evaluate official policy and the built map on held-out data (2k + 8k)

Simulation only (Stage B): NO memory savings. Compare within this node/run only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.calibration_data import build_corpus_samples  # noqa: E402
from v4_kv_quant.harness import run_teacher_forced  # noqa: E402
from v4_kv_quant.metrics import logit_comparison_metrics, next_token_nll  # noqa: E402
from v4_kv_quant.p2p_workaround import ensure_host_staged_p2p  # noqa: E402
from v4_kv_quant.policy import NAMED_POLICIES  # noqa: E402
from v4_kv_quant.sensitivity import build_map_from_sweep, run_sensitivity_sweep  # noqa: E402
from v4_kv_quant.stats import StatsCollectorCache, StatsRecorder  # noqa: E402
from v4_kv_quant.targets import INDEXER_STATE, enumerate_targets, nope_width  # noqa: E402

BANNER = "Stage B FULL-MODEL CALIBRATION - QDQ simulation only; NO memory savings"
DECODE_TAIL = 8  # teacher-forced decode steps after prefill in sweeps/evals


def load_model(path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_host_staged_p2p()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype="auto", device_map="auto", attn_implementation="eager"
    )
    return tok, model.eval()


def corpus_stage(args, tok, out: Path) -> dict:
    ids_file = out / "token_ids.json"
    if ids_file.exists():
        data = json.loads(ids_file.read_text())
        print(f"[corpus] reusing {ids_file}")
        return data
    calib_2k = build_corpus_samples(tok, args.calib_2k, args.seq_2k, seed=args.calib_seed)
    calib_8k = build_corpus_samples(tok, args.calib_8k, args.seq_8k, seed=args.calib_seed + 1)
    skip = max(
        max(calib_2k.provenance["tokens_consumed_per_source"].values()),
        max(calib_8k.provenance["tokens_consumed_per_source"].values()),
    )
    held_2k = build_corpus_samples(tok, args.held_2k, args.seq_2k, seed=args.heldout_seed, skip_tokens=skip)
    held_8k = build_corpus_samples(tok, args.held_8k, args.seq_8k, seed=args.heldout_seed + 1, skip_tokens=skip)
    data = {
        "calib_2k": {"ids": calib_2k.token_ids(), "provenance": calib_2k.provenance},
        "calib_8k": {"ids": calib_8k.token_ids(), "provenance": calib_8k.provenance},
        "held_2k": {"ids": held_2k.token_ids(), "provenance": held_2k.provenance},
        "held_8k": {"ids": held_8k.token_ids(), "provenance": held_8k.provenance},
    }
    ids_file.write_text(json.dumps(data) + "\n")
    print(f"[corpus] built and saved -> {ids_file}")
    return data


def tensors(data: dict, key: str) -> list[torch.Tensor]:
    return [torch.tensor([w], dtype=torch.long) for w in data[key]["ids"]]


@torch.no_grad()
def stats_stage(model, samples: list[torch.Tensor], out: Path) -> None:
    recorder = StatsRecorder(group_size_main=64)
    for i, ids in enumerate(samples):
        cache = StatsCollectorCache(model.config, recorder)
        model(ids.to("cuda"), past_key_values=cache, use_cache=True)
        del cache
        if (i + 1) % 8 == 0:
            print(f"[stats] {i + 1}/{len(samples)} sequences", flush=True)
    (out / "activation_stats.json").write_text(json.dumps(recorder.summary(), indent=2) + "\n")
    print(f"[stats] done -> activation_stats.json")


def probe_batch(data: dict, args) -> torch.Tensor:
    return torch.cat(tensors(data, "calib_2k")[: args.probe_batch], dim=0).to("cuda")


def sweep_stage(model, probe: torch.Tensor, targets, label: str, out: Path, kind_main: str):
    print(f"[{label}] {len(targets)} targets, probe {tuple(probe.shape)}", flush=True)
    _, baseline_nll, records = run_sensitivity_sweep(
        model, probe, probe.shape[1] - DECODE_TAIL, targets,
        kind_main=kind_main, progress=True,
    )
    (out / f"{label}.json").write_text(json.dumps({
        "baseline_nll": baseline_nll, "kind_main": kind_main,
        "records": [r.as_dict() for r in records],
    }, indent=2) + "\n")
    ranked = sorted(records, key=lambda r: r.score, reverse=True)
    for r in ranked[:5]:
        print(f"  worst: {r.target.key:<40} score={r.score:.3e}")
    return records


def load_records(out: Path, label: str):
    from v4_kv_quant.sensitivity import SensitivityRecord  # noqa: F401

    raw = json.loads((out / f"{label}.json").read_text())
    return raw["records"]


def worst_states(out: Path, top: int) -> list[tuple[int, str]]:
    raw = json.loads((out / "screening.json").read_text())
    mains = [r for r in raw["records"] if r["target"]["state"] != INDEXER_STATE]
    mains.sort(key=lambda r: r["score"], reverse=True)
    return [(r["target"]["layer_idx"], r["target"]["state"]) for r in mains[:top]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="/workspace/models/DeepSeek-V4-Flash")
    ap.add_argument("--stages", default="corpus,stats,screening")
    ap.add_argument("--out-dir", default="results/calibration_full")
    ap.add_argument("--calib-2k", type=int, default=32)
    ap.add_argument("--calib-8k", type=int, default=8)
    ap.add_argument("--held-2k", type=int, default=8)
    ap.add_argument("--held-8k", type=int, default=2)
    ap.add_argument("--seq-2k", type=int, default=2048)
    ap.add_argument("--seq-8k", type=int, default=8192)
    ap.add_argument("--calib-seed", type=int, default=11)
    ap.add_argument("--heldout-seed", type=int, default=12)
    ap.add_argument("--probe-batch", type=int, default=4)
    ap.add_argument("--refine-top", type=int, default=15)
    ap.add_argument("--fp8-spot", type=int, default=8)
    ap.add_argument("--fp8-fraction", type=float, default=0.75)
    ap.add_argument("--fp4-fraction", type=float, default=0.0)
    ap.add_argument("--indexer-min-overlap", type=float, default=0.9)
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(BANNER)
    print(f"stages={stages} out={out}")

    # Corpus first: it needs only the tokenizer, and streaming failures must not cost
    # a 13-minute model load (learned in run 1 — gated dataset, WORKLOG B4).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    data = corpus_stage(args, tok, out)
    if set(stages) == {"corpus"}:
        return 0

    _, model = load_model(args.model_path)
    config = model.config
    print("[loaded]", flush=True)

    if "stats" in stages:
        stats_stage(model, tensors(data, "calib_2k") + tensors(data, "calib_8k"), out)

    if "screening" in stages:
        state_targets = enumerate_targets(config, group_size_main=nope_width(config))
        sweep_stage(model, probe_batch(data, args), state_targets, "screening", out, kind_main="fp4_e2m1")

    if "refine" in stages:
        keep = set(worst_states(out, args.refine_top))
        group_targets = [t for t in enumerate_targets(config, group_size_main=64)
                         if (t.layer_idx, t.state) in keep]
        sweep_stage(model, probe_batch(data, args), group_targets, "refine", out, kind_main="fp4_e2m1")

    if "fp8spot" in stages:
        keep = set(worst_states(out, args.fp8_spot))
        spot_targets = [t for t in enumerate_targets(config, group_size_main=nope_width(config))
                        if (t.layer_idx, t.state) in keep]
        sweep_stage(model, probe_batch(data, args), spot_targets, "fp8spot", out, kind_main="fp8_e4m3")

    if "map" in stages:
        from v4_kv_quant.sensitivity import SensitivityRecord
        from v4_kv_quant.targets import QuantTarget

        raw = load_records(out, "refine") + [
            r for r in load_records(out, "screening") if r["target"]["state"] == INDEXER_STATE
        ]
        records = []
        for r in raw:
            r = dict(r)
            r.pop("score", None)  # derived property, not a constructor field
            records.append(SensitivityRecord(target=QuantTarget(**r.pop("target")), **r))
        pm = build_map_from_sweep(
            records, name=f"v4-flash-b4-{stamp}",
            fp8_fraction=args.fp8_fraction, fp4_fraction=args.fp4_fraction,
            indexer_min_overlap=args.indexer_min_overlap,
            provenance={"created_utc": stamp, "token_ids_file": "token_ids.json",
                        "design": "D-012", "torch": torch.__version__,
                        "probe_batch": args.probe_batch, "config": vars(args)},
        )
        pm.validate(config)
        pm.to_json(out / "precision_map.json")
        print(f"[map] {len(pm.entries)} entries -> precision_map.json")

    if "heldout" in stages:
        from v4_kv_quant.precision_map import PrecisionMap

        results = {}
        cases = {"held_2k": torch.cat(tensors(data, "held_2k")[:4], dim=0).to("cuda"),
                 "held_8k": tensors(data, "held_8k")[0].to("cuda")}
        for label, ids in cases.items():
            prefill = ids.shape[1] - DECODE_TAIL
            base = run_teacher_forced(model, ids, prefill)
            row = {}
            variants = {"official": {"policy": NAMED_POLICIES["reference_official_qdq"]()}}
            if (out / "precision_map.json").exists():
                variants["mapped"] = {"precision_map": PrecisionMap.from_json(out / "precision_map.json")}
            for vname, kw in variants.items():
                q = run_teacher_forced(model, ids, prefill, **kw)
                m = logit_comparison_metrics(base.logits, q.logits)
                m["nll_baseline"] = next_token_nll(base.logits, ids)
                m["nll_quantized"] = next_token_nll(q.logits, ids)
                m["nll_delta"] = m["nll_quantized"] - m["nll_baseline"]
                row[vname] = m
                print(f"[heldout] {label}/{vname}: KL={m['kl_mean']:.3e} "
                      f"top1={m['top1_agreement']:.4f} dNLL={m['nll_delta']:+.4e}", flush=True)
            results[label] = row
        (out / "heldout_eval.json").write_text(json.dumps(results, indent=2) + "\n")

    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
