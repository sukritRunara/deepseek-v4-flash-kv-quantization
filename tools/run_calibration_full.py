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
  indexer8k  indexer sensitivity on an 8k probe — the 2k probe CANNOT measure it:
             index_topk=512 >= compressed entries at 2048 tokens, so selection never
             binds and overlap is 1.0 by construction (D-015)
  map        build + validate precision map: refine records (group-64, worst states)
             + screening records for non-refined states + indexer8k records;
             --map-suffix writes precision_map_<suffix>.json (candidate maps)
  heldout    evaluate official policy and EVERY precision_map*.json on held-out data
             (2k + 8k, chunked prefill)
  heldout32k evaluate official + final map on one 32k held-out sequence (D-012)

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
from v4_kv_quant.metrics import indexer_topk_overlap, logit_comparison_metrics, next_token_nll  # noqa: E402
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
def stats_stage(model, samples: list[torch.Tensor], out: Path, prefill_chunk: int) -> None:
    recorder = StatsRecorder(group_size_main=64)
    for i, ids in enumerate(samples):
        cache = StatsCollectorCache(model.config, recorder)
        ids = ids.to("cuda")
        # Chunked prefill through the same cache, as in the bench engine: one-shot 8192
        # OOMs a 96 GB card under eager attention (same class as WORKLOG B2; hit on the
        # first at-scale stats run, WORKLOG 2026-07-20 B4). Write structure is identical
        # chunked vs one-shot; values agree to ~1e-7 relative (chunk-shaped kernels
        # reorder reductions — D-004 path noise; test-pinned with tolerance).
        for start in range(0, ids.shape[1], prefill_chunk):
            model(ids[:, start : start + prefill_chunk], past_key_values=cache, use_cache=True)
        del cache
        if (i + 1) % 8 == 0:
            print(f"[stats] {i + 1}/{len(samples)} sequences", flush=True)
    (out / "activation_stats.json").write_text(json.dumps(recorder.summary(), indent=2) + "\n")
    print(f"[stats] done -> activation_stats.json")


def probe_batch(data: dict, args) -> torch.Tensor:
    return torch.cat(tensors(data, "calib_2k")[: args.probe_batch], dim=0).to("cuda")


def eval_variants(out: Path, args) -> dict:
    """official policy + every precision_map*.json in out + --heldout-policies."""
    from v4_kv_quant.precision_map import PrecisionMap

    variants = {"official": {"policy": NAMED_POLICIES["reference_official_qdq"]()}}
    for map_file in sorted(out.glob("precision_map*.json")):
        variants[map_file.stem] = {"precision_map": PrecisionMap.from_json(map_file)}
    for pname in [p.strip() for p in args.heldout_policies.split(",") if p.strip()]:
        variants[pname] = {"policy": NAMED_POLICIES[pname]()}
    return variants


def eval_cases(model, cases: dict, variants: dict, args, offload: bool = False) -> dict:
    """Teacher-forced baseline-vs-variant metrics per case; offload=True streams
    logits/picks to CPU with chunkwise-GPU metric math (65k-scale runs)."""
    dev = "cuda" if offload else None
    results = {}
    for label, ids in cases.items():
        prefill = ids.shape[1] - DECODE_TAIL
        base = run_teacher_forced(model, ids, prefill,
                                  prefill_chunk=args.stats_prefill_chunk,
                                  logits_to_cpu=offload)
        row = {}
        for vname, kw in variants.items():
            q = run_teacher_forced(model, ids, prefill,
                                   prefill_chunk=args.stats_prefill_chunk,
                                   logits_to_cpu=offload, **kw)
            m = logit_comparison_metrics(base.logits, q.logits, compute_device=dev)
            m["nll_baseline"] = next_token_nll(base.logits, ids, compute_device=dev)
            m["nll_quantized"] = next_token_nll(q.logits, ids, compute_device=dev)
            m["nll_delta"] = m["nll_quantized"] - m["nll_baseline"]
            m["indexer"] = indexer_topk_overlap(base.indexer_picks, q.indexer_picks)
            row[vname] = m
            print(f"[{label}/{vname}] KL={m['kl_mean']:.3e} "
                  f"top1={m['top1_agreement']:.4f} dNLL={m['nll_delta']:+.4e} "
                  f"idx_overlap={m['indexer'].get('mean_overlap')}", flush=True)
        results[label] = row
    return results


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
    ap.add_argument("--model-path", default="/home/sukrit/models/DeepSeek-V4-Flash")
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
    ap.add_argument("--probe8k-batch", type=int, default=2)
    ap.add_argument("--stats-prefill-chunk", type=int, default=2048)
    ap.add_argument("--refine-top", type=int, default=15)
    ap.add_argument("--fp8-spot", type=int, default=8)
    ap.add_argument("--fp8-fraction", type=float, default=0.75)
    ap.add_argument("--fp4-fraction", type=float, default=0.0)
    ap.add_argument("--indexer-min-overlap", type=float, default=0.9)
    ap.add_argument("--map-suffix", default="",
                    help="write precision_map_<suffix>.json (candidate maps, D-015)")
    ap.add_argument("--heldout-policies", default="",
                    help="comma list of NAMED_POLICIES to evaluate alongside official"
                         " + maps in heldout/heldout32k (e.g. step-0 all-FP4)")
    ap.add_argument("--n-32k", type=int, default=2)
    ap.add_argument("--n-65k", type=int, default=2)
    ap.add_argument("--retrieval-lens", default="8192,32768,65536")
    ap.add_argument("--retrieval-seqs", type=int, default=2)
    ap.add_argument("--retrieval-needles", type=int, default=8)
    ap.add_argument("--retrieval2-needles", type=int, default=16)
    ap.add_argument("--retrieval-seed", type=int, default=31)
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

    model_stages = {"stats", "screening", "refine", "fp8spot", "indexer8k",
                    "heldout", "heldout32k", "heldout65k", "retrieval", "retrieval2"}
    if model_stages & set(stages):
        _, model = load_model(args.model_path)
        config = model.config
        print("[loaded]", flush=True)
    else:  # map building/validation needs only the config — skip the 13-min load
        from transformers import AutoConfig

        model = None
        config = AutoConfig.from_pretrained(args.model_path)

    if "stats" in stages:
        stats_stage(model, tensors(data, "calib_2k") + tensors(data, "calib_8k"), out,
                    prefill_chunk=args.stats_prefill_chunk)

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

    if "indexer8k" in stages:
        idx_targets = [t for t in enumerate_targets(config, group_size_main=nope_width(config))
                       if t.state == INDEXER_STATE]
        probe8k = torch.cat(tensors(data, "calib_8k")[: args.probe8k_batch], dim=0).to("cuda")
        print(f"[indexer8k] {len(idx_targets)} indexer targets, probe {tuple(probe8k.shape)}, "
              f"prefill_chunk={args.stats_prefill_chunk} (2k probe cannot bind top-k, D-015)",
              flush=True)
        _, baseline_nll, records = run_sensitivity_sweep(
            model, probe8k, probe8k.shape[1] - DECODE_TAIL, idx_targets,
            progress=True, prefill_chunk=args.stats_prefill_chunk,
        )
        (out / "indexer8k.json").write_text(json.dumps({
            "baseline_nll": baseline_nll, "probe": list(probe8k.shape),
            "prefill_chunk": args.stats_prefill_chunk,
            "records": [r.as_dict() for r in records],
        }, indent=2) + "\n")
        for r in sorted(records, key=lambda r: r.score, reverse=True)[:5]:
            ov = (r.indexer_overlap or {}).get("mean_overlap")
            print(f"  worst: {r.target.key:<40} score={r.score:.3e} overlap={ov}")

    if "map" in stages:
        from v4_kv_quant.sensitivity import SensitivityRecord
        from v4_kv_quant.targets import QuantTarget

        # Composition (D-015): group-64 refine records inside the most sensitive
        # states; state-level screening records for every non-refined main state (the
        # insensitive majority must receive entries or it silently stays BF16);
        # indexer records from the 8k probe where top-k selection actually binds.
        refine_raw = load_records(out, "refine")
        refined_states = {(r["target"]["layer_idx"], r["target"]["state"]) for r in refine_raw}
        screening_raw = load_records(out, "screening")
        main_fallback = [
            r for r in screening_raw
            if r["target"]["state"] != INDEXER_STATE
            and (r["target"]["layer_idx"], r["target"]["state"]) not in refined_states
        ]
        if (out / "indexer8k.json").exists():
            indexer_raw = load_records(out, "indexer8k")
            indexer_source = "indexer8k"
        else:
            indexer_raw = [r for r in screening_raw if r["target"]["state"] == INDEXER_STATE]
            indexer_source = "screening-2k (WARNING: top-k never binds at 2k; D-015)"
        raw = refine_raw + main_fallback + indexer_raw
        records = []
        fields = ("kind", "nll_delta", "kl_mean", "max_abs_logit_err",
                  "top1_agreement", "indexer_overlap")
        for r in raw:
            r = dict(r)
            r.pop("score", None)  # derived property, not a constructor field
            target = QuantTarget(**r.pop("target"))
            kwargs = {k: r.pop(k) for k in fields if k in r}
            # leftovers (nan_count/inf_count, ...) were spread from `extra` by as_dict
            records.append(SensitivityRecord(target=target, extra=r, **kwargs))
        pm = build_map_from_sweep(
            records, name=f"v4-flash-b4-{stamp}",
            fp8_fraction=args.fp8_fraction, fp4_fraction=args.fp4_fraction,
            indexer_min_overlap=args.indexer_min_overlap,
            provenance={"created_utc": stamp, "token_ids_file": "token_ids.json",
                        "design": "D-012 + D-015", "torch": torch.__version__,
                        "probe_batch": args.probe_batch,
                        "composition": {"refine_group64": len(refine_raw),
                                        "screening_state_level": len(main_fallback),
                                        "indexer": len(indexer_raw),
                                        "indexer_source": indexer_source},
                        "config": vars(args)},
        )
        pm.validate(config)
        suffix = f"_{args.map_suffix}" if args.map_suffix else ""
        pm.to_json(out / f"precision_map{suffix}.json")
        print(f"[map] {len(pm.entries)} entries -> precision_map{suffix}.json "
              f"(indexer source: {indexer_source})")

    if "heldout" in stages:
        # Grown held-out usage (FUTURE_WORK #3): ALL built windows — 8×2k as two
        # batch-4 cases, both 8k sequences.
        h2k = tensors(data, "held_2k")
        h8k = tensors(data, "held_8k")
        cases = {"held_2k_a": torch.cat(h2k[:4], dim=0).to("cuda"),
                 "held_2k_b": torch.cat(h2k[4:8], dim=0).to("cuda")}
        for i, t in enumerate(h8k):
            cases[f"held_8k_{'ab'[i]}"] = t.to("cuda")
        results = eval_cases(model, cases, eval_variants(out, args), args)
        (out / "heldout_eval.json").write_text(json.dumps(results, indent=2) + "\n")

    if "heldout32k" in stages or "heldout65k" in stages:
        # Long-sequence held-out ids: chained disjoint stream regions (D-012).
        consumed = [max(data[k]["provenance"]["tokens_consumed_per_source"].values())
                    for k in ("calib_2k", "calib_8k", "held_2k", "held_8k")]
        skip32 = max(consumed[:2]) + max(consumed[2:])

        def long_ids(path: Path, n: int, seq_len: int, seed: int, skip: int) -> dict:
            if path.exists():
                blob = json.loads(path.read_text())
                if len(blob["ids"]) >= n:
                    print(f"[long-heldout] reusing {path}")
                    return blob
            sample = build_corpus_samples(tok, n, seq_len, seed=seed, skip_tokens=skip)
            blob = {"ids": sample.token_ids(), "provenance": sample.provenance,
                    "skip_tokens": skip}
            path.write_text(json.dumps(blob) + "\n")
            return blob

    if "heldout32k" in stages:
        ids32 = long_ids(out / "token_ids_32k.json", args.n_32k, 32768,
                         args.heldout_seed + 2, skip32)
        cases = {f"held_32k_{'abcd'[i]}": torch.tensor([w], dtype=torch.long).to("cuda")
                 for i, w in enumerate(ids32["ids"][: args.n_32k])}
        results = eval_cases(model, cases, eval_variants(out, args), args)
        (out / "heldout32k_eval.json").write_text(json.dumps(results, indent=2) + "\n")

    if "heldout65k" in stages:
        skip65 = skip32 + 2_000_000  # own region, far past every 32k consumption
        ids65 = long_ids(out / "token_ids_65k.json", args.n_65k, 65536,
                         args.heldout_seed + 3, skip65)
        cases = {f"held_65k_{'abcd'[i]}": torch.tensor([w], dtype=torch.long).to("cuda")
                 for i, w in enumerate(ids65["ids"][: args.n_65k])}
        # logits offload: a 65k run's logits are ~17 GiB — cannot stay on GPU 0.
        results = eval_cases(model, cases, eval_variants(out, args), args, offload=True)
        (out / "heldout65k_eval.json").write_text(json.dumps(results, indent=2) + "\n")

    def run_retrieval(prefix: str, generator, n_needles: int, skip_offset: int) -> None:
        from v4_kv_quant.retrieval import (
            Needle,
            RetrievalSample,
            build_retrieval_sample,
            score_retrieval,
        )

        rid_file = out / f"{prefix}_ids.json"
        lens = [int(x) for x in args.retrieval_lens.split(",")]
        if rid_file.exists():
            rblob = json.loads(rid_file.read_text())
            print(f"[{prefix}] reusing {rid_file}")
        else:
            consumed = [max(data[k]["provenance"]["tokens_consumed_per_source"].values())
                        for k in ("calib_2k", "calib_8k", "held_2k", "held_8k")]
            # own stream region, far past all NLL held-out builds (which consume <2M)
            skip_r = max(consumed[:2]) + max(consumed[2:]) + skip_offset
            rblob = {"samples": [], "skip_tokens_base": skip_r,
                     "needles_per_sample": n_needles}
            for li, seq_len in enumerate(lens):
                fill = build_corpus_samples(tok, args.retrieval_seqs, seq_len,
                                            seed=args.retrieval_seed + li,
                                            skip_tokens=skip_r)
                for si, window in enumerate(fill.token_ids()):
                    needles = generator(
                        tok, n_needles,
                        seed=args.retrieval_seed * 1000 + li * 100 + si)
                    sample = build_retrieval_sample(window, needles, seq_len)
                    rblob["samples"].append({
                        "length": seq_len, "input_ids": sample.input_ids,
                        "needles": [vars(n) for n in sample.needles],
                    })
                skip_r += max(fill.provenance["tokens_consumed_per_source"].values())
            rid_file.write_text(json.dumps(rblob) + "\n")
            print(f"[{prefix}] built {len(rblob['samples'])} samples -> {rid_file}")

        variants = {"baseline": {}} | eval_variants(out, args)
        per_sample: dict = {}
        for s in rblob["samples"]:
            sample = RetrievalSample(
                input_ids=s["input_ids"],
                needles=[Needle(**d) for d in s["needles"]],
            )
            ids = torch.tensor([s["input_ids"]], dtype=torch.long).to("cuda")
            base_picks = None
            for vname, kw in variants.items():
                run = run_teacher_forced(model, ids, prefill_len=ids.shape[1],
                                         prefill_chunk=args.stats_prefill_chunk,
                                         logits_to_cpu=True, **kw)
                sc = score_retrieval(run.logits, sample)
                # per-kind breakdown (plain vs updated) for interference analysis
                by_kind = {}
                for rec, n_ in zip(sc["needles"], sample.needles):
                    by_kind.setdefault(n_.kind, []).append(rec["token_acc"])
                sc["token_acc_by_kind"] = {k: sum(v) / len(v) for k, v in by_kind.items()}
                if vname == "baseline":
                    base_picks = run.indexer_picks
                else:
                    sc["indexer_vs_baseline"] = indexer_topk_overlap(
                        base_picks, run.indexer_picks)
                per_sample.setdefault(str(s["length"]), {}).setdefault(
                    vname, []).append(sc)
                print(f"[{prefix} {s['length']}/{vname}] acc={sc['token_acc']:.3f} "
                      f"exact={sc['exact_rate']:.3f} nll={sc['nll_mean']:.3f} "
                      f"by_kind={sc['token_acc_by_kind']}", flush=True)
                del run
        summary = {}
        for length, rows in per_sample.items():
            summary[length] = {}
            for vname, scs in rows.items():
                summary[length][vname] = {
                    "token_acc": sum(x["token_acc"] for x in scs) / len(scs),
                    "exact_rate": sum(x["exact_rate"] for x in scs) / len(scs),
                    "nll_mean": sum(x["nll_mean"] for x in scs) / len(scs),
                    "n_samples": len(scs),
                }
        (out / f"{prefix}_eval.json").write_text(json.dumps(
            {"summary": summary, "per_sample": per_sample}, indent=2) + "\n")
        print(f"[{prefix}] done -> {prefix}_eval.json")

    if "retrieval" in stages:
        from v4_kv_quant.retrieval import make_needles_text

        run_retrieval("retrieval", make_needles_text, args.retrieval_needles,
                      skip_offset=5_000_000)

    if "retrieval2" in stages:
        from v4_kv_quant.retrieval import make_needles_text_v2

        # v2 (D-016): paraphrased cues + fact updates + name collisions; own region
        run_retrieval("retrieval2", make_needles_text_v2, args.retrieval2_needles,
                      skip_offset=10_000_000)

    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
