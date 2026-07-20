#!/usr/bin/env python
"""B3 instrumentation identity gates on the real checkpoint (committed driver).

Reconstruction of the Phase-B B3 scratchpad driver (its output is preserved at
``artifacts/phase_b_runpod/b3_gates_run2.log``; recipe in WORKLOG "Phase B - B3"),
committed so host re-validation (GCP_TRANSITION step 5b) never needs a rewrite:

  gate A  QDQCache(baseline_bf16) bitwise identical to the stock cache
          (logits AND indexer picks) - Stage-B identity on the real model;
  gate B  QuantizedStorageCache bitwise identical to QDQCache under
          reference_official_qdq (D-009 storage==qdq re-check on CUDA).

Both gates run on a natural and a random-id prompt (prefill ~512 tokens, 8
teacher-forced decode steps). Also printed (informative, NOT a gate): official-QDQ vs
baseline max|dlogit| - expected O(units), dominated by indexer top-k flips at
near-ties (D-004); quality is judged in B5 by top-k overlap + KL/NLL, not max-logit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.harness import TeacherForcedResult, run_teacher_forced  # noqa: E402
from v4_kv_quant.p2p_workaround import ensure_host_staged_p2p  # noqa: E402
from v4_kv_quant.policy import NAMED_POLICIES  # noqa: E402

# Deterministic natural-language prefill source (repeated to reach the target length).
NATURAL_TEXT = (
    "Pipeline parallelism splits a large neural network across multiple accelerators, "
    "assigning each device a contiguous block of layers so that activations flow from "
    "one device to the next like work moving down an assembly line. Its efficiency "
    "depends on keeping every device busy: micro-batching divides each batch into "
    "smaller chunks that occupy the pipeline stages concurrently, shrinking the idle "
    "bubbles at the start and end of every step. Key-value caches complicate this "
    "picture, because attention layers must retain state for every token generated so "
    "far, and the memory devoted to that state grows linearly with context length. "
    "Quantizing the cache to eight- or four-bit formats trades a small amount of "
    "numerical fidelity for a large reduction in memory traffic, which in turn allows "
    "longer contexts or larger batches on the same hardware. "
)


def bitwise_equal(a: TeacherForcedResult, b: TeacherForcedResult) -> bool:
    if not torch.equal(a.logits, b.logits):
        return False
    if len(a.indexer_picks) != len(b.indexer_picks):
        return False
    return all(torch.equal(x, y) for x, y in zip(a.indexer_picks, b.indexer_picks))


def gate(name: str, ok: bool, failures: list[str]) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: logits and indexer picks "
          f"{'bitwise identical' if ok else 'DIFFER'}", flush=True)
    if not ok:
        failures.append(name)


def run_prompt(model, label: str, ids: torch.Tensor, prefill: int,
               failures: list[str]) -> dict:
    baseline = run_teacher_forced(model, ids, prefill)
    qdq_identity = run_teacher_forced(
        model, ids, prefill, policy=NAMED_POLICIES["baseline_bf16"]())
    a_ok = bitwise_equal(baseline, qdq_identity)
    gate(f"gateA identity ({label})", a_ok, failures)

    official = NAMED_POLICIES["reference_official_qdq"]()
    qdq_official = run_teacher_forced(model, ids, prefill, policy=official)
    storage_official = run_teacher_forced(
        model, ids, prefill, policy=official, storage=True)
    b_ok = bitwise_equal(qdq_official, storage_official)
    gate(f"gateB storage==qdq ({label})", b_ok, failures)

    dlogit = (qdq_official.logits - baseline.logits).abs().max().item()
    print(f"[info] official-QDQ vs baseline max|dlogit| ({label}): {dlogit:.3e}",
          flush=True)
    return {"gateA_bitwise": a_ok, "gateB_bitwise": b_ok,
            "official_vs_baseline_max_abs_dlogit": dlogit}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="/home/sukrit/models/DeepSeek-V4-Flash")
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--out", default="results/b3_gates.json")
    args = ap.parse_args()
    seq_len = args.prefill + args.decode

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_host_staged_p2p()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", attn_implementation="eager"
    ).eval()
    print("[loaded]", flush=True)
    device = model.device

    natural_ids = tok.encode(NATURAL_TEXT * 8, add_special_tokens=True)
    if len(natural_ids) < seq_len:
        raise RuntimeError(f"natural prompt too short: {len(natural_ids)} < {seq_len}")
    prompts = {
        "natural": torch.tensor([natural_ids[:seq_len]], dtype=torch.long).to(device),
        f"random-{args.prefill}": torch.randint(
            0, model.config.vocab_size, (1, seq_len),
            generator=torch.Generator().manual_seed(0),
        ).to(device),
    }

    failures: list[str] = []
    report = {"model_path": args.model_path, "prefill": args.prefill,
              "decode": args.decode, "prompts": {}}
    with torch.no_grad():
        for label, ids in prompts.items():
            report["prompts"][label] = run_prompt(model, label, ids, args.prefill,
                                                  failures)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["all_gates_passed"] = not failures
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("ALL GATES PASSED" if not failures else f"GATES FAILED: {failures}",
          flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
