"""
tests/verify_quantized_kv.py

Verification suite for quantized latent KV cache on DeepSeek-V2-Lite.

Compares up to five configurations:
  Baseline. BF16 latent cache (KVLatentCache, no quantization)
  A. Full FP8  — all channels quantized to FP8
  B. Mixed FP8/BF16 — sensitivity-guided, fp8_fraction channels FP8
  C. Full FP4  — all channels quantized to FP4 e2m1
  D. Mixed FP4/FP8/BF16 — 3-level sensitivity-guided config

Metrics reported:
  - Max |Δlogit| per decode step (primary correctness signal)
  - Mean |Δlogit| per decode step
  - Top-1 token match rate
  - Perplexity on a held-out set (optional, --eval-ppl)
  - Theoretical cache memory per configuration

Usage:
    # Full test suite (FP8 + FP4 + mixed):
    python tests/verify_quantized_kv.py \\
        --model-path ./models/DeepSeek-V2-Lite \\
        --n-calibration 128 --seq-len 512 --gen-tokens 64

    # FP8 only, skip all mixed/FP4 tests:
    python tests/verify_quantized_kv.py \\
        --model-path ./models/DeepSeek-V2-Lite \\
        --no-mixed --no-fp4

    # With perplexity evaluation:
    python tests/verify_quantized_kv.py \\
        --model-path ./models/DeepSeek-V2-Lite \\
        --eval-ppl --ppl-samples 64
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Repo root on path ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ops.kv_latent_cache import KVLatentCache
from ops.kv_latent_cache_quantized import QuantizedKVLatentCache
from ops.sensitivity import SensitivityAnalyzer
from src.calibration_data import load_calibration_data

# Shared sensitivity scores — computed once, reused across mixed tests
_cached_scores = None


def load_model_and_tokenizer(model_path: str, device: str):
    """Load DeepSeek-V2-Lite with the KVLatentCache patch applied."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from kv_patch import patch_kv_model

    print(f"[setup] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[setup] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = patch_kv_model(model)
    model.eval()
    print(f"[setup] Model loaded and patched")
    return model, tokenizer


@torch.no_grad()
def collect_logits(
    model,
    input_ids: torch.Tensor,
    cache,
    gen_tokens: int,
    device: str,
    reference_ids: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Collect logits using teacher forcing — both models see identical inputs.

    First runs the prompt through the model to fill the cache, then uses a
    fixed reference continuation (from the BF16 baseline) as the input at
    every decode step. This means both BF16 and FP8 models always see the
    same tokens, so logit differences measure only quantization error.

    Args:
        model:         Patched DeepSeek model.
        input_ids:     Prompt token IDs [1, prompt_len].
        cache:         KVLatentCache or QuantizedKVLatentCache instance.
        gen_tokens:    Number of positions to collect logits for.
        device:        Inference device.
        reference_ids: [1, gen_tokens] fixed continuation tokens to feed at
                       each decode step. If None (first/baseline run), uses
                       greedy decoding and returns the generated ids alongside
                       logits so they can be passed as reference_ids for
                       subsequent runs.

    Returns:
        If reference_ids is None:  (logits_list, generated_ids)
        If reference_ids provided: logits_list
    """
    logits_list = []
    generated_ids = []

    # ── Prefill: run the full prompt through the model ────────────────────────
    current_ids = input_ids.to(device)
    outputs = model(
        input_ids=current_ids,
        past_key_values=cache,
        use_cache=True,
    )
    past = outputs.past_key_values

    # First logit is the prediction after the last prompt token
    logit = outputs.logits[:, -1, :]
    logits_list.append(logit.squeeze(0).cpu())

    if reference_ids is None:
        next_token = logit.argmax(dim=-1, keepdim=True)
    else:
        next_token = reference_ids[:, 0:1].to(device)
    generated_ids.append(next_token.squeeze().item())

    # ── Decode: teacher-forced continuation ───────────────────────────────────
    for step in range(1, gen_tokens):
        outputs = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        logit = outputs.logits[:, -1, :]
        logits_list.append(logit.squeeze(0).cpu())

        if reference_ids is None:
            next_token = logit.argmax(dim=-1, keepdim=True)
        else:
            next_token = reference_ids[:, step:step+1].to(device)
        generated_ids.append(next_token.squeeze().item())

    if reference_ids is None:
        ref = torch.tensor(generated_ids, dtype=torch.long).unsqueeze(0)
        return logits_list, ref
    return logits_list


@torch.no_grad()
def compute_perplexity(
    model,
    samples: list[torch.Tensor],
    cache_factory,
    device: str,
) -> float:
    """Compute perplexity over a list of tokenized samples using a given cache."""
    total_nll = 0.0
    total_tokens = 0

    for sample in tqdm(samples, desc="  ppl"):
        input_ids = sample.to(device)
        cache = cache_factory()
        outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=False)
        logits = outputs.logits  # [1, S, vocab]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    return torch.exp(torch.tensor(total_nll / total_tokens)).item()


def compare_logits(
    baseline_logits: list[torch.Tensor],
    test_logits: list[torch.Tensor],
    label: str,
) -> dict:
    """Compare two logit sequences and return metrics."""
    assert len(baseline_logits) == len(test_logits)

    max_diffs = []
    mean_diffs = []
    top1_matches = []

    for base, test in zip(baseline_logits, test_logits):
        diff = (base - test).abs()
        max_diffs.append(diff.max().item())
        mean_diffs.append(diff.mean().item())
        top1_matches.append(base.argmax().item() == test.argmax().item())

    result = {
        "label": label,
        "max_abs_diff":    max(max_diffs),
        "mean_abs_diff":   sum(mean_diffs) / len(mean_diffs),
        "top1_match_rate": sum(top1_matches) / len(top1_matches),
        "top1_all_match":  (sum(top1_matches) / len(top1_matches)) >= 0.985,
    }
    return result


def print_results(results: list[dict]) -> None:
    w = 40
    print(f"\n{'='*70}")
    print(f"  Quantized KV Cache Verification Results")
    print(f"{'='*70}")
    for r in results:
        label = r["label"]
        print(f"\n  [{label}]")
        print(f"    max  |Δlogit|   : {r['max_abs_diff']:.6f}")
        print(f"    mean |Δlogit|   : {r['mean_abs_diff']:.6f}")
        print(f"    top-1 match     : {r['top1_match_rate']*100:.1f}%  {'[PASS]' if r['top1_all_match'] else '[FAIL]'}")
        if "ppl_baseline" in r:
            print(f"    PPL (BF16)      : {r['ppl_baseline']:.3f}")
            print(f"    PPL (quantized) : {r['ppl_quantized']:.3f}")
            print(f"    PPL degradation : {r['ppl_degradation']:+.3f}")
        if "memory_mb" in r:
            mem = r["memory_mb"]
            base_mem = r.get("baseline_memory_mb")
            if base_mem and base_mem > 0:
                reduction = (1 - mem / base_mem) * 100
                print(f"    cache memory    : {mem:.2f} MB  (BF16: {base_mem:.2f} MB,  -{reduction:.1f}%)")
            else:
                print(f"    cache memory    : {mem:.2f} MB")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Verify quantized latent KV cache")
    parser.add_argument("--model-path", required=True, help="Path to DeepSeek-V2-Lite")
    parser.add_argument("--prompt", default="Explain the attention mechanism in transformers in detail:", help="Prompt for generation tests")
    parser.add_argument("--gen-tokens", type=int, default=64, help="Tokens to generate per test")
    parser.add_argument("--n-calibration", type=int, default=128, help="Calibration samples for sensitivity")
    parser.add_argument("--seq-len", type=int, default=512, help="Token length per calibration sample")
    parser.add_argument("--fp8-fraction", type=float, default=0.8, help="Fraction of channels quantized to FP8 in mixed FP8/BF16 test")
    parser.add_argument("--fp4-fraction", type=float, default=0.5, help="Fraction of channels quantized to FP4 in 3-level mixed test")
    parser.add_argument("--bf16-fraction", type=float, default=0.2, help="Fraction of channels kept BF16 in 3-level mixed test (remainder split FP4/FP8)")
    parser.add_argument("--no-mixed", action="store_true", help="Skip mixed FP8/BF16 test (Test B)")
    parser.add_argument("--no-fp4", action="store_true", help="Skip FP4 tests (Test C and D)")
    parser.add_argument("--no-fp3", action="store_true", help="Skip FP3 test (Test E)")
    parser.add_argument("--no-int2", action="store_true", help="Skip INT2 test (Test F)")
    parser.add_argument("--sensitivity-method", choices=["weight_only", "activation"], default="activation")
    parser.add_argument("--scores-path", default=None, help="Load pre-computed sensitivity scores from file")
    parser.add_argument("--save-scores", default=None, help="Save sensitivity scores to file")
    parser.add_argument("--eval-ppl", action="store_true", help="Evaluate perplexity on held-out samples")
    parser.add_argument("--ppl-samples", type=int, default=64, help="Number of samples for PPL evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-results", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Quantized Latent KV Cache Verification")
    print(f"  Model  : {args.model_path}")
    print(f"  Device : {args.device}")
    print(f"{'='*70}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # ── Tokenize prompt ───────────────────────────────────────────────────────
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"[test] Prompt: {repr(args.prompt[:60])}")
    print(f"[test] Prompt tokens: {prompt_ids.shape[1]}")

    # ── Baseline: BF16 latent cache (greedy — generates reference continuation) ─
    print("\n── Baseline (BF16 latent cache) ─────────────────────────────────────")
    baseline_cache = KVLatentCache()
    baseline_logits, reference_ids = collect_logits(
        model, prompt_ids, baseline_cache, args.gen_tokens, args.device
    )
    print(f"  top-1 tokens: {[tokenizer.decode([l.argmax().item()]) for l in baseline_logits[:5]]}")
    print(f"  reference continuation: {tokenizer.decode(reference_ids[0].tolist())!r:.80}")

    # BF16 baseline cache size — used to compute reduction % for each test
    baseline_cache_mb = baseline_cache.cache_size_bytes()["total_bytes"] / 1e6
    print(f"  BF16 cache size: {baseline_cache_mb:.2f} MB")

    results = []

    # ── Test A: Full FP8 (all channels) ──────────────────────────────────────
    print("\n── Test A: Full FP8 (all channels quantized) ────────────────────────")
    full_fp8_cache = QuantizedKVLatentCache(quant_config=None, simulate_fp8=True)
    full_fp8_logits = collect_logits(
        model, prompt_ids, full_fp8_cache, args.gen_tokens, args.device,
        reference_ids=reference_ids,
    )
    res_a = compare_logits(baseline_logits, full_fp8_logits, "Full FP8 (100% channels)")
    res_a["memory_mb"] = full_fp8_cache.cache_size_bytes()["total_bytes"] / 1e6
    res_a["baseline_memory_mb"] = baseline_cache_mb
    results.append(res_a)
    print(f"  max |Δlogit| = {res_a['max_abs_diff']:.6f}  top-1 match = {res_a['top1_match_rate']*100:.1f}%")

    # ── Shared sensitivity scores (computed once, reused for B and D) ────────
    scores = None
    analyzer = None
    need_sensitivity = (not args.no_mixed) or (not args.no_fp4)
    if need_sensitivity:
        analyzer = SensitivityAnalyzer(model, method=args.sensitivity_method)
        if args.scores_path and Path(args.scores_path).exists():
            print(f"\n  Loading pre-computed scores from {args.scores_path}")
            scores = SensitivityAnalyzer.load_scores(args.scores_path)
        else:
            print(f"\n── Sensitivity analysis ({args.sensitivity_method}) ──────────────────────────────")
            calibration_samples = load_calibration_data(
                tokenizer,
                n_samples=args.n_calibration,
                seq_len=args.seq_len,
                seed=args.seed,
            )
            scores = analyzer.run(calibration_samples, device=args.device)
            if args.save_scores:
                analyzer.save_scores(scores, args.save_scores)
        analyzer.summary(scores)

    # ── Test B: Mixed FP8/BF16 ────────────────────────────────────────────────
    if not args.no_mixed:
        print(f"\n── Test B: Mixed FP8/BF16 ({args.fp8_fraction:.0%} FP8) ────────────────────")
        quant_config = analyzer.make_quant_config(scores, fp8_fraction=args.fp8_fraction)

        mixed_cache = QuantizedKVLatentCache(quant_config=quant_config)
        mixed_logits = collect_logits(
            model, prompt_ids, mixed_cache, args.gen_tokens, args.device,
            reference_ids=reference_ids,
        )
        res_b = compare_logits(baseline_logits, mixed_logits, f"Mixed FP8/BF16 ({args.fp8_fraction:.0%} FP8)")
        res_b["memory_mb"] = mixed_cache.cache_size_bytes()["total_bytes"] / 1e6
        res_b["baseline_memory_mb"] = baseline_cache_mb
        results.append(res_b)
        print(f"  max |Δlogit| = {res_b['max_abs_diff']:.6f}  top-1 match = {res_b['top1_match_rate']*100:.1f}%")
        print(mixed_cache.memory_report())

    # ── Test C: Full FP4 (all channels) ──────────────────────────────────────
    if not args.no_fp4:
        print("\n── Test C: Full FP4 (all channels quantized) ────────────────────────")
        full_fp4_cache = QuantizedKVLatentCache(precision="fp4")
        full_fp4_logits = collect_logits(
            model, prompt_ids, full_fp4_cache, args.gen_tokens, args.device,
            reference_ids=reference_ids,
        )
        res_c = compare_logits(baseline_logits, full_fp4_logits, "Full FP4 (100% channels)")
        res_c["memory_mb"] = full_fp4_cache.cache_size_bytes()["total_bytes"] / 1e6
        res_c["baseline_memory_mb"] = baseline_cache_mb
        results.append(res_c)
        print(f"  max |Δlogit| = {res_c['max_abs_diff']:.6f}  top-1 match = {res_c['top1_match_rate']*100:.1f}%")
        print(full_fp4_cache.memory_report())

    # ── Test D: 3-level mixed FP4/FP8/BF16 ───────────────────────────────────
    if not args.no_fp4 and not args.no_mixed:
        # fp8_fraction derived from the remaining channels after FP4 and BF16
        fp8_frac = max(0.0, 1.0 - args.fp4_fraction - args.bf16_fraction)
        label_d = (
            f"Mixed FP4/FP8/BF16 "
            f"({args.fp4_fraction:.0%} FP4 / {fp8_frac:.0%} FP8 / {args.bf16_fraction:.0%} BF16)"
        )
        print(f"\n── Test D: {label_d} ─────────")
        precision_config = analyzer.make_mixed_precision_config(
            scores,
            fp4_fraction=args.fp4_fraction,
            fp8_fraction=fp8_frac,
        )
        mixed3_cache = QuantizedKVLatentCache(precision_config=precision_config)
        mixed3_logits = collect_logits(
            model, prompt_ids, mixed3_cache, args.gen_tokens, args.device,
            reference_ids=reference_ids,
        )
        res_d = compare_logits(baseline_logits, mixed3_logits, label_d)
        res_d["memory_mb"] = mixed3_cache.cache_size_bytes()["total_bytes"] / 1e6
        res_d["baseline_memory_mb"] = baseline_cache_mb
        results.append(res_d)
        print(f"  max |Δlogit| = {res_d['max_abs_diff']:.6f}  top-1 match = {res_d['top1_match_rate']*100:.1f}%")
        print(mixed3_cache.memory_report())

    # ── Test E: Full FP3 ──────────────────────────────────────────────────────
    if not args.no_fp3:
        print("\n── Test E: Full FP3 (all channels quantized) ────────────────────────")
        full_fp3_cache = QuantizedKVLatentCache(precision="fp3")
        full_fp3_logits = collect_logits(
            model, prompt_ids, full_fp3_cache, args.gen_tokens, args.device,
            reference_ids=reference_ids,
        )
        res_e = compare_logits(baseline_logits, full_fp3_logits, "Full FP3 (100% channels)")
        res_e["memory_mb"] = full_fp3_cache.cache_size_bytes()["total_bytes"] / 1e6
        res_e["baseline_memory_mb"] = baseline_cache_mb
        results.append(res_e)
        print(f"  max |Δlogit| = {res_e['max_abs_diff']:.6f}  top-1 match = {res_e['top1_match_rate']*100:.1f}%")
        print(full_fp3_cache.memory_report())

    # ── Test F: Full INT2 ─────────────────────────────────────────────────────
    if not args.no_int2:
        print("\n── Test F: Full INT2 (all channels quantized) ───────────────────────")
        full_int2_cache = QuantizedKVLatentCache(precision="int2")
        full_int2_logits = collect_logits(
            model, prompt_ids, full_int2_cache, args.gen_tokens, args.device,
            reference_ids=reference_ids,
        )
        res_f = compare_logits(baseline_logits, full_int2_logits, "Full INT2 (100% channels)")
        res_f["memory_mb"] = full_int2_cache.cache_size_bytes()["total_bytes"] / 1e6
        res_f["baseline_memory_mb"] = baseline_cache_mb
        results.append(res_f)
        print(f"  max |Δlogit| = {res_f['max_abs_diff']:.6f}  top-1 match = {res_f['top1_match_rate']*100:.1f}%")
        print(full_int2_cache.memory_report())

    # ── Perplexity evaluation ─────────────────────────────────────────────────
    if args.eval_ppl:
        print(f"\n── Perplexity evaluation ({args.ppl_samples} samples) ───────────────────")
        ppl_samples = load_calibration_data(
            tokenizer,
            n_samples=args.ppl_samples,
            seq_len=args.seq_len,
            seed=args.seed + 1,   # different seed from calibration
        )

        print("  Computing baseline PPL (BF16)...")
        ppl_base = compute_perplexity(model, ppl_samples, KVLatentCache, args.device)
        print(f"  BF16 PPL = {ppl_base:.3f}")

        # PPL for each non-baseline result, matched by label order
        ppl_configs = [
            ("Full FP8",  lambda: QuantizedKVLatentCache(precision="fp8")),
        ]
        if not args.no_mixed:
            _qc = quant_config  # captured from Test B
            ppl_configs.append(
                (f"Mixed FP8/BF16 ({args.fp8_fraction:.0%})",
                 lambda: QuantizedKVLatentCache(quant_config=_qc))
            )
        
        if not args.no_fp4:
            ppl_configs.append(("Full FP4", lambda: QuantizedKVLatentCache(precision="fp4")))
        if not args.no_fp4 and not args.no_mixed:
            _pc = precision_config
            ppl_configs.append(("Mixed FP4/FP8/BF16",
                                lambda: QuantizedKVLatentCache(precision_config=_pc)))
        if not args.no_fp3:
            ppl_configs.append(("Full FP3", lambda: QuantizedKVLatentCache(precision="fp3")))
        if not args.no_int2:
            ppl_configs.append(("Full INT2", lambda: QuantizedKVLatentCache(precision="int2")))    

        for i, (lbl, factory) in enumerate(ppl_configs):
            print(f"  Computing {lbl} PPL...")
            ppl_q = compute_perplexity(model, ppl_samples, factory, args.device)
            print(f"  {lbl} PPL = {ppl_q:.3f}  (Δ = {ppl_q - ppl_base:+.3f})")
            if i < len(results):
                results[i]["ppl_baseline"]   = ppl_base
                results[i]["ppl_quantized"]  = ppl_q
                results[i]["ppl_degradation"] = ppl_q - ppl_base

    # ── Print summary ─────────────────────────────────────────────────────────
    print_results(results)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()