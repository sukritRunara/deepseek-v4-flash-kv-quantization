"""Benchmark engine: baseline vs Stage-B QDQ vs Stage-C storage caches.

Fair-comparison rules (CLAUDE.md): every variant consumes IDENTICAL fixed token streams
(same prompts, same predetermined decode ids, same seeds), the same device and dtype,
with warmup runs discarded and every timed span synchronized on CUDA. Medians over
repeated trials are the headline numbers; per-trial values are always saved.

NON-TRANSFERABILITY: tiny-model and GX10 timings do not predict RTX PRO 6000 behavior
(CLAUDE.md constraint 8). The engine measures; claims belong to the RunPod phase.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache

from .memory import cache_memory_report
from .policy import NAMED_POLICIES, KVQuantPolicy
from .qdq import effective_group_size
from .qdq_cache import QDQCache, indexer_query_qdq
from .storage import fp4_store, fp8_store, load
from .storage_cache import QuantizedStorageCache
from .tiny_model import deterministic_input_ids

VARIANTS = ("baseline", "qdq", "storage")


@dataclass
class BenchSettings:
    """Fully describes one benchmark invocation. Loadable from JSON (config-driven)."""

    model: str = "tiny"  # "tiny" or ignored when model_path is set
    model_path: str | None = None  # full-checkpoint path (RunPod)
    device_map: str | None = None  # e.g. "auto" for multi-GPU full model
    dtype: str | None = None
    device: str = "auto"  # auto -> cuda if available else cpu (tiny model only)
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    policy: str = "reference_official_qdq"
    batch: int = 2
    prompt_lens: list[int] = field(default_factory=lambda: [64])
    decode_tokens: int = 32
    trials: int = 5
    warmup: int = 2
    seed: int = 0
    note: str = ""

    def __post_init__(self):
        unknown = [v for v in self.variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown variants {unknown}; valid: {VARIANTS}")
        if self.policy not in NAMED_POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; valid: {list(NAMED_POLICIES)}")
        if self.trials < 1 or self.warmup < 0:
            raise ValueError("trials must be >= 1 and warmup >= 0")

    @classmethod
    def from_file(cls, path: str | Path, **overrides) -> "BenchSettings":
        data = json.loads(Path(path).read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cache_for(variant: str, config, policy: KVQuantPolicy):
    if variant == "baseline":
        return DynamicCache(config=config)
    if variant == "qdq":
        return QDQCache(config, policy)
    if variant == "storage":
        return QuantizedStorageCache(config, policy)
    raise ValueError(variant)


def _peak_memory(device: str) -> dict[str, Any] | None:
    if not device.startswith("cuda"):
        return None
    per_gpu = []
    for i in range(torch.cuda.device_count()):
        per_gpu.append({
            "device": i,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(i),
        })
    return {"per_gpu": per_gpu}


@torch.no_grad()
def _one_trial(model, variant: str, policy: KVQuantPolicy, prompt_ids, decode_ids, device: str) -> dict:
    cache = _cache_for(variant, model.config, policy)
    if device.startswith("cuda"):
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    # TTFT: prompt prefill forward + first-token selection
    _sync(device)
    t0 = time.perf_counter()
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    first_token = out.logits[:, -1].argmax(-1)
    _sync(device)
    ttft_s = time.perf_counter() - t0
    del first_token

    # decode: predetermined ids keep every variant's token history identical
    itl_s: list[float] = []
    for t in range(decode_ids.shape[1]):
        _sync(device)
        t0 = time.perf_counter()
        out = model(decode_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        out.logits[:, -1].argmax(-1)
        _sync(device)
        itl_s.append(time.perf_counter() - t0)

    prompt_tokens = prompt_ids.shape[0] * prompt_ids.shape[1]
    decode_total_s = sum(itl_s)
    report = cache_memory_report(cache, label=variant)
    return {
        "ttft_s": ttft_s,
        "prefill_tokens_per_s": prompt_tokens / ttft_s,
        "itl_s": itl_s,
        "decode_tokens_per_s": (decode_ids.shape[0] * decode_ids.shape[1]) / decode_total_s,
        "peak_memory": _peak_memory(device),
        "cache_logical_bytes": report["total_logical_bytes"],
        "cache_storage_bytes": report["total_storage_bytes"],
    }


@torch.no_grad()
def quantization_overhead_microbench(config, device: str, context_tokens: int, batch: int, iters: int = 30) -> dict:
    """Median encode/decode cost of the quantized representations at realistic state shapes.

    Covers the CLAUDE.md 'cache write quantization overhead' and 'cache read/dequantization
    overhead' records without conflating them with full-model forward noise.
    """
    rope = config.qk_rope_head_dim
    nope = config.head_dim - rope
    group_main = effective_group_size(nope, 64)
    group_idx = effective_group_size(config.index_head_dim, 32)
    csa_rate = config.compress_rates["compressed_sparse_attention"]
    entries = max(1, context_tokens // csa_rate)

    shapes = {
        "fp8_encode_window_step": ("fp8_store", torch.randn(batch, 1, 1, nope, device=device), group_main),
        "fp8_decode_window_full": ("load8", torch.randn(batch, 1, config.sliding_window - 1, nope, device=device), group_main),
        "fp8_decode_compressed_full": ("load8", torch.randn(batch, entries, nope, device=device), group_main),
        "fp4_encode_indexer_entry": ("fp4_store", torch.randn(batch, 1, config.index_head_dim, device=device), group_idx),
        "fp4_decode_indexer_full": ("load4", torch.randn(batch, entries, config.index_head_dim, device=device), group_idx),
    }
    results = {}
    for name, (op, tensor, group) in shapes.items():
        stored = fp8_store(tensor, group_size=group) if op != "load4" else fp4_store(tensor, group_size=group)
        times = []
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            if op == "fp8_store":
                fp8_store(tensor, group_size=group)
            elif op == "fp4_store":
                fp4_store(tensor, group_size=group)
            else:
                load(stored)
            _sync(device)
            times.append(time.perf_counter() - t0)
        results[name] = {
            "median_us": statistics.median(times) * 1e6,
            "shape": list(tensor.shape),
            "iters": iters,
        }
    return results


@torch.no_grad()
def run_benchmark(model, settings: BenchSettings) -> dict:
    """Run the full variant x prompt-length matrix. Returns per-trial + median results."""
    device = settings.resolved_device()
    policy = NAMED_POLICIES[settings.policy]()
    vocab = model.config.vocab_size
    results = []
    for prompt_len in settings.prompt_lens:
        prompt_ids = deterministic_input_ids(settings.batch, prompt_len, vocab, seed=settings.seed).to(device)
        decode_ids = deterministic_input_ids(
            settings.batch, settings.decode_tokens, vocab, seed=settings.seed + 1
        ).to(device)
        for variant in settings.variants:
            context = indexer_query_qdq(model, policy) if variant in ("qdq", "storage") else None
            trials = []
            with context if context is not None else torch.no_grad():
                for _ in range(settings.warmup):
                    _one_trial(model, variant, policy, prompt_ids, decode_ids, device)
                for _ in range(settings.trials):
                    trials.append(_one_trial(model, variant, policy, prompt_ids, decode_ids, device))
            all_itl = [t for trial in trials for t in trial["itl_s"]]
            median = {
                "ttft_s": statistics.median(t["ttft_s"] for t in trials),
                "prefill_tokens_per_s": statistics.median(t["prefill_tokens_per_s"] for t in trials),
                "decode_tokens_per_s": statistics.median(t["decode_tokens_per_s"] for t in trials),
                "itl_p50_ms": statistics.median(all_itl) * 1e3,
                "cache_logical_bytes": trials[-1]["cache_logical_bytes"],
                "cache_storage_bytes": trials[-1]["cache_storage_bytes"],
            }
            results.append({
                "prompt_len": prompt_len,
                "variant": variant,
                "median": median,
                "trials": trials,
            })
    overhead = quantization_overhead_microbench(
        model.config, device, context_tokens=max(settings.prompt_lens) + settings.decode_tokens,
        batch=settings.batch,
    )
    return {
        "settings": settings.__dict__ | {"resolved_device": device},
        "policy": policy.to_dict(),
        "results": results,
        "quantization_overhead_microbench": overhead,
        "non_transferability": (
            "Timings measured on the local development machine with a tiny random model "
            "unless a full checkpoint path was configured; they do not predict target-"
            "hardware performance (CLAUDE.md constraint 8). Compare variants only within "
            "this run (same node, prompts, seeds, batch, stack)."
        ),
    }
