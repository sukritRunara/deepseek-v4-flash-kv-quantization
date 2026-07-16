#!/usr/bin/env python
"""Inspect DeepSeek-V4 cache states on a tiny random model.

Builds a tiny randomly-initialized DeepSeek-V4 causal LM covering every V4
attention layer type (sliding / CSA / HCA), runs a prefill forward followed by
cached decode steps, then walks the cache object and emits:

  * a readable report on stdout;
  * a JSON document (--json-out) with one record per (layer, state):
      layer index, layer type, module class, cache class, state name, shape,
      dtype, device, entry count, persistent/buffer classification.

Requires only the pinned vendor/transformers checkout: no model weights, no
network. Runs on CPU by default; --device cuda works identically.

Usage:
    python tools/inspect_v4_cache.py [--seq-len 21] [--decode-steps 5]
        [--batch 2] [--device cpu] [--seed 0]
        [--json-out results/inspect_v4_cache.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids  # noqa: E402

# Classification of cache-layer attributes (see docs/V4_CACHE_ARCHITECTURE.md §2).
# persistent: grows/lives for the whole sequence. bounded_buffer: bounded temporary
# state (< compress_rate tokens, or exactly one overlap window). counter: python int.
STATE_CLASSIFICATION = {
    "keys": "persistent (sliding window, bounded to window-1)",
    "values": "persistent (alias of keys on CSA/HCA layers)",
    "compressed_kv": "persistent (append-only, one entry per compress_rate tokens)",
    "buffer_kv": "bounded_buffer (< compress_rate tokens, drained at window close)",
    "buffer_gate": "bounded_buffer (< compress_rate tokens, drained at window close)",
    "overlap_kv": "bounded_buffer (exactly one window Ca slice, CSA only)",
    "overlap_gate": "bounded_buffer (exactly one window Ca slice, CSA only)",
    "entry_count": "counter (int; entry_count * compress_rate = next window start position)",
    "cumulative_length": "counter (int; total tokens seen)",
}

DICT_STATES = ("buffer_kv", "buffer_gate", "compressed_kv", "overlap_kv", "overlap_gate", "entry_count")


def tensor_record(t: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
    }


def collect_layer_states(layer_idx: int, layer_type: str, attn_module, cache_layer) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = {
        "layer_index": layer_idx,
        "layer_type": layer_type,
        "module_class": type(attn_module).__name__,
        "cache_class": type(cache_layer).__name__,
    }

    keys, values = cache_layer.keys, cache_layer.values
    records.append(
        base
        | {"state": "keys", "classification": STATE_CLASSIFICATION["keys"], "entry_count": keys.shape[-2]}
        | tensor_record(keys)
    )
    records.append(
        base
        | {
            "state": "values",
            "classification": STATE_CLASSIFICATION["values"],
            "entry_count": values.shape[-2],
            "shares_storage_with_keys": bool(values.data_ptr() == keys.data_ptr()),
        }
        | tensor_record(values)
    )
    records.append(
        base
        | {
            "state": "cumulative_length",
            "classification": STATE_CLASSIFICATION["cumulative_length"],
            "entry_count": cache_layer.cumulative_length,
            "shape": None,
            "dtype": "int",
            "device": None,
        }
    )

    for attr in DICT_STATES:
        if not hasattr(cache_layer, attr):
            continue
        for name, value in getattr(cache_layer, attr).items():
            state_id = f"{attr}[{name}]"
            record = base | {"state": state_id, "classification": STATE_CLASSIFICATION[attr]}
            if torch.is_tensor(value):
                # entry axis: dim 1 for [B, T, D]-shaped compressor states
                record |= tensor_record(value) | {"entry_count": value.shape[1]}
            elif value is None:
                record |= {"shape": None, "dtype": None, "device": None, "entry_count": 0}
            else:  # int counter
                record |= {"shape": None, "dtype": "int", "device": None, "entry_count": value}
            records.append(record)
    return records


def runtime_assertions(model, cache, batch: int, total_tokens: int) -> list[str]:
    """Cheap invariants re-checked on every run (full suite: tests/test_v4_cache_semantics.py)."""
    config = model.config
    checks: list[str] = []
    window = config.sliding_window

    def check(name: str, condition: bool) -> None:
        checks.append(f"{'PASS' if condition else 'FAIL'}  {name}")
        if not condition:
            raise AssertionError(f"runtime assertion failed: {name}")

    for i, layer in enumerate(cache.layers):
        layer_type = config.layer_types[i]
        check(f"layer{i} cumulative_length == {total_tokens}", layer.cumulative_length == total_tokens)
        check(
            f"layer{i} window keys bounded to min(total, window-1)",
            layer.keys.shape[-2] == min(total_tokens, window - 1),
        )
        check(
            f"layer{i} keys shape [B, num_kv_heads=1, ., head_dim]",
            layer.keys.shape[0] == batch
            and layer.keys.shape[1] == config.num_key_value_heads
            and layer.keys.shape[-1] == config.head_dim,
        )
        if layer_type == "sliding_attention":
            check(f"layer{i} sliding: K == V contents", bool(torch.equal(layer.keys, layer.values)))
            continue
        check(f"layer{i} {layer_type}: K is V (shared storage)", layer.values.data_ptr() == layer.keys.data_ptr())
        rate = config.compress_rates[layer_type]
        expected_entries = total_tokens // rate
        expected_buffer = total_tokens % rate
        for name, kv in layer.compressed_kv.items():
            dim = config.index_head_dim if name == "indexer" else config.head_dim
            check(f"layer{i} compressed_kv[{name}] entries == {expected_entries}", kv.shape[1] == expected_entries)
            check(f"layer{i} compressed_kv[{name}] width == {dim}", kv.shape[-1] == dim)
            check(f"layer{i} entry_count[{name}] == {expected_entries}", layer.entry_count[name] == expected_entries)
        for name, buf in layer.buffer_kv.items():
            series = 2 if layer_type == "compressed_sparse_attention" else 1
            dim = config.index_head_dim if name == "indexer" else config.head_dim
            check(f"layer{i} buffer_kv[{name}] length == {expected_buffer}", buf.shape[1] == expected_buffer)
            check(f"layer{i} buffer_kv[{name}] width == {series}*{dim}", buf.shape[-1] == series * dim)
        if hasattr(layer, "overlap_kv"):
            for name, ov in layer.overlap_kv.items():
                if ov is None:
                    continue
                dim = config.index_head_dim if name == "indexer" else config.head_dim
                check(f"layer{i} overlap_kv[{name}] == [B, m={rate}, {dim}]", tuple(ov.shape) == (batch, rate, dim))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=21, help="prefill length (default 21: not window/rate aligned)")
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="results/inspect_v4_cache.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = build_tiny_model(seed=args.seed, device=args.device)
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 88)
    print("DeepSeek-V4 tiny-model cache inspection (no weights, deterministic)")
    print("=" * 88)
    print(f"device={args.device} seed={args.seed} params={total_params / 1e6:.2f}M dtype={model.dtype}")
    print(f"layer_types={config.layer_types}")
    print(f"sliding_window={config.sliding_window} compress_rates={config.compress_rates}")
    print(f"head_dim={config.head_dim} qk_rope_head_dim={config.qk_rope_head_dim} "
          f"index_head_dim={config.index_head_dim} index_topk={config.index_topk}")

    input_ids = deterministic_input_ids(args.batch, args.seq_len).to(args.device)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    token = input_ids[:, -1:]
    for _ in range(args.decode_steps):
        with torch.no_grad():
            out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
    total_tokens = args.seq_len + args.decode_steps

    if not torch.isfinite(out.logits).all():
        raise AssertionError("non-finite logits after decode")
    print(f"\nprefill {args.seq_len} tokens + {args.decode_steps} decode steps -> "
          f"{total_tokens} total; cache container: {type(cache).__name__}")

    records: list[dict[str, Any]] = []
    attn_modules = [layer.self_attn for layer in model.model.layers]
    for i, cache_layer in enumerate(cache.layers):
        layer_records = collect_layer_states(i, config.layer_types[i], attn_modules[i], cache_layer)
        records.extend(layer_records)
        print(f"\n-- layer {i} [{config.layer_types[i]}] "
              f"module={type(attn_modules[i]).__name__} cache={type(cache_layer).__name__}")
        for r in layer_records:
            shape = "-" if r["shape"] is None else str(tuple(r["shape"]))
            shared = " (shares keys storage)" if r.get("shares_storage_with_keys") else ""
            print(f"   {r['state']:<28} {shape:<18} {str(r['dtype']):<10} entries={r['entry_count']:<4}"
                  f" {r['classification']}{shared}")

    print("\nruntime assertions:")
    for line in runtime_assertions(model, cache, args.batch, total_tokens):
        print("  " + line)

    report = {
        "description": "DeepSeek-V4 tiny-model cache state inventory",
        "seed": args.seed,
        "device": args.device,
        "batch": args.batch,
        "prefill_tokens": args.seq_len,
        "decode_steps": args.decode_steps,
        "config": {
            "layer_types": config.layer_types,
            "sliding_window": config.sliding_window,
            "compress_rates": config.compress_rates,
            "head_dim": config.head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "index_head_dim": config.index_head_dim,
            "index_topk": config.index_topk,
            "num_key_value_heads": config.num_key_value_heads,
        },
        "cache_container": type(cache).__name__,
        "states": records,
    }
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nJSON report written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
