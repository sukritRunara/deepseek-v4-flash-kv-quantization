"""Honest cache memory accounting (CLAUDE.md: values + scales + padding + buffers).

Reports, per layer and per state tensor:
  * logical_bytes  — `numel() * element_size()`: what the state inherently needs;
  * storage_bytes  — `untyped_storage().nbytes()`: what the allocator actually holds.
    Stock dynamic layers keep window views into larger concatenated tensors, so
    storage_bytes can exceed logical_bytes (retained history) — reported, not hidden.

K=V aliasing is counted once (same `data_ptr`); the stock sliding layer's duplicated V
copy is itemized explicitly. Storage-cache layers expose their quantized tensors
(codes / scales / rope / raw + BF16 buffers) via `memory_state_tensors()`.
"""

from __future__ import annotations

import torch


def _tensor_entry(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "logical_bytes": t.numel() * t.element_size(),
        "storage_bytes": t.untyped_storage().nbytes(),
    }


def _stock_layer_tensors(layer) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    keys, values = layer.keys, layer.values
    if torch.is_tensor(keys):
        tensors["keys"] = keys
    if torch.is_tensor(values) and torch.is_tensor(keys):
        if values.data_ptr() == keys.data_ptr():
            pass  # K=V alias: count once
        else:
            tensors["values (duplicate of keys)"] = values
    for attr in ("compressed_kv", "buffer_kv", "buffer_gate", "overlap_kv", "overlap_gate"):
        for name, value in getattr(layer, attr, {}).items():
            if torch.is_tensor(value):
                tensors[f"{attr}[{name}]"] = value
    return tensors


def cache_memory_report(cache, label: str = "") -> dict:
    layers = []
    total_logical = 0
    total_storage = 0
    for i, layer in enumerate(cache.layers):
        if hasattr(layer, "memory_state_tensors"):
            tensors = layer.memory_state_tensors()
        else:
            tensors = _stock_layer_tensors(layer)
        states = {name: _tensor_entry(t) for name, t in tensors.items()}
        layer_logical = sum(s["logical_bytes"] for s in states.values())
        layer_storage = sum(s["storage_bytes"] for s in states.values())
        total_logical += layer_logical
        total_storage += layer_storage
        layers.append({
            "layer_index": i,
            "layer_class": type(layer).__name__,
            "logical_bytes": layer_logical,
            "storage_bytes": layer_storage,
            "states": states,
        })
    return {
        "label": label,
        "total_logical_bytes": total_logical,
        "total_storage_bytes": total_storage,
        "layers": layers,
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    base, cand = baseline["total_logical_bytes"], candidate["total_logical_bytes"]
    return {
        "baseline_label": baseline["label"],
        "candidate_label": candidate["label"],
        "baseline_logical_bytes": base,
        "candidate_logical_bytes": cand,
        "bytes_saved": base - cand,
        "ratio": cand / base if base else float("nan"),
        "per_layer_ratio": [
            (c["logical_bytes"] / b["logical_bytes"]) if b["logical_bytes"] else float("nan")
            for b, c in zip(baseline["layers"], candidate["layers"])
        ],
    }
