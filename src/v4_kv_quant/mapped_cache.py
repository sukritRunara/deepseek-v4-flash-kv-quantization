"""Cache layers that apply a per-group `PrecisionMap` at the write boundary.

Same injection points and guarantees as `qdq_cache` (QDQ once per value, RoPE slice
untouched, `_layer_type = None` so the stock registry is never hijacked), but the
treatment is resolved per (layer, state, channel group) from the map instead of one
policy per state family. An empty map is bit-exact identity.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)

from .policy import KVQuantPolicy, StatePolicy
from .precision_map import MapEntry, PrecisionMap
from .qdq import fp4_e2m1_qdq, fp8_e4m3_qdq
from .qdq_cache import apply_indexer_policy, indexer_query_qdq


def _qdq_slice(values: torch.Tensor, kind: str, scale_group: int) -> torch.Tensor:
    if kind == "fp8_e4m3":
        return fp8_e4m3_qdq(values, group_size=scale_group, pow2_scale=True)
    if kind == "fp4_e2m1":
        return fp4_e2m1_qdq(values, group_size=scale_group)
    raise ValueError(f"kind {kind!r} not applicable to a channel slice")


def apply_main_entries(states: torch.Tensor, entries: list[MapEntry], rope_dim: int) -> torch.Tensor:
    """QDQ the mapped channel groups of a main-KV tensor `[..., head_dim]`.

    Entry channel ranges are relative to the nope slice; the RoPE slice and unmapped
    groups pass through bit-untouched.
    """
    if not entries or states.numel() == 0:
        return states
    nope_end = states.shape[-1] - rope_dim
    out = states.clone()
    for entry in entries:
        assert entry.end <= nope_end, "validated map guarantees entries stay inside the nope slice"
        piece = states[..., entry.start : entry.end]
        out[..., entry.start : entry.end] = _qdq_slice(piece, entry.kind, entry.effective_scale_group())
    return out


def _indexer_policy_from(entries: list[MapEntry]) -> StatePolicy:
    entry = entries[0]  # validation enforces a single full-coverage indexer entry per layer
    return StatePolicy(kind=entry.kind, group_size=entry.effective_scale_group())


def apply_indexer_entries(states: torch.Tensor, entries: list[MapEntry]) -> torch.Tensor:
    if not entries or states.numel() == 0:
        return states
    return apply_indexer_policy(states, _indexer_policy_from(entries))


class _MappedMixin:
    _map: PrecisionMap
    _rope_dim: int
    _layer_idx: int

    def _entries(self, state: str) -> list[MapEntry]:
        return self._map.entries_for(self._layer_idx, state)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        quantized = apply_main_entries(key_states, self._entries("window_kv"), self._rope_dim)
        return super().update(quantized, quantized, *args, **kwargs)

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if name == "compressor":
            compressed = apply_main_entries(compressed, self._entries("compressed_kv"), self._rope_dim)
        elif name == "indexer":
            compressed = apply_indexer_entries(compressed, self._entries("indexer_kv"))
        else:
            raise ValueError(f"unknown compressor state name {name!r}")
        return super().update_compressor_states(name, compressed)


class MappedSlidingWindowLayer(_MappedMixin, DynamicSlidingWindowLayer):
    _layer_type = None

    def __init__(self, config, precision_map: PrecisionMap, layer_idx: int):
        super().__init__(sliding_window=config.sliding_window)
        self._map = precision_map
        self._rope_dim = config.qk_rope_head_dim
        self._layer_idx = layer_idx

    # sliding-only layers have no compressor; only the window write applies
    def update_compressor_states(self, name, compressed):  # pragma: no cover - defensive
        raise RuntimeError("sliding_attention layers have no compressor states")


class MappedHCACacheLayer(_MappedMixin, DeepseekV4HCACache):
    _layer_type = None

    def __init__(self, config, precision_map: PrecisionMap, layer_idx: int):
        super().__init__(config)
        self._map = precision_map
        self._rope_dim = config.qk_rope_head_dim
        self._layer_idx = layer_idx


class MappedCSACacheLayer(_MappedMixin, DeepseekV4CSACache):
    _layer_type = None

    def __init__(self, config, precision_map: PrecisionMap, layer_idx: int):
        super().__init__(config)
        self._map = precision_map
        self._rope_dim = config.qk_rope_head_dim
        self._layer_idx = layer_idx


_LAYER_CLASSES = {
    "sliding_attention": MappedSlidingWindowLayer,
    "compressed_sparse_attention": MappedCSACacheLayer,
    "heavily_compressed_attention": MappedHCACacheLayer,
}


class MappedQDQCache(Cache):
    """Drop-in `past_key_values` applying a validated `PrecisionMap` at write time."""

    def __init__(self, config, precision_map: PrecisionMap):
        precision_map.validate(config)
        layers = [
            _LAYER_CLASSES[layer_type](config, precision_map, layer_idx)
            for layer_idx, layer_type in enumerate(config.layer_types)
        ]
        super().__init__(layers=layers)
        self.precision_map = precision_map

    def __repr__(self):
        return f"MappedQDQCache(map={self.precision_map.name!r}, entries={len(self.precision_map.entries)})"


def indexer_query_context(model, precision_map: PrecisionMap):
    """Symmetric query-QDQ context for maps that quantize indexer keys.

    The official kernel quantizes indexer queries and keys identically; a map with any
    indexer entry therefore also needs query QDQ. All indexer entries in one map must
    share a kind (asserted here) — per-layer query policies would require per-layer
    scorer wrappers, which `indexer_query_qdq` applies uniformly.
    """
    entries = precision_map.indexer_entries()
    if not entries:
        return indexer_query_qdq(model, KVQuantPolicy())  # identity: swaps nothing
    kinds = {(e.kind, e.effective_scale_group()) for e in entries}
    if len(kinds) != 1:
        raise ValueError(f"indexer entries must share one kind/scale-group, got {kinds}")
    return indexer_query_qdq(model, KVQuantPolicy(indexer_q=_indexer_policy_from(entries)))
