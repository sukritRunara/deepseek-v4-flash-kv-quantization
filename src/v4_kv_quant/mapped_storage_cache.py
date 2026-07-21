"""Stage-C storage for per-group `PrecisionMap`s (D-016).

`QuantizedStorageCache` stores uniform policies; calibrated maps are per
(layer, state, channel group). This module stores each mapped state as ordered
CHANNEL SEGMENTS — one low-precision store per map entry, raw BF16 for unmapped
gaps and the RoPE slice — so a mixed FP8/FP4/BF16 state has real packed storage.

Hard contract (test-enforced, mirrors D-009): logits AND indexer picks are
BITWISE identical to `MappedQDQCache` under the same map — every Stage-B mapped
quality number transfers to actual storage unchanged. This holds because each
segment reuses the `load(store(x)) == qdq(x)` primitives with exactly the group
size the mapped QDQ path passes to `_qdq_slice`.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)

from .policy import StatePolicy
from .precision_map import MapEntry, PrecisionMap
from .storage import StoredTensor, fp4_store, fp8_store, load
from .storage_cache import (
    QuantStore,
    _StorageCompressedMixin,
    _StorageWindowMixin,
    _indexer_store,
)


class SegmentedStore:
    """Per-channel-segment storage for one mapped main-KV state.

    Segments tile `[0, head_dim)` in ascending channel order: map entries become
    low-precision stores, everything else (gaps + RoPE) raw BF16. Interface is
    QuantStore-compatible so the storage-cache mixins work unchanged.
    """

    def __init__(self, entries: list[MapEntry], nope_end: int, rope_dim: int):
        self.segments: list[tuple[int, int, tuple[str, int] | None]] = []
        cursor = 0
        for e in sorted(entries, key=lambda e: e.start):
            if e.start < cursor:
                raise ValueError("overlapping map entries")
            if e.start > cursor:
                self.segments.append((cursor, e.start, None))
            self.segments.append((e.start, e.end, (e.kind, e.effective_scale_group())))
            cursor = e.end
        if cursor < nope_end:
            self.segments.append((cursor, nope_end, None))
        if rope_dim:
            self.segments.append((nope_end, nope_end + rope_dim, None))
        self._parts: list[dict[str, torch.Tensor]] = [{} for _ in self.segments]
        self._meta: list[tuple[int, int, torch.dtype] | None] = [None] * len(self.segments)

    # -- encode/decode ---------------------------------------------------------
    def encode(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        out = []
        for i, (start, end, spec) in enumerate(self.segments):
            piece = x[..., start:end]
            if spec is None:
                out.append({"raw": piece})
                continue
            kind, group = spec
            if kind == "fp8_e4m3":
                stored = fp8_store(piece, group_size=group, pow2_scale=True)
            elif kind == "fp4_e2m1":
                stored = fp4_store(piece, group_size=group)
            else:
                raise ValueError(f"kind {kind!r} not applicable to a channel segment")
            self._meta[i] = (stored.group_size, stored.width, stored.out_dtype)
            out.append({"codes": stored.codes, "scales": stored.scales})
        return out

    def decode(self, parts: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        pieces = []
        for i, (start, end, spec) in enumerate(self.segments):
            if spec is None:
                pieces.append(parts[i]["raw"])
            else:
                group, width, out_dtype = self._meta[i]
                pieces.append(load(StoredTensor(
                    parts[i]["codes"], parts[i]["scales"], group, width, out_dtype)))
        return torch.cat(pieces, dim=-1)

    def decode_all(self, reference: torch.Tensor) -> torch.Tensor:
        if not self._parts[0]:
            shape = list(reference.shape)
            shape[-2] = 0
            return reference.new_zeros(shape)
        return self.decode(self._parts)

    # -- persistence -----------------------------------------------------------
    def append(self, parts: list[dict[str, torch.Tensor]]) -> None:
        for i, new in enumerate(parts):
            if not self._parts[i]:
                self._parts[i] = {k: v.contiguous() for k, v in new.items()}
            else:
                self._parts[i] = {k: torch.cat([self._parts[i][k], new[k]], dim=-2)
                                  for k in self._parts[i]}

    def trim_last(self, keep: int) -> None:
        for i, part in enumerate(self._parts):
            if part:
                self._parts[i] = {k: v[..., -keep:, :].contiguous()
                                  for k, v in part.items()}

    @property
    def rows(self) -> int:
        return next(iter(self._parts[0].values())).shape[-2] if self._parts[0] else 0

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        out = {}
        for i, part in enumerate(self._parts):
            start, end, _ = self.segments[i]
            for name, t in part.items():
                out[f"seg{start}-{end}.{name}"] = t
        return out


def _mapped_indexer_store(entries: list[MapEntry]) -> QuantStore:
    if not entries:
        return QuantStore(StatePolicy())  # kind "bf16": raw storage
    e = entries[0]  # map validation enforces one full-coverage indexer entry
    return _indexer_store(StatePolicy(kind=e.kind, group_size=e.effective_scale_group()))


def _main_segmented(map_: PrecisionMap, layer_idx: int, state: str, config) -> SegmentedStore:
    rope = config.qk_rope_head_dim
    return SegmentedStore(map_.entries_for(layer_idx, state),
                          nope_end=config.head_dim - rope, rope_dim=rope)


class MappedStorageSlidingLayer(_StorageWindowMixin, DynamicSlidingWindowLayer):
    _layer_type = None

    def __init__(self, config, precision_map: PrecisionMap, layer_idx: int):
        super().__init__(sliding_window=config.sliding_window)
        self._window = _main_segmented(precision_map, layer_idx, "window_kv", config)

    def update(self, key_states, value_states, *args, **kwargs):
        full = self._window_update(key_states)
        return full, full


class MappedStorageHCALayer(_StorageCompressedMixin, DeepseekV4HCACache):
    _layer_type = None

    def __init__(self, config, precision_map: PrecisionMap, layer_idx: int):
        super().__init__(config)
        self._window = _main_segmented(precision_map, layer_idx, "window_kv", config)
        self._compressed = _main_segmented(precision_map, layer_idx, "compressed_kv", config)


class MappedStorageCSALayer(_StorageCompressedMixin, DeepseekV4CSACache):
    _layer_type = None

    def __init__(self, config, precision_map: PrecisionMap, layer_idx: int):
        super().__init__(config)
        self._window = _main_segmented(precision_map, layer_idx, "window_kv", config)
        self._compressed = _main_segmented(precision_map, layer_idx, "compressed_kv", config)
        self._indexer = _mapped_indexer_store(precision_map.entries_for(layer_idx, "indexer_kv"))

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if name == "indexer":
            return self._append_and_return(self._indexer, name, compressed)
        return super().update_compressor_states(name, compressed)

    def _store_tensors(self) -> dict[str, torch.Tensor]:
        out = super()._store_tensors()
        out |= {f"indexer_kv.{name}": t for name, t in self._indexer.tensors.items()}
        return out


_LAYER_CLASSES = {
    "sliding_attention": MappedStorageSlidingLayer,
    "compressed_sparse_attention": MappedStorageCSALayer,
    "heavily_compressed_attention": MappedStorageHCALayer,
}


class MappedStorageCache(Cache):
    """Drop-in `past_key_values` with actual per-group low-precision storage."""

    def __init__(self, config, precision_map: PrecisionMap):
        precision_map.validate(config)
        layers = [
            _LAYER_CLASSES[layer_type](config, precision_map, layer_idx)
            for layer_idx, layer_type in enumerate(config.layer_types)
        ]
        super().__init__(layers=layers)
        self.precision_map = precision_map

    def __repr__(self):
        return (f"MappedStorageCache(map={self.precision_map.name!r}, "
                f"layers={len(self.layers)})")
