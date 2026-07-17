"""Stage-C cache layers: actual low-precision storage with dequantize-on-read.

Reuses the Task-02 `KVQuantPolicy` — the same kinds now select REAL storage formats:

  window_kv / compressed_kv:
      bf16      -> raw tensors (still fixes the sliding-layer K=V duplication)
      fp8_e4m3  -> float8_e4m3fn codes + e8m0/fp32 scales + raw RoPE slice
      fp4_e2m1  -> packed nibbles + e8m0 scales + raw RoPE slice
  indexer_kv:
      fp4_e2m1_hadamard -> Hadamard-rotated, packed nibbles + e8m0 scales (rotated basis,
                           symmetric query rotation via the Task-02 scorer wrapper)

Guarantees (test-enforced):
  * values returned to attention are BITWISE what the Stage-B QDQ cache returns — every
    Stage-B quality number transfers to Stage-C unchanged;
  * codes are written once (append-only / window-trim), never re-quantized;
  * `keys`/`values` remain empty placeholders — the quantized tensors are the only
    persistent KV storage (dequantized views are per-forward temporaries);
  * window trim re-contiguates so no hidden full-history storage is retained;
  * `cumulative_length` / `entry_count` bookkeeping identical to stock (compressor RoPE
    positions depend on `entry_count`);
  * compressor buffers / overlap state stay stock BF16 (never quantized, injection plan §5).

Correctness prototype: dequantize-on-read in pure PyTorch is expected to be SLOWER than
baseline until Stage-D fusion. Do not time it on GX10.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)

from .policy import KVQuantPolicy, StatePolicy
from .qdq import effective_group_size, hadamard_transform
from .storage import StoredTensor, fp4_store, fp8_store, load


class QuantStore:
    """Append/trim/decode storage for one cache state along its sequence dim (-2).

    Window states are `[B, 1, S, D]`, compressed/indexer states `[B, T, D]` — the
    sequence axis is dim -2 in both, so one implementation serves all states.
    """

    def __init__(self, policy: StatePolicy, rope_dim: int = 0, rotate: bool = False):
        self.kind = policy.kind
        self.requested_group = policy.group_size
        self.pow2_scale = policy.pow2_scale
        self.rope_dim = rope_dim
        self.rotate = rotate
        self.tensors: dict[str, torch.Tensor] = {}
        self._group: int | None = None
        self._width: int | None = None
        self._out_dtype: torch.dtype | None = None

    # -- encode/decode -------------------------------------------------------
    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize new rows `[..., S, D]` into storage parts (does not append)."""
        parts: dict[str, torch.Tensor] = {}
        if self.kind == "bf16":
            parts["raw"] = x
        else:
            body, rope = (x[..., : -self.rope_dim], x[..., -self.rope_dim :]) if self.rope_dim else (x, None)
            if self.rotate:
                body = hadamard_transform(body)
            group = effective_group_size(body.shape[-1], self.requested_group)
            if self.kind == "fp8_e4m3":
                stored = fp8_store(body, group_size=group, pow2_scale=self.pow2_scale)
            elif self.kind in ("fp4_e2m1", "fp4_e2m1_hadamard"):
                stored = fp4_store(body, group_size=group)
            else:
                raise ValueError(f"unknown storage kind {self.kind!r}")
            self._group, self._width, self._out_dtype = stored.group_size, stored.width, stored.out_dtype
            parts["codes"], parts["scales"] = stored.codes, stored.scales
            if rope is not None:
                parts["rope"] = rope
        return parts

    def decode(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "bf16":
            return parts["raw"]
        body = load(StoredTensor(parts["codes"], parts["scales"], self._group, self._width, self._out_dtype))
        if self.rope_dim:
            return torch.cat([body, parts["rope"]], dim=-1)
        return body  # rotated kinds stay in the rotated basis, matching the QDQ cache

    def decode_all(self, reference: torch.Tensor) -> torch.Tensor:
        """Dequantize the full store; empty `[..., 0, D]` (shaped like `reference`) if unwritten."""
        if not self.tensors:
            shape = list(reference.shape)
            shape[-2] = 0
            return reference.new_zeros(shape)
        return self.decode(self.tensors)

    # -- persistence ---------------------------------------------------------
    def append(self, parts: dict[str, torch.Tensor]) -> None:
        if not self.tensors:
            self.tensors = {k: v.contiguous() for k, v in parts.items()}
            return
        self.tensors = {k: torch.cat([self.tensors[k], parts[k]], dim=-2) for k in self.tensors}

    def trim_last(self, keep: int) -> None:
        """Keep the last `keep` sequence rows; re-contiguate so old storage is dropped."""
        if self.tensors:
            self.tensors = {k: v[..., -keep:, :].contiguous() for k, v in self.tensors.items()}

    @property
    def rows(self) -> int:
        return next(iter(self.tensors.values())).shape[-2] if self.tensors else 0


def _main_store(policy: StatePolicy, rope_dim: int) -> QuantStore:
    if policy.kind == "fp4_e2m1_hadamard":
        raise ValueError("rotation is not valid for main KV (would mix the RoPE slice)")
    return QuantStore(policy, rope_dim=rope_dim)


def _indexer_store(policy: StatePolicy) -> QuantStore:
    return QuantStore(policy, rope_dim=0, rotate=policy.kind == "fp4_e2m1_hadamard")


class _StorageWindowMixin:
    _window: QuantStore
    sliding_window: int

    def _window_update(self, key_states: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.lazy_initialization(key_states, key_states)
            self.values = self.keys  # both stay empty placeholders; quantized store is the storage
        self.cumulative_length += key_states.shape[-2]
        new_parts = self._window.encode(key_states)
        full = torch.cat([self._window.decode_all(key_states), self._window.decode(new_parts)], dim=-2)
        self._window.append(new_parts)
        self._window.trim_last(self.sliding_window - 1)
        return full

    def _store_tensors(self) -> dict[str, torch.Tensor]:
        out = {f"window_kv.{name}": t for name, t in self._window.tensors.items()}
        for attr in ("buffer_kv", "buffer_gate", "overlap_kv", "overlap_gate"):
            for name, value in getattr(self, attr, {}).items():
                if torch.is_tensor(value):
                    out[f"{attr}[{name}]"] = value
        return out

    def memory_state_tensors(self) -> dict[str, torch.Tensor]:
        return self._store_tensors()


class QuantizedStorageSlidingLayer(_StorageWindowMixin, DynamicSlidingWindowLayer):
    _layer_type = None

    def __init__(self, config, policy: KVQuantPolicy):
        super().__init__(sliding_window=config.sliding_window)
        self._window = _main_store(policy.window_kv, config.qk_rope_head_dim)

    def update(self, key_states, value_states, *args, **kwargs):
        full = self._window_update(key_states)
        return full, full  # K=V: also removes the stock layer's duplicated V copy


class _StorageCompressedMixin(_StorageWindowMixin):
    _compressed: QuantStore

    def update(self, key_states, value_states, *args, **kwargs):
        full = self._window_update(key_states)
        return full, full

    def _append_and_return(self, store: QuantStore, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if compressed.shape[-2] > 0:
            store.append(store.encode(compressed))
        self.entry_count[name] += compressed.shape[-2]
        return store.decode_all(compressed)

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if name == "compressor":
            return self._append_and_return(self._compressed, name, compressed)
        raise ValueError(f"unknown compressor state name {name!r}")

    def _store_tensors(self) -> dict[str, torch.Tensor]:
        out = super()._store_tensors()
        out |= {f"compressed_kv.{name}": t for name, t in self._compressed.tensors.items()}
        return out


class QuantizedStorageHCALayer(_StorageCompressedMixin, DeepseekV4HCACache):
    _layer_type = None

    def __init__(self, config, policy: KVQuantPolicy):
        super().__init__(config)
        rope_dim = config.qk_rope_head_dim
        self._window = _main_store(policy.window_kv, rope_dim)
        self._compressed = _main_store(policy.compressed_kv, rope_dim)


class QuantizedStorageCSALayer(_StorageCompressedMixin, DeepseekV4CSACache):
    _layer_type = None

    def __init__(self, config, policy: KVQuantPolicy):
        super().__init__(config)
        rope_dim = config.qk_rope_head_dim
        self._window = _main_store(policy.window_kv, rope_dim)
        self._compressed = _main_store(policy.compressed_kv, rope_dim)
        self._indexer = _indexer_store(policy.indexer_kv)

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if name == "indexer":
            return self._append_and_return(self._indexer, name, compressed)
        return super().update_compressor_states(name, compressed)

    def _store_tensors(self) -> dict[str, torch.Tensor]:
        out = super()._store_tensors()
        out |= {f"indexer_kv.{name}": t for name, t in self._indexer.tensors.items()}
        return out


_LAYER_CLASSES = {
    "sliding_attention": QuantizedStorageSlidingLayer,
    "compressed_sparse_attention": QuantizedStorageCSALayer,
    "heavily_compressed_attention": QuantizedStorageHCALayer,
}


class QuantizedStorageCache(Cache):
    """Drop-in `past_key_values` with actual low-precision storage per a `KVQuantPolicy`."""

    def __init__(self, config, policy: KVQuantPolicy):
        layers = [_LAYER_CLASSES[layer_type](config, policy) for layer_type in config.layer_types]
        super().__init__(layers=layers)
        self.policy = policy

    def __repr__(self):
        return f"QuantizedStorageCache(policy={self.policy.name!r}, layers={len(self.layers)})"


def build_storage_cache(config, policy: KVQuantPolicy) -> QuantizedStorageCache:
    return QuantizedStorageCache(config, policy)
