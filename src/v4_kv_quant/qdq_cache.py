"""QDQ-simulating cache layers for DeepSeek-V4 (Stage B — numerics only, BF16/FP32 storage).

Injection strategy (docs/QUANTIZATION_INJECTION_PLAN.md): every quantizable cache write
flows through `update()` (window KV) or `update_compressor_states()` (compressed entries,
main + indexer), both owned by per-layer cache classes. We subclass those classes and QDQ
the *incoming* states before delegating to the stock implementation:

* QDQ happens exactly once per value (window write / entry emission) — appended history is
  never touched again, which the tests pin bitwise;
* the trailing `qk_rope_head_dim` dims of main KV are never modified (precise slice);
* indexer entries follow the official path: Hadamard-rotate then FP4-QDQ the full vector,
  stored in the ROTATED basis — queries are rotated symmetrically by `indexer_query_qdq`,
  so scorer dot products match the official rotated-space computation (model.py:414-420);
* `_layer_type = None` on every subclass: `CacheLayerMixin.__init_subclass__` would otherwise
  re-register these classes over the stock ones in `DYNAMIC_LAYER_TYPE_MAPPING`, silently
  hijacking every `DynamicCache(config=...)` construction, including baselines.

No production cache class, attention module, or generated modeling file is modified.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)

from .policy import KVQuantPolicy, StatePolicy
from .qdq import effective_group_size, fp4_e2m1_qdq, fp8_e4m3_qdq, hadamard_transform


def apply_main_kv_policy(states: torch.Tensor, policy: StatePolicy, rope_dim: int) -> torch.Tensor:
    """QDQ the non-RoPE slice of a main-KV tensor `[..., head_dim]`; RoPE slice untouched."""
    if policy.is_identity or states.numel() == 0:
        return states
    nope, rope = states[..., :-rope_dim], states[..., -rope_dim:]
    group = effective_group_size(nope.shape[-1], policy.group_size)
    if policy.kind == "fp8_e4m3":
        quantized = fp8_e4m3_qdq(nope, group_size=group, pow2_scale=policy.pow2_scale)
    elif policy.kind == "fp4_e2m1":
        quantized = fp4_e2m1_qdq(nope, group_size=group)
    else:
        raise ValueError(f"policy kind {policy.kind!r} not valid for main KV (rotation mixes RoPE dims)")
    return torch.cat([quantized, rope], dim=-1)


def apply_indexer_policy(states: torch.Tensor, policy: StatePolicy) -> torch.Tensor:
    """QDQ a full indexer vector `[..., index_head_dim]` (keys or queries)."""
    if policy.is_identity or states.numel() == 0:
        return states
    if policy.kind == "fp4_e2m1_hadamard":
        states = hadamard_transform(states)
    group = effective_group_size(states.shape[-1], policy.group_size)
    if policy.kind in ("fp4_e2m1", "fp4_e2m1_hadamard"):
        return fp4_e2m1_qdq(states, group_size=group)
    if policy.kind == "fp8_e4m3":
        return fp8_e4m3_qdq(states, group_size=group, pow2_scale=policy.pow2_scale)
    raise ValueError(f"unknown policy kind {policy.kind!r}")


class QDQSlidingWindowLayer(DynamicSlidingWindowLayer):
    """Sliding-only layer with QDQ at the window write boundary (official: model.py:506)."""

    _layer_type = None  # do NOT register over the stock class

    def __init__(self, config, policy: KVQuantPolicy):
        super().__init__(sliding_window=config.sliding_window)
        self._policy = policy
        self._rope_dim = config.qk_rope_head_dim

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        quantized = apply_main_kv_policy(key_states, self._policy.window_kv, self._rope_dim)
        # V4 is K=V; pass the same quantized tensor for both slots so the returned
        # `full` states (what attention consumes) carry the QDQ'd values.
        return super().update(quantized, quantized, *args, **kwargs)


class _QDQCompressedMixin:
    """Shared QDQ overrides for the CSA/HCA cache layers (window + compressed writes)."""

    _policy: KVQuantPolicy
    _rope_dim: int

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        quantized = apply_main_kv_policy(key_states, self._policy.window_kv, self._rope_dim)
        return super().update(quantized, quantized, *args, **kwargs)

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        # QDQ only the NEWLY EMITTED entries; super() appends them, so history is
        # structurally never re-quantized (acceptance gate: QDQ exactly once).
        if name == "compressor":
            compressed = apply_main_kv_policy(compressed, self._policy.compressed_kv, self._rope_dim)
        elif name == "indexer":
            compressed = apply_indexer_policy(compressed, self._policy.indexer_kv)
        else:
            raise ValueError(f"unknown compressor state name {name!r}")
        return super().update_compressor_states(name, compressed)


class QDQHCACacheLayer(_QDQCompressedMixin, DeepseekV4HCACache):
    _layer_type = None

    def __init__(self, config, policy: KVQuantPolicy):
        super().__init__(config)
        self._policy = policy
        self._rope_dim = config.qk_rope_head_dim


class QDQCSACacheLayer(_QDQCompressedMixin, DeepseekV4CSACache):
    _layer_type = None

    def __init__(self, config, policy: KVQuantPolicy):
        super().__init__(config)
        self._policy = policy
        self._rope_dim = config.qk_rope_head_dim


_LAYER_CLASSES = {
    "sliding_attention": QDQSlidingWindowLayer,
    "compressed_sparse_attention": QDQCSACacheLayer,
    "heavily_compressed_attention": QDQHCACacheLayer,
}


class QDQCache(Cache):
    """Drop-in `past_key_values` whose layers apply a `KVQuantPolicy` at write time."""

    def __init__(self, config, policy: KVQuantPolicy):
        layers = [_LAYER_CLASSES[layer_type](config, policy) for layer_type in config.layer_types]
        super().__init__(layers=layers)
        self.policy = policy

    def __repr__(self):
        return f"QDQCache(policy={self.policy.name!r}, layers={len(self.layers)})"


def build_qdq_cache(config, policy: KVQuantPolicy) -> QDQCache:
    return QDQCache(config, policy)


class QDQIndexerScorer(nn.Module):
    """Wraps `DeepseekV4IndexerScorer`; QDQs the (post-RoPE) queries symmetrically with the
    indexer keys before scoring — the official kernel quantizes both sides (model.py:414-416).
    The scorer is the only module that receives q after RoPE, so wrapping it needs no edit
    to `DeepseekV4Indexer.forward`."""

    def __init__(self, inner: nn.Module, policy: StatePolicy):
        super().__init__()
        self.inner = inner
        self._policy = policy

    def forward(self, q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor):
        return self.inner(apply_indexer_policy(q, self._policy), compressed_kv, hidden_states)


@contextmanager
def indexer_query_qdq(model, policy: KVQuantPolicy):
    """Temporarily swap every CSA indexer's scorer for a query-QDQ wrapper; always restores."""
    swapped: list[tuple[nn.Module, nn.Module]] = []
    if not policy.indexer_q.is_identity:
        for layer in model.model.layers:
            compressor = getattr(layer.self_attn, "compressor", None)
            indexer = getattr(compressor, "indexer", None) if compressor is not None else None
            if indexer is not None:
                swapped.append((indexer, indexer.scorer))
                indexer.scorer = QDQIndexerScorer(indexer.scorer, policy.indexer_q)
    try:
        yield model
    finally:
        for indexer, original in swapped:
            indexer.scorer = original
