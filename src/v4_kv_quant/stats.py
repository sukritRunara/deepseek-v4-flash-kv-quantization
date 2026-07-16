"""Pass-through activation-statistics collector for cache writes.

Records, per (layer, state): element count, amax, mean|x|, RMS, per-group amax, and the
RMS quantization error the official QDQ would introduce (FP8 on main-KV nope slices;
rotated FP4 on indexer vectors). Values written to the cache are NEVER modified —
a run with `StatsCollectorCache` is bit-exact vs the stock cache (test-pinned).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)

from .qdq import effective_group_size, fp4_e2m1_qdq, fp8_e4m3_qdq, hadamard_transform


@dataclass
class StateStats:
    elements: int = 0
    amax: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_sq_qdq_err: float = 0.0
    per_group_amax: list[float] = field(default_factory=list)
    scale_group_size: int = 0

    def as_dict(self) -> dict:
        if self.elements == 0:
            return {"elements": 0}
        return {
            "elements": self.elements,
            "amax": self.amax,
            "mean_abs": self.sum_abs / self.elements,
            "rms": (self.sum_sq / self.elements) ** 0.5,
            "rms_qdq_err": (self.sum_sq_qdq_err / self.elements) ** 0.5,
            "per_group_amax": self.per_group_amax,
            "scale_group_size": self.scale_group_size,
        }


class StatsRecorder:
    def __init__(self, group_size_main: int = 64, group_size_indexer: int = 32):
        self.group_size_main = group_size_main
        self.group_size_indexer = group_size_indexer
        self.stats: dict[str, StateStats] = {}

    def _get(self, layer_idx: int, state: str) -> StateStats:
        return self.stats.setdefault(f"layer{layer_idx}/{state}", StateStats())

    def _accumulate(self, entry: StateStats, values: torch.Tensor, qdq_err: torch.Tensor, group: int):
        flat = values.detach().float().reshape(-1, values.shape[-1])
        entry.elements += flat.numel()
        entry.amax = max(entry.amax, flat.abs().max().item())
        entry.sum_abs += flat.abs().sum().item()
        entry.sum_sq += flat.square().sum().item()
        entry.sum_sq_qdq_err += qdq_err.detach().float().square().sum().item()
        group_amax = flat.unflatten(-1, (flat.shape[-1] // group, group)).abs().amax(dim=(0, 2))
        if not entry.per_group_amax:
            entry.per_group_amax = [0.0] * group_amax.numel()
            entry.scale_group_size = group
        entry.per_group_amax = [max(a, b.item()) for a, b in zip(entry.per_group_amax, group_amax)]

    def record_main(self, layer_idx: int, state: str, nope: torch.Tensor) -> None:
        if nope.numel() == 0:
            return
        group = effective_group_size(nope.shape[-1], self.group_size_main)
        err = fp8_e4m3_qdq(nope, group_size=group) - nope
        self._accumulate(self._get(layer_idx, state), nope, err, group)

    def record_indexer(self, layer_idx: int, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        group = effective_group_size(values.shape[-1], self.group_size_indexer)
        rotated = hadamard_transform(values)  # error measured in the stored (rotated) basis
        err = fp4_e2m1_qdq(rotated, group_size=group) - rotated
        self._accumulate(self._get(layer_idx, "indexer_kv"), rotated, err, group)

    def summary(self) -> dict:
        return {key: stats.as_dict() for key, stats in sorted(self.stats.items())}


class _StatsMixin:
    _recorder: StatsRecorder
    _rope_dim: int
    _layer_idx: int

    def update(self, key_states, value_states, *args, **kwargs):
        self._recorder.record_main(self._layer_idx, "window_kv", key_states[..., : -self._rope_dim])
        return super().update(key_states, value_states, *args, **kwargs)

    def update_compressor_states(self, name, compressed):
        if name == "compressor":
            self._recorder.record_main(self._layer_idx, "compressed_kv", compressed[..., : -self._rope_dim])
        elif name == "indexer":
            self._recorder.record_indexer(self._layer_idx, compressed)
        return super().update_compressor_states(name, compressed)


class StatsSlidingWindowLayer(_StatsMixin, DynamicSlidingWindowLayer):
    _layer_type = None

    def __init__(self, config, recorder: StatsRecorder, layer_idx: int):
        super().__init__(sliding_window=config.sliding_window)
        self._recorder, self._rope_dim, self._layer_idx = recorder, config.qk_rope_head_dim, layer_idx


class StatsHCACacheLayer(_StatsMixin, DeepseekV4HCACache):
    _layer_type = None

    def __init__(self, config, recorder: StatsRecorder, layer_idx: int):
        super().__init__(config)
        self._recorder, self._rope_dim, self._layer_idx = recorder, config.qk_rope_head_dim, layer_idx


class StatsCSACacheLayer(_StatsMixin, DeepseekV4CSACache):
    _layer_type = None

    def __init__(self, config, recorder: StatsRecorder, layer_idx: int):
        super().__init__(config)
        self._recorder, self._rope_dim, self._layer_idx = recorder, config.qk_rope_head_dim, layer_idx


_LAYER_CLASSES = {
    "sliding_attention": StatsSlidingWindowLayer,
    "compressed_sparse_attention": StatsCSACacheLayer,
    "heavily_compressed_attention": StatsHCACacheLayer,
}


class StatsCollectorCache(Cache):
    def __init__(self, config, recorder: StatsRecorder):
        layers = [
            _LAYER_CLASSES[layer_type](config, recorder, layer_idx)
            for layer_idx, layer_type in enumerate(config.layer_types)
        ]
        super().__init__(layers=layers)
        self.recorder = recorder
