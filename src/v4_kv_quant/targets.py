"""Quantization target taxonomy: (layer_idx, layer_type, state, contiguous channel group).

States per layer type (docs/V4_CACHE_ARCHITECTURE.md §2):
  sliding_attention              -> window_kv
  compressed_sparse_attention    -> window_kv, compressed_kv, indexer_kv
  heavily_compressed_attention   -> window_kv, compressed_kv

Group semantics:
  * window_kv / compressed_kv: contiguous channel groups within the NON-RoPE slice
    (channel indices are relative to the nope slice; the trailing RoPE slice is never
    a target). Production group size 64 (official scale groups); tiny models fall back
    to the whole nope width via `effective_group_size`.
  * indexer_kv: exactly ONE state-level target covering the full vector. The official
    FP4 path Hadamard-rotates the whole vector first, so channel groups in the original
    basis are not independently quantizable — per-group indexer calibration would not
    correspond to any deployable storage decision.

Compressor buffers / overlap / gates are deliberately NOT targets (injection plan §5).
"""

from __future__ import annotations

from dataclasses import dataclass

from .qdq import effective_group_size

MAIN_STATES = ("window_kv", "compressed_kv")
INDEXER_STATE = "indexer_kv"

STATES_BY_LAYER_TYPE = {
    "sliding_attention": ("window_kv",),
    "compressed_sparse_attention": ("window_kv", "compressed_kv", "indexer_kv"),
    "heavily_compressed_attention": ("window_kv", "compressed_kv"),
}


@dataclass(frozen=True)
class QuantTarget:
    """One independently quantizable unit of cache state."""

    layer_idx: int
    layer_type: str
    state: str  # window_kv | compressed_kv | indexer_kv
    group_index: int
    start: int  # channel range; relative to the nope slice for main states,
    end: int    # absolute within the full vector for indexer_kv

    @property
    def key(self) -> str:
        return f"layer{self.layer_idx}/{self.state}/g{self.group_index}"


def nope_width(config) -> int:
    return config.head_dim - config.qk_rope_head_dim


def enumerate_targets(
    config,
    group_size_main: int = 64,
    layer_indices: list[int] | None = None,
) -> list[QuantTarget]:
    """All quantizable targets for a config, in deterministic order.

    `group_size_main` is the requested contiguous group width for main-KV states
    (production: 64); `effective_group_size` handles tiny-model widths. Indexer states
    yield one whole-vector target each.
    """
    width = nope_width(config)
    group = effective_group_size(width, group_size_main)
    targets: list[QuantTarget] = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        for state in STATES_BY_LAYER_TYPE[layer_type]:
            if state == INDEXER_STATE:
                targets.append(
                    QuantTarget(layer_idx, layer_type, state, 0, 0, config.index_head_dim)
                )
                continue
            for g in range(width // group):
                targets.append(
                    QuantTarget(layer_idx, layer_type, state, g, g * group, (g + 1) * group)
                )
    return targets
