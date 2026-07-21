"""Teacher-forced comparison harness (tiny model, Stage B).

Baseline and quantized runs consume IDENTICAL token histories (ground-truth ids fed at
every decode step), so logit differences measure only the cache policy's numerical effect
— the methodology mandated by CLAUDE.md and inherited from the reference PoC's
`collect_logits` (reference/v2_mla_poc/tests/verify_quantized_kv.py:83).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .policy import KVQuantPolicy
from .qdq_cache import build_qdq_cache, indexer_query_qdq


@dataclass
class TeacherForcedResult:
    logits: torch.Tensor  # [B, S, V] — positions 0..S-1, identical histories across runs
    indexer_picks: list[torch.Tensor] = field(default_factory=list)  # per indexer forward


def _find_indexers(model) -> list:
    indexers = []
    for layer in model.model.layers:
        compressor = getattr(layer.self_attn, "compressor", None)
        indexer = getattr(compressor, "indexer", None) if compressor is not None else None
        if indexer is not None:
            indexers.append(indexer)
    return indexers


@torch.no_grad()
def run_teacher_forced(
    model,
    input_ids: torch.Tensor,
    prefill_len: int,
    policy: KVQuantPolicy | None = None,
    precision_map=None,
    capture_indexer: bool = True,
    storage: bool = False,
    prefill_chunk: int | None = None,
    logits_to_cpu: bool = False,
) -> TeacherForcedResult:
    """Prefill of `prefill_len` tokens, then ground-truth single-token decode.

    Cache selection (mutually exclusive):
      * default            -> stock `DynamicCache` baseline;
      * `policy=...`       -> `QDQCache` (Task-02 QDQ simulation), or the Stage-C
                              `QuantizedStorageCache` (actual storage) when `storage=True`;
      * `precision_map=...`-> `MappedQDQCache` (per-group map, Task-03 calibration).
    Either quantized mode also arms the symmetric indexer-query QDQ wrapper when the
    policy/map quantizes indexer keys.

    `prefill_chunk` feeds the prefill in slices through the same cache (semantics
    test-pinned chunked==one-shot; needed at 8k+ where one-shot eager attention OOMs a
    96 GB card — WORKLOG 2026-07-20 B4). Comparisons must use the SAME chunking on
    both sides: chunk-shaped kernels differ from one-shot at the last ulp, which flips
    near-tied selective-indexer picks (D-004).

    `logits_to_cpu` streams each chunk's logits (and captured indexer picks) to host
    memory as produced: a full 65k-run's logits are ~17 GiB — resident-on-GPU they OOM
    next to the weights. Pair with `compute_device="cuda"` in the metrics calls.
    """
    from transformers import DynamicCache  # local import keeps module import light

    if policy is not None and precision_map is not None:
        raise ValueError("pass either policy or precision_map, not both")
    seq_len = input_ids.shape[1]
    if not 0 < prefill_len <= seq_len:
        raise ValueError(f"prefill_len {prefill_len} out of range for seq_len {seq_len}")

    result = TeacherForcedResult(logits=torch.empty(0))
    hooks = []
    if capture_indexer:
        def _capture(mod, args, out):
            picks = out.detach()
            result.indexer_picks.append(picks.cpu() if logits_to_cpu else picks.clone())

        for indexer in _find_indexers(model):
            hooks.append(indexer.register_forward_hook(_capture))

    if policy is not None:
        if storage:
            from .storage_cache import QuantizedStorageCache

            cache = QuantizedStorageCache(model.config, policy)
        else:
            cache = build_qdq_cache(model.config, policy)
        context = indexer_query_qdq(model, policy)
    elif precision_map is not None:
        from .mapped_cache import indexer_query_context

        if storage:
            from .mapped_storage_cache import MappedStorageCache

            cache = MappedStorageCache(model.config, precision_map)
        else:
            from .mapped_cache import MappedQDQCache

            cache = MappedQDQCache(model.config, precision_map)
        context = indexer_query_context(model, precision_map)
    else:
        cache = DynamicCache(config=model.config)
        context = nullcontext(model)
    def _keep(logits: torch.Tensor) -> torch.Tensor:
        return logits.cpu() if logits_to_cpu else logits

    chunks: list[torch.Tensor] = []
    try:
        with context:
            if prefill_chunk is not None and prefill_len > prefill_chunk:
                for start in range(0, prefill_len, prefill_chunk):
                    out = model(input_ids[:, start : min(start + prefill_chunk, prefill_len)],
                                past_key_values=cache, use_cache=True)
                    chunks.append(_keep(out.logits))
            else:
                out = model(input_ids[:, :prefill_len], past_key_values=cache, use_cache=True)
                chunks.append(_keep(out.logits))
            for t in range(prefill_len, seq_len):
                out = model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
                chunks.append(_keep(out.logits))
    finally:
        for hook in hooks:
            hook.remove()

    result.logits = torch.cat(chunks, dim=1)
    assert result.logits.shape[1] == seq_len
    return result
