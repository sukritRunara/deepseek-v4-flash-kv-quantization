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
    capture_indexer: bool = True,
) -> TeacherForcedResult:
    """One-shot prefill of `prefill_len` tokens, then ground-truth single-token decode.

    `policy=None` runs the stock `DynamicCache` baseline; otherwise a `QDQCache` with the
    policy plus the indexer-query QDQ wrapper (active only if the policy asks for it).
    """
    from transformers import DynamicCache  # local import keeps module import light

    seq_len = input_ids.shape[1]
    if not 0 < prefill_len <= seq_len:
        raise ValueError(f"prefill_len {prefill_len} out of range for seq_len {seq_len}")

    result = TeacherForcedResult(logits=torch.empty(0))
    hooks = []
    if capture_indexer:
        for indexer in _find_indexers(model):
            hooks.append(
                indexer.register_forward_hook(
                    lambda mod, args, out: result.indexer_picks.append(out.detach().clone())
                )
            )

    cache = DynamicCache(config=model.config) if policy is None else build_qdq_cache(model.config, policy)
    context = nullcontext(model) if policy is None else indexer_query_qdq(model, policy)
    chunks: list[torch.Tensor] = []
    try:
        with context:
            out = model(input_ids[:, :prefill_len], past_key_values=cache, use_cache=True)
            chunks.append(out.logits)
            for t in range(prefill_len, seq_len):
                out = model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
                chunks.append(out.logits)
    finally:
        for hook in hooks:
            hook.remove()

    result.logits = torch.cat(chunks, dim=1)
    assert result.logits.shape[1] == seq_len
    return result
