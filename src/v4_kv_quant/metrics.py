"""Quality metrics for baseline-vs-quantized comparisons (CLAUDE.md metric list)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_comparison_metrics(baseline: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    """Compare two logit tensors of identical shape `[B, T, V]` (teacher-forced histories)."""
    if baseline.shape != test.shape:
        raise ValueError(f"shape mismatch {tuple(baseline.shape)} vs {tuple(test.shape)}")
    base32, test32 = baseline.float(), test.float()
    diff = (base32 - test32).abs()
    log_p = F.log_softmax(base32, dim=-1)
    log_q = F.log_softmax(test32, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # KL(baseline || test) per position
    return {
        "max_abs_logit_err": diff.max().item(),
        "mean_abs_logit_err": diff.mean().item(),
        "rms_logit_err": diff.square().mean().sqrt().item(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "top1_agreement": (base32.argmax(-1) == test32.argmax(-1)).float().mean().item(),
        "nan_count": int(test32.isnan().sum().item()),
        "inf_count": int(test32.isinf().sum().item()),
    }


def next_token_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Mean next-token NLL: `logits[:, t]` predicts `input_ids[:, t+1]`."""
    shifted_logits = logits[:, :-1].float().flatten(0, 1)
    targets = input_ids[:, 1:].flatten()
    return F.cross_entropy(shifted_logits, targets).item()


def indexer_topk_overlap(
    baseline_picks: list[torch.Tensor], test_picks: list[torch.Tensor]
) -> dict[str, float]:
    """Per-position overlap of indexer top-k selections vs baseline.

    Each list entry is a `[B, S_chunk, k]` pick tensor from one indexer forward (k may vary
    per chunk); `-1` marks invalid sentinels. Overlap for one (batch, position) is
    `|valid(base) ∩ valid(test)| / |valid(base)|`; positions with no valid baseline picks
    are skipped. Returns mean/min overlap and the fraction of positions with a perfect match.
    """
    overlaps: list[float] = []
    for base_chunk, test_chunk in zip(baseline_picks, test_picks, strict=True):
        if base_chunk.shape[:2] != test_chunk.shape[:2]:
            raise ValueError("pick chunks misaligned between runs")
        for b in range(base_chunk.shape[0]):
            for s in range(base_chunk.shape[1]):
                base_set = {int(i) for i in base_chunk[b, s].tolist() if i >= 0}
                if not base_set:
                    continue
                test_set = {int(i) for i in test_chunk[b, s].tolist() if i >= 0}
                overlaps.append(len(base_set & test_set) / len(base_set))
    if not overlaps:
        return {"positions": 0, "mean_overlap": 1.0, "min_overlap": 1.0, "exact_match_rate": 1.0}
    return {
        "positions": len(overlaps),
        "mean_overlap": sum(overlaps) / len(overlaps),
        "min_overlap": min(overlaps),
        "exact_match_rate": sum(1.0 for o in overlaps if o == 1.0) / len(overlaps),
    }
