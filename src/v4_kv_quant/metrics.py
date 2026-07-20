"""Quality metrics for baseline-vs-quantized comparisons (CLAUDE.md metric list)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_comparison_metrics(
    baseline: torch.Tensor, test: torch.Tensor, position_chunk: int = 1024
) -> dict[str, float]:
    """Compare two logit tensors of identical shape `[B, T, V]` (teacher-forced histories).

    Computed in `position_chunk`-sized slices along the flattened position axis:
    softmax/argmax are per-position, so slicing is exact — only summation order changes
    (fp64 accumulators). Materializing full-vocab fp32 log-softmax for a whole 8k+
    sequence costs ~8.5 GiB per tensor and OOMs GPU 0 next to the resident weights
    (WORKLOG 2026-07-20 B4 overnight); 32k would need ~17 GiB per tensor.
    """
    if baseline.shape != test.shape:
        raise ValueError(f"shape mismatch {tuple(baseline.shape)} vs {tuple(test.shape)}")
    base_flat = baseline.flatten(0, -2)
    test_flat = test.flatten(0, -2)
    n_pos, n_elem = base_flat.shape[0], base_flat.numel()
    dev = baseline.device
    abs_sum = torch.zeros((), dtype=torch.float64, device=dev)
    sq_sum = torch.zeros((), dtype=torch.float64, device=dev)
    kl_sum = torch.zeros((), dtype=torch.float64, device=dev)
    top1_hits = torch.zeros((), dtype=torch.float64, device=dev)
    abs_max = torch.zeros((), dtype=torch.float32, device=dev)
    kl_max = torch.full((), -float("inf"), dtype=torch.float32, device=dev)
    nan_count = torch.zeros((), dtype=torch.int64, device=dev)
    inf_count = torch.zeros((), dtype=torch.int64, device=dev)
    for start in range(0, n_pos, position_chunk):
        b32 = base_flat[start : start + position_chunk].float()
        t32 = test_flat[start : start + position_chunk].float()
        diff = (b32 - t32).abs()
        abs_sum += diff.sum(dtype=torch.float64)
        sq_sum += diff.square().sum(dtype=torch.float64)
        abs_max = torch.maximum(abs_max, diff.max())
        log_p = F.log_softmax(b32, dim=-1)
        log_q = F.log_softmax(t32, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # KL(baseline || test) per position
        kl_sum += kl.sum(dtype=torch.float64)
        kl_max = torch.maximum(kl_max, kl.max())
        top1_hits += (b32.argmax(-1) == t32.argmax(-1)).sum(dtype=torch.float64)
        nan_count += t32.isnan().sum()
        inf_count += t32.isinf().sum()
    return {
        "max_abs_logit_err": abs_max.item(),
        "mean_abs_logit_err": (abs_sum / n_elem).item(),
        "rms_logit_err": (sq_sum / n_elem).sqrt().item(),
        "kl_mean": (kl_sum / n_pos).item(),
        "kl_max": kl_max.item(),
        "top1_agreement": (top1_hits / n_pos).item(),
        "nan_count": int(nan_count.item()),
        "inf_count": int(inf_count.item()),
    }


def next_token_nll(
    logits: torch.Tensor, input_ids: torch.Tensor, position_chunk: int = 1024
) -> float:
    """Mean next-token NLL: `logits[:, t]` predicts `input_ids[:, t+1]`.

    Chunked along positions for the same memory reason as `logit_comparison_metrics`
    (cross-entropy is per-token; sum/N is exactly the previous mean reduction).
    """
    shifted_logits = logits[:, :-1].flatten(0, 1)
    targets = input_ids[:, 1:].flatten().to(logits.device)
    n = targets.shape[0]
    total = torch.zeros((), dtype=torch.float64, device=logits.device)
    for start in range(0, n, position_chunk):
        total += F.cross_entropy(
            shifted_logits[start : start + position_chunk].float(),
            targets[start : start + position_chunk],
            reduction="sum",
        ).to(torch.float64)
    return (total / n).item()


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
