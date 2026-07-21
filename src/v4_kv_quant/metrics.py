"""Quality metrics for baseline-vs-quantized comparisons (CLAUDE.md metric list)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_comparison_metrics(
    baseline: torch.Tensor, test: torch.Tensor, position_chunk: int = 1024,
    compute_device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compare two logit tensors of identical shape `[B, T, V]` (teacher-forced histories).

    Computed in `position_chunk`-sized slices along the flattened position axis:
    softmax/argmax are per-position, so slicing is exact — only summation order changes
    (fp64 accumulators). Materializing full-vocab fp32 log-softmax for a whole 8k+
    sequence costs ~8.5 GiB per tensor and OOMs GPU 0 next to the resident weights
    (WORKLOG 2026-07-20 B4 overnight); 32k would need ~17 GiB per tensor.

    `compute_device` moves each chunk there for the math — pass "cuda" when the
    logits live on CPU (65k-eval offload path) to keep the math fast while only one
    chunk at a time occupies GPU memory.
    """
    if baseline.shape != test.shape:
        raise ValueError(f"shape mismatch {tuple(baseline.shape)} vs {tuple(test.shape)}")
    base_flat = baseline.flatten(0, -2)
    test_flat = test.flatten(0, -2)
    n_pos, n_elem = base_flat.shape[0], base_flat.numel()
    dev = torch.device(compute_device) if compute_device is not None else baseline.device
    abs_sum = torch.zeros((), dtype=torch.float64, device=dev)
    sq_sum = torch.zeros((), dtype=torch.float64, device=dev)
    kl_sum = torch.zeros((), dtype=torch.float64, device=dev)
    top1_hits = torch.zeros((), dtype=torch.float64, device=dev)
    abs_max = torch.zeros((), dtype=torch.float32, device=dev)
    kl_max = torch.full((), -float("inf"), dtype=torch.float32, device=dev)
    nan_count = torch.zeros((), dtype=torch.int64, device=dev)
    inf_count = torch.zeros((), dtype=torch.int64, device=dev)
    for start in range(0, n_pos, position_chunk):
        b32 = base_flat[start : start + position_chunk].to(dev).float()
        t32 = test_flat[start : start + position_chunk].to(dev).float()
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
    logits: torch.Tensor, input_ids: torch.Tensor, position_chunk: int = 1024,
    compute_device: str | torch.device | None = None,
) -> float:
    """Mean next-token NLL: `logits[:, t]` predicts `input_ids[:, t+1]`.

    Chunked along positions for the same memory reason as `logit_comparison_metrics`
    (cross-entropy is per-token; sum/N is exactly the previous mean reduction).
    """
    dev = torch.device(compute_device) if compute_device is not None else logits.device
    shifted_logits = logits[:, :-1].flatten(0, 1)
    targets = input_ids[:, 1:].flatten()
    n = targets.shape[0]
    total = torch.zeros((), dtype=torch.float64, device=dev)
    for start in range(0, n, position_chunk):
        total += F.cross_entropy(
            shifted_logits[start : start + position_chunk].to(dev).float(),
            targets[start : start + position_chunk].to(dev),
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

    Vectorized (sort + batched searchsorted): the original per-position Python-set loop
    is O(positions) interpreter work — ~1.4M set intersections for one 65k run, unusably
    slow. Rows are unique-valued (top-k indices), so counting sorted-search hits equals
    set intersection.
    """
    chunks: list[torch.Tensor] = []
    for base_chunk, test_chunk in zip(baseline_picks, test_picks, strict=True):
        if base_chunk.shape[:2] != test_chunk.shape[:2]:
            raise ValueError("pick chunks misaligned between runs")
        b = base_chunk.flatten(0, 1).long()  # [N, k_b]
        t = test_chunk.flatten(0, 1).long()  # [N, k_t]
        big = torch.iinfo(torch.long).max
        n_base = (b >= 0).sum(-1)  # valid base picks per position
        b_sorted, _ = torch.where(b >= 0, b, torch.full_like(b, big)).sort(dim=-1)
        # invalid test entries -> big-1: never equals a real pick or the `big` sentinel
        t_query = torch.where(t >= 0, t, torch.full_like(t, big - 1))
        idx = torch.searchsorted(b_sorted, t_query).clamp(max=b_sorted.shape[-1] - 1)
        matches = (b_sorted.gather(-1, idx) == t_query) & (t >= 0)
        keep = n_base > 0
        if keep.any():
            chunks.append(matches.sum(-1)[keep].double() / n_base[keep].double())
    if not chunks:
        return {"positions": 0, "mean_overlap": 1.0, "min_overlap": 1.0, "exact_match_rate": 1.0}
    overlaps = torch.cat([c.cpu() for c in chunks])
    return {
        "positions": int(overlaps.numel()),
        "mean_overlap": overlaps.mean().item(),
        "min_overlap": overlaps.min().item(),
        "exact_match_rate": (overlaps == 1.0).double().mean().item(),
    }
