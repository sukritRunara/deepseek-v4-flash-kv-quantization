"""Empirical one-target perturbation sensitivity and precision-map construction.

Primary sensitivity truth (CLAUDE.md): quantize ONE target (a single-entry precision
map), run teacher-forced against the BF16 baseline on the SAME token history, and
measure downstream damage — ΔNLL and logit KL for every target, plus top-k overlap
for indexer targets. Gradient-weighted ranking is deliberately deferred (optional per
CLAUDE.md; it must be validated against this empirical truth before use).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .harness import TeacherForcedResult, run_teacher_forced
from .metrics import indexer_topk_overlap, logit_comparison_metrics, next_token_nll
from .precision_map import MapEntry, PrecisionMap
from .targets import INDEXER_STATE, QuantTarget

DEFAULT_MAIN_KIND = "fp8_e4m3"
DEFAULT_INDEXER_KIND = "fp4_e2m1_hadamard"


def default_kind_for(target: QuantTarget) -> str:
    return DEFAULT_INDEXER_KIND if target.state == INDEXER_STATE else DEFAULT_MAIN_KIND


def perturbation_map(target: QuantTarget, kind: str | None = None, indexer_scale_group: int = 32) -> PrecisionMap:
    """Single-entry map = the one-group perturbation experiment for `target`."""
    kind = kind or default_kind_for(target)
    scale_group = indexer_scale_group if target.state == INDEXER_STATE else 0
    return PrecisionMap(
        name=f"perturb-{target.key}-{kind}",
        entries=[MapEntry.for_target(target, kind, scale_group_size=scale_group)],
    )


@dataclass
class SensitivityRecord:
    target: QuantTarget
    kind: str
    nll_delta: float
    kl_mean: float
    max_abs_logit_err: float
    top1_agreement: float
    indexer_overlap: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Ranking score (higher = more sensitive). KL is the primary signal (always
        >= 0); |ΔNLL| breaks ties — both are CLAUDE.md primary metrics."""
        return self.kl_mean + abs(self.nll_delta)

    def as_dict(self) -> dict:
        return {
            "target": {
                "layer_idx": self.target.layer_idx,
                "layer_type": self.target.layer_type,
                "state": self.target.state,
                "group_index": self.target.group_index,
                "start": self.target.start,
                "end": self.target.end,
            },
            "kind": self.kind,
            "nll_delta": self.nll_delta,
            "kl_mean": self.kl_mean,
            "max_abs_logit_err": self.max_abs_logit_err,
            "top1_agreement": self.top1_agreement,
            "score": self.score,
            "indexer_overlap": self.indexer_overlap,
            **self.extra,
        }


@torch.no_grad()
def measure_target(
    model,
    input_ids: torch.Tensor,
    prefill_len: int,
    target: QuantTarget,
    baseline: TeacherForcedResult,
    baseline_nll: float,
    kind: str | None = None,
    indexer_scale_group: int = 32,
) -> SensitivityRecord:
    kind = kind or default_kind_for(target)
    run = run_teacher_forced(
        model,
        input_ids,
        prefill_len,
        precision_map=perturbation_map(target, kind, indexer_scale_group),
        capture_indexer=target.state == INDEXER_STATE,
    )
    metrics = logit_comparison_metrics(baseline.logits, run.logits)
    overlap = None
    if target.state == INDEXER_STATE:
        overlap = indexer_topk_overlap(baseline.indexer_picks, run.indexer_picks)
    return SensitivityRecord(
        target=target,
        kind=kind,
        nll_delta=next_token_nll(run.logits, input_ids) - baseline_nll,
        kl_mean=metrics["kl_mean"],
        max_abs_logit_err=metrics["max_abs_logit_err"],
        top1_agreement=metrics["top1_agreement"],
        indexer_overlap=overlap,
        extra={"nan_count": metrics["nan_count"], "inf_count": metrics["inf_count"]},
    )


@torch.no_grad()
def run_sensitivity_sweep(
    model,
    input_ids: torch.Tensor,
    prefill_len: int,
    targets: list[QuantTarget],
    kind_main: str = DEFAULT_MAIN_KIND,
    kind_indexer: str = DEFAULT_INDEXER_KIND,
    indexer_scale_group: int = 32,
    progress: bool = False,
) -> tuple[TeacherForcedResult, float, list[SensitivityRecord]]:
    """Measure every target against one shared baseline run. Deterministic."""
    baseline = run_teacher_forced(model, input_ids, prefill_len, capture_indexer=True)
    baseline_nll = next_token_nll(baseline.logits, input_ids)
    records = []
    for i, target in enumerate(targets):
        kind = kind_indexer if target.state == INDEXER_STATE else kind_main
        record = measure_target(
            model, input_ids, prefill_len, target, baseline, baseline_nll, kind, indexer_scale_group
        )
        records.append(record)
        if progress:
            print(f"  [{i + 1:>3}/{len(targets)}] {target.key:<28} kind={kind:<18} "
                  f"score={record.score:.3e} dNLL={record.nll_delta:+.3e} KL={record.kl_mean:.3e}")
    return baseline, baseline_nll, records


def build_map_from_sweep(
    records: list[SensitivityRecord],
    name: str,
    fp8_fraction: float = 0.75,
    fp4_fraction: float = 0.0,
    indexer_min_overlap: float = 0.9,
    provenance: dict | None = None,
) -> PrecisionMap:
    """Assign precisions from sensitivity ranking (least sensitive quantized hardest).

    Main-KV targets are sorted ascending by `score`; the least-sensitive
    `fp4_fraction` get FP4, the next `fp8_fraction` get FP8, the rest stay BF16
    (mirrors the reference PoC's `make_mixed_precision_config`, re-keyed to the V4
    taxonomy). Indexer targets get the official Hadamard-FP4 path iff their measured
    top-k overlap stays >= `indexer_min_overlap`.
    """
    if not 0.0 <= fp4_fraction + fp8_fraction <= 1.0:
        raise ValueError("fp4_fraction + fp8_fraction must be in [0, 1]")
    entries: list[MapEntry] = []
    main = sorted((r for r in records if r.target.state != INDEXER_STATE), key=lambda r: r.score)
    n_fp4 = int(len(main) * fp4_fraction)
    n_fp8 = int(len(main) * fp8_fraction)
    for i, record in enumerate(main):
        if i < n_fp4:
            kind = "fp4_e2m1"
        elif i < n_fp4 + n_fp8:
            kind = "fp8_e4m3"
        else:
            continue  # most sensitive stay BF16 (no entry)
        entries.append(MapEntry.for_target(record.target, kind))
    for record in (r for r in records if r.target.state == INDEXER_STATE):
        overlap = (record.indexer_overlap or {}).get("mean_overlap", 0.0)
        if overlap >= indexer_min_overlap:
            entries.append(
                MapEntry.for_target(record.target, DEFAULT_INDEXER_KIND, scale_group_size=32)
            )
    return PrecisionMap(
        name=name,
        entries=entries,
        provenance={
            "method": "one-target empirical perturbation (teacher-forced)",
            "ranking_score": "kl_mean + |nll_delta|",
            "fp8_fraction": fp8_fraction,
            "fp4_fraction": fp4_fraction,
            "indexer_min_overlap": indexer_min_overlap,
            "n_main_targets": len(main),
            **(provenance or {}),
        },
    )
