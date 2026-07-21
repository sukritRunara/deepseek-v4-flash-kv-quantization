"""Retrieval eval mechanics (assembly, scoring) + long-context metric upgrades.

The tiny model cannot retrieve anything (random weights); these tests pin the
MECHANICS: position bookkeeping, span extraction, determinism, and equivalence of
the vectorized indexer-overlap with the original per-position set semantics.
"""

from __future__ import annotations

import pytest
import torch

from v4_kv_quant.harness import run_teacher_forced
from v4_kv_quant.metrics import indexer_topk_overlap, logit_comparison_metrics
from v4_kv_quant.retrieval import (
    Needle,
    build_retrieval_sample,
    make_needles_text,
    score_retrieval,
)
from v4_kv_quant.tiny_model import build_tiny_model, deterministic_input_ids


def _needle(name, depth, stmt, cue, value):
    return Needle(name=name, value="v", depth=depth, statement_ids=stmt,
                  cue_ids=cue, value_ids=value)


def test_assembly_positions_and_composition():
    filler = list(range(1000, 1100))  # distinctive ids
    needles = [
        _needle("a", 0.25, [1, 2, 3], [10, 11], [7, 8]),
        _needle("b", 0.75, [4, 5], [12], [9]),
    ]
    sample = build_retrieval_sample(filler, needles, target_len=80)
    ids = sample.input_ids
    assert len(ids) == 80
    a, b = sample.needles
    # statements land where recorded, in depth order
    assert ids[a.statement_start : a.statement_start + 3] == [1, 2, 3]
    assert ids[b.statement_start : b.statement_start + 2] == [4, 5]
    assert a.statement_start < b.statement_start
    # tail: cue+value pairs, spans point exactly at the value ids
    assert ids[a.value_start : a.value_end] == [7, 8]
    assert ids[b.value_start : b.value_end] == [9]
    assert ids[a.value_start - 2 : a.value_start] == [10, 11]
    # queries strictly after every statement
    assert a.value_start > b.statement_start + 2
    # filler content preserved around insertions (no duplication/loss)
    non_filler = set(range(1, 13))
    kept = [t for t in ids if t not in non_filler]
    assert kept == filler[: len(kept)]


def test_assembly_refuses_short_filler():
    with pytest.raises(ValueError):
        build_retrieval_sample(list(range(10)), [_needle("a", 0.5, [1], [2], [3])],
                               target_len=100)


def test_scoring_span_math():
    filler = list(range(1000, 1050))
    needles = [_needle("a", 0.5, [1, 2], [10], [7, 8])]
    sample = build_retrieval_sample(filler, needles, target_len=40)
    n = sample.needles[0]
    vocab = 2000
    logits = torch.full((1, 40, vocab), -10.0)
    # perfect prediction: logits at value_start-1 predicts first value token, etc.
    logits[0, n.value_start - 1, 7] = 10.0
    logits[0, n.value_start, 8] = 10.0
    res = score_retrieval(logits, sample)
    assert res["token_acc"] == 1.0 and res["exact_rate"] == 1.0
    # break the second value token -> half accuracy, no exact match
    logits[0, n.value_start, 8] = -10.0
    logits[0, n.value_start, 9] = 10.0
    res = score_retrieval(logits, sample)
    assert res["token_acc"] == pytest.approx(0.5)
    assert res["exact_rate"] == 0.0


class FakeTok:
    """Whitespace tokenizer with stable word ids (prefix-stable by construction)."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 5) for w in text.split()]


def test_make_needles_text_deterministic_and_consistent():
    tok = FakeTok()
    a = make_needles_text(tok, 4, seed=7)
    tok2 = FakeTok()
    b = make_needles_text(tok2, 4, seed=7)
    assert [n.name for n in a] == [n.name for n in b]
    assert [n.value for n in a] == [n.value for n in b]
    assert len({n.name for n in a}) == 4
    for n in a:
        assert n.value_ids, "value must contribute at least one token"
        assert 0.0 < n.depth < 1.0


def test_vectorized_overlap_matches_set_semantics():
    torch.manual_seed(0)
    base = torch.randint(0, 50, (3, 7, 6))
    test = torch.randint(0, 50, (3, 7, 6))
    # ensure per-row uniqueness (top-k picks are unique); inject sentinels
    base = base.argsort(-1)  # permutations of 0..5 => unique
    test = test.argsort(-1)
    base[0, 0] = -1  # no valid base picks -> position skipped
    test[1, 2, :3] = -1
    got = indexer_topk_overlap([base], [test])
    # reference: original per-position set implementation
    overlaps = []
    for b in range(3):
        for s in range(7):
            bs = {int(i) for i in base[b, s].tolist() if i >= 0}
            if not bs:
                continue
            ts = {int(i) for i in test[b, s].tolist() if i >= 0}
            overlaps.append(len(bs & ts) / len(bs))
    assert got["positions"] == len(overlaps)
    assert got["mean_overlap"] == pytest.approx(sum(overlaps) / len(overlaps))
    assert got["min_overlap"] == pytest.approx(min(overlaps))
    assert got["exact_match_rate"] == pytest.approx(
        sum(1.0 for o in overlaps if o == 1.0) / len(overlaps))


def test_harness_logits_to_cpu_matches_resident():
    model = build_tiny_model(seed=0, index_topk=64)
    ids = deterministic_input_ids(1, 24)
    resident = run_teacher_forced(model, ids, prefill_len=16, prefill_chunk=8)
    offload = run_teacher_forced(model, ids, prefill_len=16, prefill_chunk=8,
                                 logits_to_cpu=True)
    assert offload.logits.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in offload.indexer_picks)
    torch.testing.assert_close(offload.logits, resident.logits.cpu())
    m = logit_comparison_metrics(resident.logits, offload.logits.to(resident.logits.device))
    assert m["max_abs_logit_err"] == 0.0
