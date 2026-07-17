"""Corpus loader tests (D-012). No network: sources inject synthetic texts."""

from __future__ import annotations

import pytest
import torch

from v4_kv_quant.calibration_data import CorpusSource, build_corpus_samples


class FakeTokenizer:
    """Maps each word 'w<i>' to id i (stable, padding-free)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(w[1:]) for w in text.split()]


def _texts(start: int, n_words: int, chunk: int = 97):
    words = [f"w{i}" for i in range(start, start + n_words)]
    return [" ".join(words[i : i + chunk]) for i in range(0, n_words, chunk)]


def _sources(weight_a=0.8, weight_b=0.2):
    return (
        CorpusSource("srcA", None, "train", "text", weight_a, texts=_texts(0, 40_000)),
        CorpusSource("srcB", None, "train", "text", weight_b, texts=_texts(100_000, 40_000)),
    )


def test_windows_exact_nonoverlapping_and_mixed():
    out = build_corpus_samples(FakeTokenizer(), n_samples=10, seq_len=64, seed=3, sources=_sources())
    assert len(out.samples) == 10
    assert all(s.shape == (1, 64) and s.dtype == torch.long for s in out.samples)
    # non-overlap within each source: all ids across windows are unique
    all_ids = torch.cat(out.samples, dim=1).flatten().tolist()
    assert len(all_ids) == len(set(all_ids))
    # both sources present at ~weights (8 vs 2 windows -> ids < 100k and >= 100k)
    n_b = sum(1 for s in out.samples if s[0, 0].item() >= 100_000)
    assert n_b == 2
    # contiguity inside each window (FakeTokenizer ids are sequential per source)
    for s in out.samples:
        ids = s[0].tolist()
        assert ids == list(range(ids[0], ids[0] + 64))


def test_deterministic_and_seed_sensitive():
    a = build_corpus_samples(FakeTokenizer(), 6, 32, seed=1, sources=_sources())
    b = build_corpus_samples(FakeTokenizer(), 6, 32, seed=1, sources=_sources())
    c = build_corpus_samples(FakeTokenizer(), 6, 32, seed=2, sources=_sources())
    assert all(torch.equal(x, y) for x, y in zip(a.samples, b.samples))
    assert any(not torch.equal(x, y) for x, y in zip(a.samples, c.samples))


def test_skip_tokens_gives_disjoint_splits():
    calib = build_corpus_samples(FakeTokenizer(), 8, 32, seed=1, sources=_sources())
    max_used = max(calib.provenance["tokens_consumed_per_source"].values())
    held = build_corpus_samples(FakeTokenizer(), 4, 32, seed=2, sources=_sources(),
                                skip_tokens=max_used)
    calib_ids = set(torch.cat(calib.samples, dim=1).flatten().tolist())
    held_ids = set(torch.cat(held.samples, dim=1).flatten().tolist())
    assert not calib_ids & held_ids
    assert held.provenance["skip_tokens"] == max_used


def test_stream_exhaustion_is_loud():
    tiny = (CorpusSource("srcA", None, "train", "text", 1.0, texts=_texts(0, 100)),)
    with pytest.raises(RuntimeError, match="stream exhausted"):
        build_corpus_samples(FakeTokenizer(), 10, 64, seed=0, sources=tiny)
