"""Real-corpus calibration data for the full model (D-012).

Port of the reference PoC's C4 streaming pattern (`reference/v2_mla_poc/src/
calibration_data.py`, classified "reuse with modification" in REFERENCE_PORT_MAP.md)
with the required fixes: windows are genuinely NON-OVERLAPPING (the reference's
`random.sample` of starts could overlap), sequences are unpadded (compression
boundaries make left padding unsafe — V4_CACHE_ARCHITECTURE.md §6.2), sources are
mixed by token share, and calibration/held-out sets come from DISJOINT stream regions
by construction. Token ids (plus provenance) are returned for saving alongside results.

Tests inject `texts` so no network access is needed; production streams via
`datasets` (C4-English + a code slice per D-012).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import torch


@dataclass
class CorpusSource:
    """One text source and its share of the total token budget."""

    dataset: str
    config: str | None
    split: str
    text_column: str
    weight: float
    texts: Iterable[str] | None = None  # injected for tests; None -> stream via datasets

    def doc_iter(self) -> Iterator[str]:
        if self.texts is not None:
            yield from self.texts
            return
        from datasets import load_dataset

        ds = load_dataset(self.dataset, self.config, split=self.split, streaming=True)
        for doc in ds:
            yield doc[self.text_column]


# Code slice: codeparrot-clean (ungated, inline content, streams on datasets>=5).
# Rejected: the-stack-smol (gated), smollm-corpus/python-edu (blob pointers only, no
# text). Found the hard way — WORKLOG B4.
DEFAULT_SOURCES = (
    CorpusSource("allenai/c4", "en", "train", "text", 0.8),
    CorpusSource("codeparrot/codeparrot-clean", None, "train", "content", 0.2),
)


@dataclass
class CorpusSamples:
    """Tokenized samples plus everything needed to reproduce them."""

    samples: list[torch.Tensor]  # each [1, seq_len], CPU, long
    provenance: dict = field(default_factory=dict)

    def token_ids(self) -> list[list[int]]:
        return [s[0].tolist() for s in self.samples]


def _fill_buffer(source: CorpusSource, tokenizer, n_tokens: int, skip_tokens: int) -> list[int]:
    """Tokenize the stream until `skip_tokens + n_tokens` are collected; return the
    tail `n_tokens` (the skipped prefix guarantees disjointness between splits)."""
    buf: list[int] = []
    target = skip_tokens + n_tokens
    for text in source.doc_iter():
        if len(buf) >= target:
            break
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
    if len(buf) < target:
        raise RuntimeError(
            f"{source.dataset}: stream exhausted at {len(buf)} tokens "
            f"(need {target}); reduce n_samples/seq_len or check the source"
        )
    return buf[skip_tokens:target]


def build_corpus_samples(
    tokenizer,
    n_samples: int,
    seq_len: int,
    seed: int,
    sources: tuple[CorpusSource, ...] = DEFAULT_SOURCES,
    skip_tokens: int = 0,
) -> CorpusSamples:
    """`n_samples` unpadded [1, seq_len] windows, mixed across sources by weight.

    Windows are cut from contiguous per-source token buffers at stride `seq_len`
    (non-overlapping by construction) and shuffled deterministically by `seed`.
    `skip_tokens` skips that many tokens at the head of EVERY source stream — pass a
    value >= the calibration split's per-source consumption to build a disjoint
    held-out split.
    """
    total_weight = sum(s.weight for s in sources)
    per_source = [max(1, round(n_samples * s.weight / total_weight)) for s in sources]
    # rounding drift -> adjust the largest share
    per_source[per_source.index(max(per_source))] += n_samples - sum(per_source)

    windows: list[torch.Tensor] = []
    consumed: dict[str, int] = {}
    for source, count in zip(sources, per_source):
        need = count * seq_len
        buf = _fill_buffer(source, tokenizer, need, skip_tokens)
        consumed[source.dataset] = skip_tokens + need
        for i in range(count):
            window = buf[i * seq_len : (i + 1) * seq_len]
            windows.append(torch.tensor([window], dtype=torch.long))

    rng = random.Random(seed)
    rng.shuffle(windows)
    return CorpusSamples(
        samples=windows,
        provenance={
            "sources": [
                {"dataset": s.dataset, "config": s.config, "split": s.split,
                 "weight": s.weight, "n_samples": c}
                for s, c in zip(sources, per_source)
            ],
            "n_samples": n_samples,
            "seq_len": seq_len,
            "seed": seed,
            "skip_tokens": skip_tokens,
            "tokens_consumed_per_source": consumed,
        },
    )
