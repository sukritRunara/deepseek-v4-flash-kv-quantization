"""Long-context retrieval eval: needles planted in filler text, teacher-forced scoring.

Why this exists (FUTURE_WORK 2026-07-20): perplexity on packed web text mostly tests
short-range prediction, but KV-cache damage expresses itself as *selection drift*
(near-tied indexer picks flipping) whose real-world casualty is long-range recall.
This module scores exactly that: plant `name -> value` facts at controlled depths in
a long context, re-ask for every value at the very end, and measure whether the
model reproduces the value tokens (teacher-forced, so baseline and quantized
variants are scored on IDENTICAL token sequences — no generation stochasticity).

The core is tokenizer-agnostic (operates on token ids); `make_needles_text` is the
thin text-side helper for real tokenizers. Base-model friendly: scoring is argmax
accuracy + NLL on the value-token spans, not instruction following.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch


@dataclass
class Needle:
    """One planted fact and where it ended up in the assembled sample."""

    name: str
    value: str
    depth: float                      # requested fractional depth in the filler body
    statement_ids: list[int]          # full sentence planted at depth
    cue_ids: list[int]                # tail query prefix ("... the code for X is")
    value_ids: list[int]              # ground-truth continuation being scored
    statement_start: int = -1         # filled by assembly (absolute position)
    value_start: int = -1             # filled by assembly: first value token in tail
    value_end: int = -1
    # v2: extra statements owned by this needle (e.g. the ORIGINAL value before a
    # correction), each {"depth": float, "ids": [...]}. The scored value_ids always
    # correspond to the FINAL fact; extras exist to interfere.
    extra_statements: list = field(default_factory=list)
    kind: str = "plain"               # plain | updated (v2 bookkeeping)


@dataclass
class RetrievalSample:
    input_ids: list[int]
    needles: list[Needle]
    provenance: dict = field(default_factory=dict)


def build_retrieval_sample(
    filler_ids: list[int], needles: list[Needle], target_len: int
) -> RetrievalSample:
    """Assemble filler + planted statements + tail queries into one sequence.

    Layout: [filler | stmt_1 | filler | ... | stmt_n | filler | cue_1 value_1 | ...].
    Statements are inserted at their fractional depths of the filler body; all
    queries go at the very end so every retrieval must span the distance back to
    its statement. Raises if the filler cannot fill the body budget.
    """
    needles = sorted(needles, key=lambda n: n.depth)
    # every insertion: (depth, ids, needle-or-None); v2 extras interleave with primaries
    inserts: list[tuple[float, list[int], Needle | None]] = []
    for n in needles:
        inserts.append((n.depth, n.statement_ids, n))
        for ex in n.extra_statements:
            inserts.append((float(ex["depth"]), list(ex["ids"]), None))
    inserts.sort(key=lambda t: t[0])
    tail_len = sum(len(n.cue_ids) + len(n.value_ids) for n in needles)
    stmt_len = sum(len(ids) for _, ids, _ in inserts)
    body_budget = target_len - tail_len
    filler_budget = body_budget - stmt_len
    if filler_budget <= 0:
        raise ValueError(f"target_len {target_len} too small for needles+queries")
    if len(filler_ids) < filler_budget:
        raise ValueError(f"filler too short: {len(filler_ids)} < {filler_budget}")

    out: list[int] = []
    cursor = 0
    for depth, ids, owner in inserts:
        insert_at = int(depth * filler_budget)
        insert_at = max(cursor, min(insert_at, filler_budget))
        out.extend(filler_ids[cursor:insert_at])
        if owner is not None:
            owner.statement_start = len(out)
        out.extend(ids)
        cursor = insert_at
    out.extend(filler_ids[cursor:filler_budget])
    assert len(out) == body_budget
    for needle in needles:
        out.extend(needle.cue_ids)
        needle.value_start = len(out)
        out.extend(needle.value_ids)
        needle.value_end = len(out)
    assert len(out) == target_len
    return RetrievalSample(input_ids=out, needles=needles)


@torch.no_grad()
def score_retrieval(logits: torch.Tensor, sample: RetrievalSample) -> dict:
    """Score one variant's logits `[1, T, V]` against the sample's value spans.

    Position t's logits predict token t+1, so span [s, e) is predicted by
    logits[s-1 : e-1]. Returns per-needle records plus aggregates.
    """
    if logits.shape[0] != 1 or logits.shape[1] != len(sample.input_ids):
        raise ValueError(f"logits shape {tuple(logits.shape)} vs sample length "
                         f"{len(sample.input_ids)}")
    per_needle = []
    for n in sample.needles:
        span_logits = logits[0, n.value_start - 1 : n.value_end - 1].float()
        targets = torch.tensor(n.value_ids, dtype=torch.long, device=span_logits.device)
        pred = span_logits.argmax(-1)
        correct = (pred == targets)
        nll = torch.nn.functional.cross_entropy(span_logits, targets).item()
        per_needle.append({
            "name": n.name, "depth": n.depth,
            "token_acc": correct.float().mean().item(),
            "exact": bool(correct.all().item()),
            "nll": nll, "n_tokens": len(n.value_ids),
        })
    n_needles = len(per_needle)
    return {
        "needles": per_needle,
        "token_acc": sum(r["token_acc"] for r in per_needle) / n_needles,
        "exact_rate": sum(1.0 for r in per_needle if r["exact"]) / n_needles,
        "nll_mean": sum(r["nll"] for r in per_needle) / n_needles,
    }


_WORDS = ("kestrel", "quartz", "bramble", "cobalt", "falcon", "juniper", "marble",
          "nectar", "obsidian", "pluto", "saffron", "tundra", "velvet", "willow",
          "zephyr", "harbor", "lantern", "meadow", "orchid", "prism")
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no confusable 0/O/1/I/L


def make_needles_text_v2(tokenizer, n_needles: int, seed: int) -> list[Needle]:
    """Harder variant (D-016): built to discriminate where v1 saturated at ceiling.

    Differences from v1: (a) QUERY cue is a PARAPHRASE of the statement wording, so
    pure verbatim induction-copy on the cue no longer suffices; (b) 25% of needles
    are UPDATED — the original value is stated early, a correction later, and the
    correction is the scored answer (interference + recency resolution); (c) names
    collide pairwise on the word part ("kestrel-42" vs "kestrel-87"), so attending
    to roughly-the-right entity is not enough.
    """
    rng = random.Random(seed)
    needles: list[Needle] = []
    words = list(_WORDS)
    rng.shuffle(words)
    names = []
    for i in range(n_needles):  # pairwise word collisions
        word = words[(i // 2) % len(words)]
        while True:
            name = f"{word}-{rng.randrange(10, 99)}"
            if name not in names:
                names.append(name)
                break
    for i, name in enumerate(names):
        value = "".join(rng.choice(_ALPHABET) for _ in range(8))
        cue_text = f"\nAs noted earlier, the code associated with {name} is"
        cue_ids = tokenizer.encode(cue_text, add_special_tokens=False)
        cue_plus = tokenizer.encode(cue_text + " " + value, add_special_tokens=False)
        if cue_plus[: len(cue_ids)] != cue_ids:
            continue
        depth = (i + 0.5) / n_needles
        updated = i % 4 == 0  # 25% get an interfering original + correction
        if updated:
            old_value = "".join(rng.choice(_ALPHABET) for _ in range(8))
            original = f"\nRemember this: the secret code for {name} is {old_value}.\n"
            correction = (f"\nCorrection: the secret code for {name} has been "
                          f"changed to {value}.\n")
            needles.append(Needle(
                name=name, value=value, depth=min(depth + 0.15, 0.97), kind="updated",
                statement_ids=tokenizer.encode(correction, add_special_tokens=False),
                cue_ids=cue_ids, value_ids=cue_plus[len(cue_ids):],
                extra_statements=[{"depth": max(depth - 0.15, 0.02),
                                   "ids": tokenizer.encode(original,
                                                           add_special_tokens=False)}],
            ))
        else:
            statement = f"\nRemember this: the secret code for {name} is {value}.\n"
            needles.append(Needle(
                name=name, value=value, depth=depth, kind="plain",
                statement_ids=tokenizer.encode(statement, add_special_tokens=False),
                cue_ids=cue_ids, value_ids=cue_plus[len(cue_ids):],
            ))
    return needles


def make_needles_text(tokenizer, n_needles: int, seed: int) -> list[Needle]:
    """Generate deterministic name->code needles with a real tokenizer.

    Value ids are derived by prefix subtraction (encode(cue) vs encode(cue+value)),
    so the scored ground truth is exactly how the value tokenizes after the cue.
    """
    rng = random.Random(seed)
    needles = []
    names_used = set()
    for i in range(n_needles):
        while True:
            name = f"{rng.choice(_WORDS)}-{rng.randrange(10, 99)}"
            if name not in names_used:
                names_used.add(name)
                break
        value = "".join(rng.choice(_ALPHABET) for _ in range(8))
        cue_text = f"\nThe secret code for {name} is"
        statement_text = f"\nRemember this: the secret code for {name} is {value}.\n"
        cue_ids = tokenizer.encode(cue_text, add_special_tokens=False)
        cue_plus = tokenizer.encode(cue_text + " " + value, add_special_tokens=False)
        if cue_plus[: len(cue_ids)] != cue_ids:  # rare BPE boundary merge: re-roll
            continue
        depth = (i + 0.5) / n_needles  # even coverage, no needle at the extremes
        needles.append(Needle(
            name=name, value=value, depth=depth,
            statement_ids=tokenizer.encode(statement_text, add_special_tokens=False),
            cue_ids=cue_ids, value_ids=cue_plus[len(cue_ids):],
        ))
    return needles
