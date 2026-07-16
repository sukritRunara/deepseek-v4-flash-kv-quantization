"""Serializable KV-cache precision policies (Stage B: QDQ simulation).

A policy fully determines the numerical treatment of every quantizable cache
state; experiments select behavior ONLY through a policy (no code edits).
RoPE dims (trailing ``qk_rope_head_dim``) are never quantized in version 1 —
the official reference keeps them BF16 ("positional precision", model.py:505)
and our tests pin them bit-exact.

Kinds:
  bf16              -> identity (no QDQ)
  fp8_e4m3          -> fp8_e4m3_qdq(group_size, pow2_scale) on the non-RoPE slice
  fp4_e2m1_hadamard -> hadamard_transform then fp4_e2m1_qdq(group_size) on the FULL
                       vector (official indexer path; stored in the rotated basis,
                       queries are rotated symmetrically by the scorer wrapper)
  fp4_e2m1          -> fp4_e2m1_qdq without rotation (experimental, main-KV FP4)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

StateKind = Literal["bf16", "fp8_e4m3", "fp4_e2m1", "fp4_e2m1_hadamard"]

POLICY_VERSION = 1


@dataclass(frozen=True)
class StatePolicy:
    """Numerical policy for one cache state family."""

    kind: StateKind = "bf16"
    group_size: int = 64
    pow2_scale: bool = True  # ue8m0-style round-up power-of-2 scales (official)

    @property
    def is_identity(self) -> bool:
        return self.kind == "bf16"


@dataclass(frozen=True)
class KVQuantPolicy:
    """Full cache precision policy. All states default to BF16 (identity)."""

    name: str = "baseline_bf16"
    version: int = POLICY_VERSION
    window_kv: StatePolicy = field(default_factory=StatePolicy)      # non-RoPE slice only
    compressed_kv: StatePolicy = field(default_factory=StatePolicy)  # non-RoPE slice only
    indexer_kv: StatePolicy = field(default_factory=StatePolicy)     # full vector
    indexer_q: StatePolicy = field(default_factory=StatePolicy)      # full vector, symmetric

    @property
    def is_identity(self) -> bool:
        return all(
            p.is_identity for p in (self.window_kv, self.compressed_kv, self.indexer_kv, self.indexer_q)
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "KVQuantPolicy":
        data = dict(data)
        version = data.get("version", POLICY_VERSION)
        if version != POLICY_VERSION:
            raise ValueError(f"unsupported policy version {version} (expected {POLICY_VERSION})")
        for key in ("window_kv", "compressed_kv", "indexer_kv", "indexer_q"):
            if key in data and isinstance(data[key], dict):
                data[key] = StatePolicy(**data[key])
        return cls(**data)

    @classmethod
    def from_json(cls, source: str | Path) -> "KVQuantPolicy":
        path = Path(source)
        text = path.read_text() if path.exists() else str(source)
        return cls.from_dict(json.loads(text))


_FP8_OFFICIAL = StatePolicy(kind="fp8_e4m3", group_size=64, pow2_scale=True)
_FP4_INDEXER = StatePolicy(kind="fp4_e2m1_hadamard", group_size=32, pow2_scale=True)
_FP4_MAIN = StatePolicy(kind="fp4_e2m1", group_size=32, pow2_scale=True)


def baseline_bf16() -> KVQuantPolicy:
    """Identity policy: must be bit-exact vs the stock cache (acceptance gate 1)."""
    return KVQuantPolicy(name="baseline_bf16")


def reference_official_qdq() -> KVQuantPolicy:
    """The full official QAT-aligned policy: FP8 g64 ue8m0 main KV (non-RoPE) +
    Hadamard FP4 g32 indexer keys and queries."""
    return KVQuantPolicy(
        name="reference_official_qdq",
        window_kv=_FP8_OFFICIAL,
        compressed_kv=_FP8_OFFICIAL,
        indexer_kv=_FP4_INDEXER,
        indexer_q=_FP4_INDEXER,
    )


def main_fp8_nonrope_rope_bf16() -> KVQuantPolicy:
    """Main KV only (window + compressed) at official FP8; indexer untouched."""
    return KVQuantPolicy(
        name="main_fp8_nonrope_rope_bf16",
        window_kv=_FP8_OFFICIAL,
        compressed_kv=_FP8_OFFICIAL,
    )


def main_fp4_nonrope_rope_bf16() -> KVQuantPolicy:
    """EXPERIMENTAL: main KV non-RoPE at FP4 g32 (beyond the official policy)."""
    return KVQuantPolicy(
        name="main_fp4_nonrope_rope_bf16",
        window_kv=_FP4_MAIN,
        compressed_kv=_FP4_MAIN,
    )


def indexer_reference_qdq() -> KVQuantPolicy:
    """Indexer keys + queries only (official FP4 path); main KV untouched."""
    return KVQuantPolicy(
        name="indexer_reference_qdq",
        indexer_kv=_FP4_INDEXER,
        indexer_q=_FP4_INDEXER,
    )


NAMED_POLICIES = {
    p().name: p
    for p in (
        baseline_bf16,
        reference_official_qdq,
        main_fp8_nonrope_rope_bf16,
        main_fp4_nonrope_rope_bf16,
        indexer_reference_qdq,
    )
}
