"""Versioned per-group precision map: the calibration output and the mixed-policy input.

A `PrecisionMap` lists entries (layer, state, channel range, kind); everything not listed
stays BF16. A map with a single entry is a one-group perturbation experiment; a full map
is a deployable mixed-precision policy — `mapped_cache.MappedQDQCache` consumes both.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .qdq import effective_group_size
from .targets import INDEXER_STATE, STATES_BY_LAYER_TYPE, QuantTarget, nope_width

MAP_VERSION = 1

MAIN_KINDS = ("fp8_e4m3", "fp4_e2m1")
INDEXER_KINDS = ("fp4_e2m1_hadamard", "fp4_e2m1", "fp8_e4m3")


@dataclass(frozen=True)
class MapEntry:
    layer_idx: int
    state: str
    group_index: int
    start: int
    end: int
    kind: str
    # width of the QDQ scale groups WITHIN this entry (defaults to the entry width;
    # indexer entries use the official 32 / tiny fallback so a full-coverage entry
    # reproduces the Task-02 whole-state policy bitwise)
    scale_group_size: int = 0

    def effective_scale_group(self) -> int:
        width = self.end - self.start
        requested = self.scale_group_size or width
        return effective_group_size(width, requested)

    @classmethod
    def for_target(cls, target: QuantTarget, kind: str, scale_group_size: int = 0) -> "MapEntry":
        return cls(
            layer_idx=target.layer_idx,
            state=target.state,
            group_index=target.group_index,
            start=target.start,
            end=target.end,
            kind=kind,
            scale_group_size=scale_group_size,
        )


@dataclass
class PrecisionMap:
    name: str = "unnamed"
    version: int = MAP_VERSION
    default: str = "bf16"
    entries: list[MapEntry] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def entries_for(self, layer_idx: int, state: str) -> list[MapEntry]:
        return [e for e in self.entries if e.layer_idx == layer_idx and e.state == state]

    def indexer_entries(self) -> list[MapEntry]:
        return [e for e in self.entries if e.state == INDEXER_STATE]

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def validate(self, config) -> None:
        """Reject maps that don't fit the config. Raises ValueError with a precise reason."""
        width = nope_width(config)
        seen: dict[tuple[int, str], list[tuple[int, int]]] = {}
        for e in self.entries:
            if not 0 <= e.layer_idx < config.num_hidden_layers:
                raise ValueError(f"{e}: layer_idx out of range")
            layer_type = config.layer_types[e.layer_idx]
            if e.state not in STATES_BY_LAYER_TYPE[layer_type]:
                raise ValueError(f"{e}: state {e.state!r} not present on {layer_type!r} layer")
            state_width = config.index_head_dim if e.state == INDEXER_STATE else width
            if not 0 <= e.start < e.end <= state_width:
                raise ValueError(f"{e}: channel range outside width {state_width}")
            allowed = INDEXER_KINDS if e.state == INDEXER_STATE else MAIN_KINDS
            if e.kind not in allowed:
                raise ValueError(f"{e}: kind {e.kind!r} not in {allowed}")
            if e.state == INDEXER_STATE and (e.start, e.end) != (0, config.index_head_dim):
                raise ValueError(
                    f"{e}: indexer entries must cover the full vector (rotation mixes channels)"
                )
            e.effective_scale_group()  # raises on incompatible scale group
            spans = seen.setdefault((e.layer_idx, e.state), [])
            for s, t in spans:
                if e.start < t and s < e.end:
                    raise ValueError(f"{e}: overlaps another entry on the same state")
            spans.append((e.start, e.end))

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "default": self.default,
            "provenance": self.provenance,
            "entries": [asdict(e) for e in self.entries],
        }

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "PrecisionMap":
        if data.get("version", MAP_VERSION) != MAP_VERSION:
            raise ValueError(f"unsupported precision-map version {data.get('version')}")
        entries = [MapEntry(**e) for e in data.get("entries", [])]
        return cls(
            name=data.get("name", "unnamed"),
            version=MAP_VERSION,
            default=data.get("default", "bf16"),
            entries=entries,
            provenance=data.get("provenance", {}),
        )

    @classmethod
    def from_json(cls, source: str | Path) -> "PrecisionMap":
        path = Path(source)
        text = path.read_text() if path.exists() else str(source)
        return cls.from_dict(json.loads(text))
