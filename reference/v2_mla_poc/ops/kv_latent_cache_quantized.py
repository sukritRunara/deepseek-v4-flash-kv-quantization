"""
ops/kv_latent_cache_quantized.py

Quantized latent KV cache for MLA models (DeepSeek-V2-Lite, Kimi-K2.6).

Extends KVLatentCache to store c_kv (kv_a_norm) in FP8 or FP4 instead of BF16,
with optional mixed-precision per channel based on sensitivity scores.

Memory layout per layer per token:
  - c_kv (quantized) : [kv_lora_rank] in FP8/FP4 (simulated as BF16)
  - c_kv scale       : per-channel [kv_lora_rank] float32 scale factors
  - k_pe             : [qk_rope] in BF16 (unchanged — positional, kept precise)

Precision levels:
  PREC_BF16 = 0 — no quantization, full BF16
  PREC_FP8  = 1 — FP8 e4m3fn, range ±448
  PREC_FP4  = 2 — FP4 e2m1,   range ±6

Quantization scheme:
  - Per-channel absmax scaling for both FP8 and FP4.
  - FP4 uses nearest-value rounding to the 8 representable magnitudes of e2m1.
  - Mixed precision: each channel independently assigned a precision level
    via precision_config (uint8 tensor per layer).

Interface:
  # Uniform FP8 (original behavior, fully backward-compatible)
  cache = QuantizedKVLatentCache()

  # Uniform FP4
  cache = QuantizedKVLatentCache(precision="fp4")

  # Mixed FP8/BF16 from sensitivity (legacy bool-mask interface)
  quant_config = analyzer.make_quant_config(scores, fp8_fraction=0.8)
  cache = QuantizedKVLatentCache(quant_config=quant_config)

  # 3-level mixed FP4/FP8/BF16 from sensitivity (new interface)
  precision_config = analyzer.make_mixed_precision_config(scores, fp4_fraction=0.5, fp8_fraction=0.3)
  cache = QuantizedKVLatentCache(precision_config=precision_config)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.kv_latent_cache import KVLatentCache

# ── Precision level constants ──────────────────────────────────────────────────
PREC_BF16 = 0   # no quantization
PREC_FP8  = 1   # FP8 e4m3fn        (8 bit)
PREC_FP4  = 2   # FP4 e2m1          (4 bit, 8 magnitudes)
PREC_FP3  = 3   # FP3 e2m0          (3 bit, 4 magnitudes — powers of 2)
PREC_INT2 = 4   # INT2 uniform      (2 bit, 4 uniform levels)

# ── FP8 constants ──────────────────────────────────────────────────────────────
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX   = torch.finfo(FP8_DTYPE).max   # 448.0

# ── FP4 e2m1 constants ─────────────────────────────────────────────────────────
# e2m1: 1 sign + 2 exponent + 1 mantissa bits
# Positive representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
FP4_MAX         = 6.0
_FP4_ABS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# ── FP3 e2m0 constants ─────────────────────────────────────────────────────────
# e2m0: 1 sign + 2 exponent + 0 mantissa bits — pure powers of 2, 4 magnitudes
# Positive representable values: 0, 1, 2, 4
FP3_MAX         = 4.0
_FP3_ABS_VALUES = torch.tensor([0.0, 1.0, 2.0, 4.0])

# ── INT2 uniform constants ─────────────────────────────────────────────────────
# 2-bit uniform quantization: 4 equally spaced levels in [0, max]
# Positive representable values: 0, 1, 2, 3  (scaled to channel range)
INT2_MAX         = 3.0
_INT2_ABS_VALUES = torch.tensor([0.0, 1.0, 2.0, 3.0])


# ── Low-level quantization functions ──────────────────────────────────────────

def _quantize_fp8(x: Tensor, quant_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Simulate FP8 e4m3fn via per-channel absmax + round-trip cast.

    Args:
        x:          [..., C] BF16 tensor.
        quant_mask: [C] bool — True = quantize this channel. None = quantize all.

    Returns:
        (x_q, scales) where x_q is BF16 after FP8 round-trip on masked channels.
    """
    C = x.shape[-1]
    x_f32 = x.float()
    ch_max = x_f32.reshape(-1, C).abs().amax(dim=0).clamp(min=1e-12)
    scales = ch_max / FP8_MAX  # [C]

    if quant_mask is None:
        quant_mask = torch.ones(C, dtype=torch.bool, device=x.device)

    result = x.clone()
    if quant_mask.any():
        idx = quant_mask.nonzero(as_tuple=True)[0]
        x_scaled = (x_f32[..., idx] / scales[idx]).clamp(-FP8_MAX, FP8_MAX)
        x_fp8 = x_scaled.to(FP8_DTYPE).to(torch.bfloat16)
        result[..., idx] = (x_fp8 * scales[idx]).to(torch.bfloat16)

    return result, scales


def _quantize_fp4(x: Tensor, quant_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Simulate FP4 e2m1 via per-channel absmax + nearest-value rounding.

    FP4 e2m1 has 8 positive representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    We scale each channel so its max maps to FP4_MAX=6, round to the nearest
    representable value, then scale back.

    Args:
        x:          [..., C] BF16 tensor.
        quant_mask: [C] bool — True = quantize this channel. None = quantize all.

    Returns:
        (x_q, scales) where x_q is BF16 after FP4 round-trip on masked channels.
    """
    C = x.shape[-1]
    x_f32 = x.float()
    ch_max = x_f32.reshape(-1, C).abs().amax(dim=0).clamp(min=1e-12)
    scales = ch_max / FP4_MAX  # [C]

    if quant_mask is None:
        quant_mask = torch.ones(C, dtype=torch.bool, device=x.device)

    result = x.clone()
    if quant_mask.any():
        idx  = quant_mask.nonzero(as_tuple=True)[0]
        vals = _FP4_ABS_VALUES.to(x.device)  # [8]

        x_scaled  = x_f32[..., idx] / scales[idx]          # [..., n]
        sign      = x_scaled.sign()
        abs_x     = x_scaled.abs().clamp(0.0, FP4_MAX)

        # Nearest FP4 value: unsqueeze for broadcast against [8] vals
        diff      = (abs_x.unsqueeze(-1) - vals).abs()      # [..., n, 8]
        nearest   = vals[diff.argmin(dim=-1)]                # [..., n]
        quantized = nearest * sign                           # [..., n]

        result[..., idx] = (quantized * scales[idx]).to(torch.bfloat16)

    return result, scales


def _quantize_nearest(
    x: Tensor,
    abs_values: Tensor,
    x_max: float,
    quant_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Generic nearest-value quantization for any discrete grid.

    Shared by FP3 and INT2 — identical logic to _quantize_fp4 but parameterized.

    Args:
        x:          [..., C] BF16 tensor.
        abs_values: [N] sorted non-negative representable magnitudes.
        x_max:      Maximum representable value (used for per-channel scaling).
        quant_mask: [C] bool — True = quantize this channel. None = quantize all.

    Returns:
        (x_q, scales) — x_q BF16 after round-trip, scales [C] float32.
    """
    C = x.shape[-1]
    x_f32 = x.float()
    ch_max = x_f32.reshape(-1, C).abs().amax(dim=0).clamp(min=1e-12)
    scales = ch_max / x_max

    if quant_mask is None:
        quant_mask = torch.ones(C, dtype=torch.bool, device=x.device)

    result = x.clone()
    if quant_mask.any():
        idx  = quant_mask.nonzero(as_tuple=True)[0]
        vals = abs_values.to(x.device)

        x_scaled  = x_f32[..., idx] / scales[idx]
        sign      = x_scaled.sign()
        abs_x     = x_scaled.abs().clamp(0.0, x_max)

        diff      = (abs_x.unsqueeze(-1) - vals).abs()
        nearest   = vals[diff.argmin(dim=-1)]
        quantized = nearest * sign

        result[..., idx] = (quantized * scales[idx]).to(torch.bfloat16)

    return result, scales


def _quantize_fp3(x: Tensor, quant_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Simulate FP3 e2m0: 4 representable magnitudes {0, 1, 2, 4}."""
    return _quantize_nearest(x, _FP3_ABS_VALUES, FP3_MAX, quant_mask)


def _quantize_int2(x: Tensor, quant_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Simulate INT2: 4 uniform levels {0, 1, 2, 3} scaled to channel range."""
    return _quantize_nearest(x, _INT2_ABS_VALUES, INT2_MAX, quant_mask)


def _apply_quantization(
    x:            Tensor,
    prec:         Tensor | None,
    default_prec: int = PREC_FP8,
) -> tuple[Tensor, Tensor]:
    """Apply per-channel quantization according to a precision map.

    Args:
        x:            [..., C] BF16 tensor.
        prec:         [C] uint8 tensor with values PREC_BF16/FP8/FP4.
                      None → uniform precision given by default_prec.
        default_prec: Precision to use when prec is None.

    Returns:
        (x_q, scales) — x_q BF16 after round-trip, scales [C] float32.
    """
    C = x.shape[-1]

    _UNIFORM_DISPATCH = {
        PREC_FP8:  _quantize_fp8,
        PREC_FP4:  _quantize_fp4,
        PREC_FP3:  _quantize_fp3,
        PREC_INT2: _quantize_int2,
    }

    if prec is None:
        fn = _UNIFORM_DISPATCH.get(default_prec)
        if fn is None:  # PREC_BF16
            return x, torch.ones(C, dtype=torch.float32, device=x.device)
        return fn(x)

    # Mixed path: dispatch each precision level independently
    result = x.clone()
    scales = torch.ones(C, dtype=torch.float32, device=x.device)

    for prec_val, fn in _UNIFORM_DISPATCH.items():
        mask = prec == prec_val
        if mask.any():
            x_q, sc = fn(x, quant_mask=mask)
            result[..., mask] = x_q[..., mask]
            scales[mask] = sc[mask]

    return result, scales


# ── Cache class ────────────────────────────────────────────────────────────────

class QuantizedKVLatentCache(KVLatentCache):
    """KVLatentCache subclass that stores c_kv in FP8 or FP4 instead of BF16.

    Inherits from KVLatentCache (which inherits from DynamicCache) so that
    DeepSeek's modeling code recognises it as a valid cache object and does
    not silently replace it with a plain DynamicCache.

    Args:
        precision_config:  NEW — dict[int, Tensor] mapping layer_idx to a [C]
                           uint8 tensor with per-channel precision levels:
                               0 = BF16 (no quantization)
                               1 = FP8 e4m3fn
                               2 = FP4 e2m1
                           Use SensitivityAnalyzer.make_mixed_precision_config()
                           to build this from calibration data.
                           If provided, overrides quant_config and precision.

        quant_config:      LEGACY — dict[int, Tensor] mapping layer_idx to a [C]
                           bool mask (True = quantize). Precision of quantized
                           channels is determined by the `precision` arg.
                           If None, all channels are quantized uniformly.

        precision:         "fp8" (default) or "fp4" — uniform precision used
                           for all quantized channels when using quant_config.

        simulate_fp8:      LEGACY — kept for backward compat, has no effect.
    """

    def __init__(
        self,
        precision_config: dict[int, Tensor] | None = None,
        quant_config:     dict[int, Tensor] | None = None,
        precision:        Literal["fp8", "fp4", "fp3", "int2"] = "fp8",
        simulate_fp8:     bool = True,   # legacy, ignored
    ) -> None:
        super().__init__()

        if precision_config is not None:
            self._prec_config  = precision_config
            self._default_prec = PREC_FP8  # fallback for layers not in config
        elif quant_config is not None:
            # Convert bool mask → uint8 precision map
            p = PREC_FP8 if precision == "fp8" else PREC_FP4
            self._prec_config = {
                layer: (mask.to(torch.uint8) * p)
                for layer, mask in quant_config.items()
            }
            self._default_prec = p
        else:
            # Uniform precision for all channels in all layers
            self._prec_config  = None
            _prec_map = {"fp8": PREC_FP8, "fp4": PREC_FP4, "fp3": PREC_FP3, "int2": PREC_INT2}
            self._default_prec = _prec_map.get(precision, PREC_FP8)

        self._scales_cache: list[Tensor | None] = []

    def update(
        self,
        key_states:   Tensor,
        value_states: Tensor,
        layer_idx:    int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Store quantized (c_kv, k_pe) and return the full accumulated cache."""
        prec = None
        if self._prec_config is not None:
            prec = self._prec_config.get(layer_idx)
            if prec is not None:
                prec = prec.to(key_states.device)

        c_kv_q, scales = _apply_quantization(key_states, prec, self._default_prec)

        # Track running-max scales per layer (for diagnostics)
        while len(self._scales_cache) <= layer_idx:
            self._scales_cache.append(None)
        if self._scales_cache[layer_idx] is None:
            self._scales_cache[layer_idx] = scales.clone()
        else:
            self._scales_cache[layer_idx] = torch.maximum(
                self._scales_cache[layer_idx], scales
            )

        return super().update(c_kv_q, value_states, layer_idx, cache_kwargs)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def precision_fractions(self) -> dict[str, float]:
        """Fraction of channels at each precision level across all layers."""
        _prec_names = {PREC_FP8: "fp8", PREC_FP4: "fp4", PREC_FP3: "fp3", PREC_INT2: "int2"}
        if self._prec_config is None:
            prec_name = _prec_names.get(self._default_prec, "bf16")
            return {k: (1.0 if k == prec_name else 0.0)
                    for k in ("bf16", "fp8", "fp4", "fp3", "int2")}
        counts = {k: 0 for k in ("bf16", "fp8", "fp4", "fp3", "int2")}
        total = 0
        for v in self._prec_config.values():
            total += v.numel()
            counts["bf16"] += (v == PREC_BF16).sum().item()
            counts["fp8"]  += (v == PREC_FP8 ).sum().item()
            counts["fp4"]  += (v == PREC_FP4 ).sum().item()
            counts["fp3"]  += (v == PREC_FP3 ).sum().item()
            counts["int2"] += (v == PREC_INT2).sum().item()
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: counts[k]/total for k in counts}

    def cache_size_bytes(self) -> dict[str, int]:
        """Estimate theoretical memory of the current cache contents.

        BF16  channels: 2.000 bytes/element
        FP8   channels: 1.000 byte/element
        FP4   channels: 0.500 bytes/element (4 bits, 2 packed per byte)
        FP3   channels: 0.375 bytes/element (3 bits, ~2.67 packed per byte)
        INT2  channels: 0.250 bytes/element (2 bits, 4 packed per byte)
        k_pe: always BF16 (2.0 bytes/element).
        """
        fracs = self.precision_fractions()
        bytes_per_elem = (
            fracs["bf16"] * 2.000 +
            fracs["fp8"]  * 1.000 +
            fracs["fp4"]  * 0.500 +
            fracs["fp3"]  * 0.375 +
            fracs["int2"] * 0.250
        )
        c_kv_bytes = 0
        kpe_bytes  = 0
        for t in self.key_cache:
            if t is not None and t.numel() > 0:
                c_kv_bytes += int(t.numel() * bytes_per_elem)
        for t in self.value_cache:
            if t is not None and t.numel() > 0:
                kpe_bytes += t.numel() * 2

        return {
            "c_kv_bytes":  c_kv_bytes,
            "kpe_bytes":   kpe_bytes,
            "total_bytes": c_kv_bytes + kpe_bytes,
        }

    def memory_report(self) -> str:
        sizes = self.cache_size_bytes()
        fracs = self.precision_fractions()
        n     = sum(1 for t in self.key_cache if t is not None and t.numel() > 0)
        lines = [f"QuantizedKVLatentCache | {n} layers populated"]
        for label, key in [("BF16", "bf16"), ("FP8", "fp8"), ("FP4", "fp4"),
                            ("FP3", "fp3"), ("INT2", "int2")]:
            if fracs[key] > 0:
                lines.append(f"  {label:<4} fraction: {fracs[key]:.1%}")
        lines += [
            f"  c_kv mem     : {sizes['c_kv_bytes'] / 1e6:.2f} MB",
            f"  k_pe mem     : {sizes['kpe_bytes']  / 1e6:.2f} MB",
            f"  total mem    : {sizes['total_bytes'] / 1e6:.2f} MB",
            f"  seq length   : {self.get_seq_length(0)}",
        ]
        return "\n".join(lines)