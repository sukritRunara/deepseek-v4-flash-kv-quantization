"""Actual low-precision storage primitives (Stage C).

Numerical contract, test-enforced: for every kind, ``load(store(x)) == qdq(x)`` **bitwise**
— the Stage-C storage cache must be indistinguishable (in values) from the Stage-B QDQ
simulation, so every Stage-B quality result transfers to Stage-C storage unchanged.

Representations:
  * FP8: codes as native ``float8_e4m3fn`` (1 B/value); one scale per contiguous group,
    stored as ``float8_e8m0fnu`` (1 B) when scales are powers of two (official ue8m0
    policy), fp32 otherwise.
  * FP4: e2m1 codes as sign bit + 3-bit magnitude index, two codes per ``uint8``
    (0.5 B/value); ``float8_e8m0fnu`` scales (FP4 scales are always powers of two).

`torch.float4_e2m1fn_x2` exists in torch 2.13 but supports no cast/copy kernels (probed,
docs/DECISIONS.md D-007), so nibble-packing into uint8 is the honest prototype; the packed
buffer is layout-compatible with a `float4_e2m1fn_x2` view if kernels want it later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .qdq import (
    FP4_AMAX_FLOOR,
    FP4_MAX,
    FP8_AMAX_FLOOR,
    FP8_MAX,
    _E2M1_GRID,
    _group_scales,
    _round_e2m1,
)


@dataclass
class StoredTensor:
    """A quantized tensor plus everything needed to reconstruct it."""

    codes: torch.Tensor  # fp8: float8_e4m3fn [..., W]; fp4: uint8 [..., W//2]
    scales: torch.Tensor  # [..., n_groups]; float8_e8m0fnu or float32
    group_size: int
    width: int  # logical channel count W
    out_dtype: torch.dtype

    def numel_logical(self) -> int:
        return self.codes.numel() if self.codes.dtype != torch.uint8 else self.codes.numel() * 2


def fp8_store(
    x: torch.Tensor, group_size: int, pow2_scale: bool = True
) -> StoredTensor:
    """Quantize to real FP8 storage; mirrors `fp8_e4m3_qdq` exactly (same scales/rounding)."""
    width = x.shape[-1]
    if width % group_size != 0:
        raise ValueError(f"width {width} not divisible by group_size {group_size}")
    x32 = x.float().unflatten(-1, (width // group_size, group_size))
    scales = _group_scales(x32, FP8_MAX, FP8_AMAX_FLOOR, pow2_scale)
    codes = (x32 / scales).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).flatten(-2)
    scale_dtype = torch.float8_e8m0fnu if pow2_scale else torch.float32
    return StoredTensor(codes, scales.squeeze(-1).to(scale_dtype), group_size, width, x.dtype)


def fp4_store(x: torch.Tensor, group_size: int) -> StoredTensor:
    """Quantize to packed e2m1 nibbles; mirrors `fp4_e2m1_qdq` exactly."""
    width = x.shape[-1]
    if width % group_size != 0:
        raise ValueError(f"width {width} not divisible by group_size {group_size}")
    if width % 2 != 0:
        raise ValueError(f"width {width} must be even to nibble-pack")
    grid = _E2M1_GRID.to(x.device)
    x32 = x.float().unflatten(-1, (width // group_size, group_size))
    scales = _group_scales(x32, FP4_MAX, FP4_AMAX_FLOOR, pow2_scale=True)
    u = (x32 / scales).clamp(-FP4_MAX, FP4_MAX).flatten(-2)
    magnitude = _round_e2m1(u.abs())
    index = torch.searchsorted(grid, magnitude.contiguous()).to(torch.uint8)  # exact grid values
    sign = ((u < 0) & (magnitude != 0)).to(torch.uint8)
    codes = index | (sign << 3)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return StoredTensor(packed, scales.squeeze(-1).to(torch.float8_e8m0fnu), group_size, width, x.dtype)


def _fp4_lut(device: torch.device) -> torch.Tensor:
    grid = _E2M1_GRID.to(device)
    return torch.cat([grid, -grid])  # code 0..7 positive, 8..15 negative


def load(stored: StoredTensor) -> torch.Tensor:
    """Dequantize to `out_dtype`; bitwise-equal to the matching QDQ simulation output."""
    scales = stored.scales.float().unsqueeze(-1)
    if stored.codes.dtype == torch.float8_e4m3fn:
        values = stored.codes.float().unflatten(-1, (stored.width // stored.group_size, stored.group_size))
    elif stored.codes.dtype == torch.uint8:
        low = stored.codes & 0xF
        high = stored.codes >> 4
        codes = torch.stack((low, high), dim=-1).flatten(-2)
        values = _fp4_lut(stored.codes.device)[codes.long()]
        values = values.unflatten(-1, (stored.width // stored.group_size, stored.group_size))
    else:
        raise ValueError(f"unknown code dtype {stored.codes.dtype}")
    return (values * scales).flatten(-2).to(stored.out_dtype)


def stored_bytes(stored: StoredTensor) -> dict[str, int]:
    """Actual bytes of the stored representation, itemized."""
    return {
        "codes": stored.codes.numel() * stored.codes.element_size(),
        "scales": stored.scales.numel() * stored.scales.element_size(),
    }
