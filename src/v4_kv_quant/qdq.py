"""Pure-PyTorch quantize-dequantize primitives matching DeepSeek-V4's official kernels.

Reference semantics: ``vendor/DeepSeek-V4-Flash/inference/kernel.py``
(``act_quant_kernel``, ``fp4_quant_kernel``, ``fast_round_scale``) and
``inference/model.py`` (``rotate_activation``). All functions:

* operate on contiguous groups along the LAST dimension;
* compute in float32 regardless of input dtype, return the input dtype
  (mirrors the kernels' BF16-in / FP32-compute / BF16-out pipeline);
* are stateless and deterministic;
* simulate only — outputs are dequantized values in the original dtype.
  Stage C (actual low-precision storage) is a separate, later concern.
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0  # float8_e4m3fn max normal
FP8_AMAX_FLOOR = 1e-4  # kernel.py:79
FP4_MAX = 6.0  # e2m1 max
FP4_AMAX_FLOOR = 6.0 * 2.0**-126  # kernel.py:163

# e2m1 non-negative representable magnitudes and the round-to-nearest-even tie
# choice at each midpoint between adjacent grid values (even mantissa bit wins:
# ties at 0.25/0.75/1.25/1.75/2.5/3.5/5.0 round to 0/1/1/2/2/4/4).
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_TIE_VALUES = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])


def ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """Exact ``2 ** ceil(log2(x))`` for positive x (bit-exact, no transcendental error).

    Mirrors ``fast_round_scale``/``fast_log2_ceil`` in the official kernel, which use IEEE
    bit manipulation. ``torch.frexp`` gives ``x = m * 2**e`` with ``m in [0.5, 1)``;
    ``m == 0.5`` iff x is a power of two, in which case ``ceil(log2 x) = e - 1``, else ``e``.
    """
    mantissa, exponent = torch.frexp(x.float())
    ceil_log2 = torch.where(mantissa == 0.5, exponent - 1, exponent)
    # 2**ceil_log2 built by IEEE-754 bit layout instead of torch.ldexp: bit-exact for the
    # normal range (amax floors keep exponents well inside [-126, 127]), and — unlike
    # ldexp in torch 2.13.0+cu130 — device-safe when the tensor lives on a CUDA device
    # that is not the current one (ldexp there returns garbage/NaN or IMAs: missing
    # device guard, found on the Phase-B 4-GPU pod; WORKLOG 2026-07-17 B3).
    return ((ceil_log2.to(torch.int32) + 127) << 23).view(torch.float32)


def _group_scales(
    x_grouped: torch.Tensor, qmax: float, amax_floor: float, pow2_scale: bool
) -> torch.Tensor:
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp_min(amax_floor)
    if pow2_scale:
        return ceil_pow2(amax / qmax)
    return amax / qmax


def fp8_e4m3_qdq(
    x: torch.Tensor,
    group_size: int = 64,
    pow2_scale: bool = True,
    return_scales: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FP8 e4m3 quantize-dequantize per contiguous group along the last dim.

    Official main-KV policy (``model.py:506``): ``group_size=64``, ``pow2_scale=True``
    (scale_fmt "ue8m0": absmax scale rounded UP to a power of two). Rounding to the
    e4m3 grid uses torch's native cast (round-to-nearest-even), matching ``T.Cast(FP8, ...)``.
    """
    if x.numel() == 0:
        return (x, x.new_zeros(*x.shape[:-1], 0)) if return_scales else x
    width = x.shape[-1]
    if width % group_size != 0:
        raise ValueError(f"last dim {width} not divisible by group_size {group_size}")
    x32 = x.float().unflatten(-1, (width // group_size, group_size))
    scales = _group_scales(x32, FP8_MAX, FP8_AMAX_FLOOR, pow2_scale)
    q = (x32 / scales).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    y = (q.float() * scales).flatten(-2).to(x.dtype)
    if return_scales:
        return y, scales.squeeze(-1)
    return y


def _round_e2m1(u: torch.Tensor) -> torch.Tensor:
    """Round non-negative values (already clamped to [0, 6]) to the e2m1 grid, ties-to-even."""
    grid = _E2M1_GRID.to(u.device)
    mids = _E2M1_MIDPOINTS.to(u.device)
    ties = _E2M1_TIE_VALUES.to(u.device)
    # For u strictly inside (mids[i-1], mids[i]) both bucketize variants return i, and
    # grid[i] is the nearest value. They disagree exactly on midpoints: right=False
    # returns the midpoint's own index i, right=True returns i+1 -> apply the tie table.
    lower = torch.bucketize(u, mids, right=False)
    upper = torch.bucketize(u, mids, right=True)
    rounded = grid[lower]
    tie_mask = lower != upper
    if tie_mask.any():
        rounded = torch.where(tie_mask, ties[lower.clamp_max(mids.numel() - 1)], rounded)
    return rounded


def fp4_e2m1_qdq(
    x: torch.Tensor,
    group_size: int = 32,
    return_scales: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FP4 e2m1 quantize-dequantize per contiguous group along the last dim.

    Official indexer policy (``model.py:368-370, 414-416``): ``group_size=32``, scale
    always power-of-two round-up of ``amax/6`` with floor ``6*2**-126`` (e8m0-representable).
    torch 2.13 cannot cast to ``float4_e2m1fn_x2`` (storage-only dtype, verified), so the
    e2m1 grid is applied in software with round-to-nearest-even tie behavior.
    """
    if x.numel() == 0:
        return (x, x.new_zeros(*x.shape[:-1], 0)) if return_scales else x
    width = x.shape[-1]
    if width % group_size != 0:
        raise ValueError(f"last dim {width} not divisible by group_size {group_size}")
    x32 = x.float().unflatten(-1, (width // group_size, group_size))
    scales = _group_scales(x32, FP4_MAX, FP4_AMAX_FLOOR, pow2_scale=True)
    u = (x32 / scales).clamp(-FP4_MAX, FP4_MAX)
    q = torch.copysign(_round_e2m1(u.abs()), u)
    y = (q * scales).flatten(-2).to(x.dtype)
    if return_scales:
        return y, scales.squeeze(-1)
    return y


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform over the last dim (power of two).

    Matches ``rotate_activation`` (``model.py:247-251``): ``hadamard_transform(x,
    scale=n**-0.5)`` from the ``fast_hadamard_transform`` package, i.e. the Sylvester
    H_n with 1/sqrt(n) normalization. Orthonormal and symmetric, hence an involution:
    ``hadamard_transform(hadamard_transform(x)) == x`` (up to float noise). Computed in
    fp32, returned in the input dtype.
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard transform needs a power-of-two dim, got {n}")
    lead = x.shape[:-1]
    y = x.float().reshape(-1, n)
    half = 1
    while half < n:
        y = y.view(-1, n // (2 * half), 2, half)
        even, odd = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((even + odd, even - odd), dim=2).view(-1, n)
        half *= 2
    return (y * n**-0.5).view(*lead, n).to(x.dtype)


def effective_group_size(width: int, requested: int) -> int:
    """Group size actually used for a state of ``width`` channels.

    Production widths divide the official group sizes exactly (448 % 64 == 0,
    128 % 32 == 0). Tiny test models have narrower states (e.g. nope width 24,
    indexer dim 16); there the whole state is one group. Anything else is a
    configuration error — silent fallback is not allowed (CLAUDE.md constraint 10).
    """
    if width % requested == 0:
        return requested
    if width < requested:
        return width
    raise ValueError(f"width {width} incompatible with group_size {requested}")
