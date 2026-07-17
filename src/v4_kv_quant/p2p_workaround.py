"""Workaround for faulty PCIe peer-to-peer (P2P) DMA on multi-GPU pods.

Discovered on the RunPod Phase-B pod (2026-07-17, see WORKLOG and D-011): direct
GPU-to-GPU copies silently corrupt data — ~50-100% of 64-256 MiB transfers, and ~100%
of transfers issued concurrently with compute on either device (the exact regime of a
``device_map="auto"`` pipelined model). Device-to-host copies are unaffected. This is a
host-level PCIe ACS/IOMMU misconfiguration, not a PyTorch/kernel/SM120 problem: with
peer access disabled, the same worst-case stress is 0/240 corrupt.

``ensure_host_staged_p2p()`` forces every ordered device pair's peer access OFF so
``cudaMemcpyPeerAsync`` stages through host memory (correct, modestly slower). Torch
enables peer access lazily on the first cross-device copy of each pair and caches that
decision, so the sequence here is: trigger the lazy enable with a tiny copy per pair,
then disable via the CUDA runtime. Call this once, after CUDA init, before moving model
data between GPUs (idempotent; safe to call when P2P is healthy, at the cost of slower
inter-GPU copies).

Validation for a given pod: ``tools/p2p_stress_check.py`` (exits non-zero on corruption).
"""

from __future__ import annotations

import ctypes
import ctypes.util

import torch

_CUDA_SUCCESS = 0
_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705


def _load_cudart() -> ctypes.CDLL:
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    found = ctypes.util.find_library("cudart")
    if found is None:
        raise OSError("libcudart not found; cannot manage CUDA peer access")
    return ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)


def ensure_host_staged_p2p(verbose: bool = True) -> int:
    """Disable CUDA peer access between all device pairs; return #pairs disabled.

    No-op (returns 0) with fewer than two visible CUDA devices.
    """
    n = torch.cuda.device_count()
    if n < 2:
        return 0
    libcudart = _load_cudart()

    # Trigger torch's lazy peer-access enablement so our disable is not undone later:
    # torch caches per-pair enablement and never re-enables after this.
    for i in range(n):
        for j in range(n):
            if i != j:
                torch.ones(1, device=f"cuda:{i}").to(f"cuda:{j}")
    torch.cuda.synchronize()

    disabled = 0
    for i in range(n):
        rc = libcudart.cudaSetDevice(i)
        if rc != _CUDA_SUCCESS:
            raise RuntimeError(f"cudaSetDevice({i}) failed: {rc}")
        for j in range(n):
            if i == j:
                continue
            rc = libcudart.cudaDeviceDisablePeerAccess(j)
            if rc == _CUDA_SUCCESS:
                disabled += 1
            elif rc != _CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
                raise RuntimeError(f"cudaDeviceDisablePeerAccess({i}->{j}) failed: {rc}")
    # Clear sticky error state left by expected 705s so later CUDA calls are unaffected.
    libcudart.cudaGetLastError()
    if verbose:
        print(f"[p2p_workaround] peer access disabled on {disabled} device pairs "
              f"(D2D now staged through host)", flush=True)
    return disabled
