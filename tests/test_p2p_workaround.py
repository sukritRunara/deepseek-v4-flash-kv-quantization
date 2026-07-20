"""D-014: the host-staged-P2P workaround is opt-in via V4_KV_FORCE_HOST_STAGED_P2P.

The RunPod Phase-B pod needed peer access forcibly disabled (D-011); the GCP G4 host
passes tools/p2p_stress_check.py natively, so the workaround must stay dormant unless
explicitly armed. These tests pin the gate order: the env check comes before ANY CUDA
interaction, so an unarmed call is side-effect-free on every host.
"""

from __future__ import annotations

import v4_kv_quant.p2p_workaround as p2p


def test_unarmed_call_returns_zero_without_touching_cuda(monkeypatch):
    monkeypatch.delenv(p2p.FORCE_ENV_VAR, raising=False)

    def _fail():
        raise AssertionError("CUDA queried although the workaround is not armed")

    monkeypatch.setattr(p2p.torch.cuda, "device_count", _fail)
    assert p2p.ensure_host_staged_p2p(verbose=False) == 0


def test_armed_env_values_other_than_1_do_not_arm(monkeypatch):
    monkeypatch.setenv(p2p.FORCE_ENV_VAR, "true")

    def _fail():
        raise AssertionError("CUDA queried although the workaround is not armed")

    monkeypatch.setattr(p2p.torch.cuda, "device_count", _fail)
    assert p2p.ensure_host_staged_p2p(verbose=False) == 0


def test_armed_but_single_device_is_noop(monkeypatch):
    monkeypatch.setenv(p2p.FORCE_ENV_VAR, "1")
    monkeypatch.setattr(p2p.torch.cuda, "device_count", lambda: 1)
    assert p2p.ensure_host_staged_p2p(verbose=False) == 0
