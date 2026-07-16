"""
ops/sensitivity.py

Per-channel sensitivity analysis for MLA latent KV cache quantization.

For each layer, we want to know: which channels of c_kv (kv_a_norm) are
safe to quantize aggressively, and which must stay in high precision?

The sensitivity of channel i is measured as how much quantization noise
in c_kv[:, :, i] perturbs the attention output. This propagates through
kv_b_proj, so the key quantity is:

    sensitivity[i] = mean_activation[i] × col_norm(kv_b_proj.weight, i)

where:
  - mean_activation[i] = E[|c_kv[:, :, i]|] over calibration data
  - col_norm[i]         = ||kv_b_proj.weight[:, i]||_2

This is the SmoothQuant-style combined score: a channel that is large
AND projects onto a high-norm direction in kv_b_proj will cause large
output perturbations when quantized.

Two methods are provided:
  - weight_only:   uses col_norm only. Fast, no calibration data needed.
  - activation:    uses mean_activation × col_norm. More accurate.

Usage:
    from ops.sensitivity import SensitivityAnalyzer

    analyzer = SensitivityAnalyzer(model, method="activation")
    scores = analyzer.run(calibration_samples, device="cuda")
    # scores: dict[int, Tensor]  layer_idx -> [kv_lora_rank] float32

    config = analyzer.make_quant_config(scores, fp8_fraction=0.8)
    # config: dict[int, Tensor]  layer_idx -> [kv_lora_rank] bool (True = FP8)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

# Precision level constants — must match kv_latent_cache_quantized.py
PREC_BF16 = 0
PREC_FP8  = 1
PREC_FP4  = 2


class SensitivityAnalyzer:
    """Compute per-channel sensitivity scores for c_kv across all MLA layers.

    Args:
        model:   The patched DeepSeek / Kimi model (post patch_kv_model).
        method:  "weight_only" or "activation".
                 weight_only — col norms of kv_b_proj only. No forward pass needed.
                 activation  — col norms × mean |c_kv| over calibration data.
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["weight_only", "activation"] = "activation",
    ) -> None:
        self.model = model
        self.method = method
        self._layers = self._find_mla_layers()

    def _find_mla_layers(self) -> dict[int, nn.Module]:
        """Return {layer_idx: attention_module} for all MLA layers."""
        layers: dict[int, nn.Module] = {}
        # DeepSeek-V2-Lite: model.model.layers[i].self_attn
        # Walk the module tree and find layers with kv_b_proj
        for name, module in self.model.named_modules():
            if hasattr(module, "kv_b_proj"):
                # Extract layer index from name (e.g. "model.layers.3.self_attn" -> 3)
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            idx = int(parts[i + 1])
                            layers[idx] = module
                        except ValueError:
                            pass
                        break
        if not layers:
            raise RuntimeError(
                "No MLA layers found (looked for modules with kv_b_proj). "
                "Make sure the model is a DeepSeek/Kimi MLA model."
            )
        print(f"[sensitivity] Found {len(layers)} MLA layers")
        return layers

    def _col_norms(self, layer_idx: int) -> Tensor:
        """L2 norm of each column of kv_b_proj.weight — shape [kv_lora_rank]."""
        attn = self._layers[layer_idx]
        W = attn.kv_b_proj.weight  # [out_dim, kv_lora_rank]
        return W.float().norm(dim=0)  # [kv_lora_rank]

    def weight_only_scores(self) -> dict[int, Tensor]:
        """Per-channel scores based on kv_b_proj column norms only.

        No forward pass required. Scores are in [kv_lora_rank] float32.
        """
        scores: dict[int, Tensor] = {}
        for idx in sorted(self._layers):
            scores[idx] = self._col_norms(idx).cpu()
        return scores

    @torch.no_grad()
    def activation_scores(
        self,
        calibration_samples: list[Tensor],
        device: str | torch.device = "cuda",
    ) -> dict[int, Tensor]:
        """Per-channel scores = mean |c_kv| × col_norm(kv_b_proj).

        Runs calibration_samples through the model with forward hooks on
        kv_a_layernorm output to collect c_kv statistics.

        Args:
            calibration_samples: List of [1, seq_len] LongTensors (from calibration_data.py).
            device:              Device to run inference on.

        Returns:
            dict[int, Tensor]: layer_idx -> [kv_lora_rank] sensitivity scores.
        """
        self.model.eval()
        self.model.to(device)

        num_layers = len(self._layers)
        kv_lora_rank = next(iter(self._layers.values())).kv_b_proj.weight.shape[1]

        # Accumulators: sum of |c_kv| and count, per layer
        sum_abs: dict[int, Tensor] = {i: torch.zeros(kv_lora_rank) for i in self._layers}
        count: dict[int, int] = {i: 0 for i in self._layers}

        hooks = []

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # output is c_kv (kv_a_norm): [B, S, kv_lora_rank]
                c_kv = output.detach().float()
                # mean over batch and sequence dims
                sum_abs[layer_idx] += c_kv.abs().mean(dim=(0, 1)).cpu()
                count[layer_idx] += 1
            return hook

        # Register hooks on kv_a_layernorm (the layer norm applied to c_kv)
        for idx, attn in self._layers.items():
            if hasattr(attn, "kv_a_layernorm"):
                h = attn.kv_a_layernorm.register_forward_hook(make_hook(idx))
                hooks.append(h)
            else:
                raise RuntimeError(
                    f"Layer {idx}: expected kv_a_layernorm attribute on attention module. "
                    f"Available attrs: {[a for a in dir(attn) if 'kv' in a.lower()]}"
                )

        print(f"[sensitivity] Running {len(calibration_samples)} calibration samples...")
        for sample in tqdm(calibration_samples, desc="calibrating"):
            input_ids = sample.to(device)
            try:
                self.model(input_ids=input_ids, use_cache=False)
            except Exception as e:
                # Remove hooks before re-raising
                for h in hooks:
                    h.remove()
                raise RuntimeError(f"Forward pass failed during calibration: {e}") from e

        for h in hooks:
            h.remove()

        # Compute final scores: mean_abs × col_norm
        scores: dict[int, Tensor] = {}
        for idx in sorted(self._layers):
            n = count[idx]
            if n == 0:
                raise RuntimeError(f"No activations collected for layer {idx} — hook may have failed.")
            mean_abs = sum_abs[idx] / n                  # [kv_lora_rank]
            col_norms = self._col_norms(idx).cpu()       # [kv_lora_rank]
            scores[idx] = mean_abs * col_norms            # element-wise product
        return scores

    def run(
        self,
        calibration_samples: list[Tensor] | None = None,
        device: str | torch.device = "cuda",
    ) -> dict[int, Tensor]:
        """Run sensitivity analysis with the configured method.

        Args:
            calibration_samples: Required if method == "activation".
            device:              Inference device.

        Returns:
            dict[int, Tensor]: layer_idx -> [kv_lora_rank] sensitivity scores.
        """
        if self.method == "weight_only":
            print("[sensitivity] Method: weight_only (column norms of kv_b_proj)")
            return self.weight_only_scores()
        elif self.method == "activation":
            if calibration_samples is None:
                raise ValueError("calibration_samples required for method='activation'")
            print("[sensitivity] Method: activation-weighted (mean |c_kv| × col_norm)")
            return self.activation_scores(calibration_samples, device=device)
        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'weight_only' or 'activation'.")

    def make_quant_config(
        self,
        scores: dict[int, Tensor],
        fp8_fraction: float = 0.8,
        global_threshold: float | None = None,
    ) -> dict[int, Tensor]:
        """Convert sensitivity scores into a per-layer boolean quantization mask.

        A channel with mask=True will be stored in FP8; mask=False stays BF16.

        Two modes:
          - fp8_fraction:     Keep the least sensitive `fp8_fraction` of channels
                              as FP8. E.g. 0.8 means 80% FP8, 20% BF16.
          - global_threshold: Channels with score below this absolute value go FP8.
                              If set, overrides fp8_fraction.

        Args:
            scores:           Output of run().
            fp8_fraction:     Fraction of channels to quantize to FP8 (0.0–1.0).
            global_threshold: Optional absolute sensitivity threshold.

        Returns:
            dict[int, Tensor]: layer_idx -> [kv_lora_rank] bool mask.
                               True  = quantize to FP8
                               False = keep in BF16
        """
        config: dict[int, Tensor] = {}

        all_scores = torch.cat(list(scores.values()))
        if global_threshold is None:
            # Compute per-fraction threshold from global score distribution
            k = int(fp8_fraction * len(all_scores))
            threshold = all_scores.kthvalue(k).values.item()
        else:
            threshold = global_threshold

        fp8_total = 0
        total = 0
        for idx in sorted(scores):
            mask = scores[idx] <= threshold  # True = FP8
            config[idx] = mask
            fp8_total += mask.sum().item()
            total += mask.numel()

        actual_fraction = fp8_total / total
        print(
            f"[sensitivity] Quant config: {fp8_total}/{total} channels FP8 "
            f"({actual_fraction:.1%}), threshold={threshold:.4f}"
        )
        return config

    def make_mixed_precision_config(
        self,
        scores: dict[int, Tensor],
        fp4_fraction: float = 0.5,
        fp8_fraction: float = 0.3,
    ) -> dict[int, Tensor]:
        """Convert sensitivity scores into a 3-level per-channel precision config.

        Channels are sorted by sensitivity (ascending). The least sensitive
        fp4_fraction go to FP4, the next fp8_fraction go to FP8, and the
        remaining (most sensitive) channels stay in BF16.

        Args:
            scores:       Output of run() — layer_idx -> [kv_lora_rank] scores.
            fp4_fraction: Fraction of channels to store in FP4 (0.0–1.0).
            fp8_fraction: Fraction of channels to store in FP8 (0.0–1.0).
                          fp4_fraction + fp8_fraction must be <= 1.0.

        Returns:
            dict[int, Tensor]: layer_idx -> [kv_lora_rank] uint8 tensor.
                               Values: 0=BF16, 1=FP8, 2=FP4
                               (constants PREC_BF16/FP8/FP4 above)
        """
        if fp4_fraction + fp8_fraction > 1.0:
            raise ValueError(
                f"fp4_fraction ({fp4_fraction}) + fp8_fraction ({fp8_fraction}) "
                f"must be <= 1.0"
            )

        all_scores = torch.cat(list(scores.values()))
        n = len(all_scores)
        sorted_scores, _ = all_scores.sort()

        # Determine score thresholds from global distribution
        k_fp4 = max(1, int(fp4_fraction * n))
        k_fp8 = max(0, int(fp8_fraction * n))

        fp4_threshold = sorted_scores[k_fp4 - 1].item()
        fp8_threshold = sorted_scores[k_fp4 + k_fp8 - 1].item() if k_fp8 > 0 else fp4_threshold

        config: dict[int, Tensor] = {}
        fp4_total = fp8_total = bf16_total = 0

        for idx in sorted(scores):
            s    = scores[idx]
            prec = torch.zeros(len(s), dtype=torch.uint8)
            prec[s <= fp4_threshold]                         = PREC_FP4
            prec[(s > fp4_threshold) & (s <= fp8_threshold)] = PREC_FP8
            # channels with s > fp8_threshold stay PREC_BF16 (0)
            config[idx] = prec

            fp4_total  += (prec == PREC_FP4).sum().item()
            fp8_total  += (prec == PREC_FP8).sum().item()
            bf16_total += (prec == PREC_BF16).sum().item()

        total = fp4_total + fp8_total + bf16_total
        print(
            f"[sensitivity] Mixed precision config: "
            f"FP4={fp4_total/total:.1%}  FP8={fp8_total/total:.1%}  BF16={bf16_total/total:.1%}  "
            f"(thresholds: FP4≤{fp4_threshold:.4f}, FP8≤{fp8_threshold:.4f})"
        )
        return config

    def save_scores(self, scores: dict[int, Tensor], path: str) -> None:
        """Save sensitivity scores to a .pt file for reuse."""
        torch.save({str(k): v for k, v in scores.items()}, path)
        print(f"[sensitivity] Scores saved to {path}")

    @staticmethod
    def load_scores(path: str) -> dict[int, Tensor]:
        """Load sensitivity scores previously saved with save_scores()."""
        raw = torch.load(path, map_location="cpu")
        return {int(k): v for k, v in raw.items()}

    def summary(self, scores: dict[int, Tensor]) -> None:
        """Print a per-layer summary of sensitivity score statistics."""
        print(f"\n{'Layer':>6}  {'min':>10}  {'p25':>10}  {'median':>10}  {'p75':>10}  {'max':>10}")
        print("-" * 62)
        for idx in sorted(scores):
            s = scores[idx]
            q = torch.quantile(s.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
            print(
                f"{idx:>6}  "
                f"{q[0].item():>10.4f}  "
                f"{q[1].item():>10.4f}  "
                f"{q[2].item():>10.4f}  "
                f"{q[3].item():>10.4f}  "
                f"{q[4].item():>10.4f}"
            )
        print()
