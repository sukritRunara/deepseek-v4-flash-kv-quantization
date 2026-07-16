"""Tiny randomly-initialized DeepSeek-V4 model factory.

Shared by tools/inspect_v4_cache.py and tests/test_v4_cache_semantics.py so that
every consumer exercises the exact same configuration. Requires only the pinned
vendor/transformers source checkout — no weights, no network.

The configuration covers all three V4 attention layer types and uses small
compression rates / window so that boundary behavior is reachable in a few
dozen tokens:

    layer 0: sliding_attention                (window only)
    layer 1: compressed_sparse_attention m=4  (overlap compressor + indexer)
    layer 2: heavily_compressed_attention m'=8 (non-overlap compressor)

The real V4-Flash uses window=128, m=4, m'=128; rates are read from the config
everywhere, never hardcoded, so shrinking m' does not change code paths.
"""

from __future__ import annotations

import torch
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

TINY_V4_KWARGS: dict = {
    "vocab_size": 128,
    "hidden_size": 64,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "partial_rotary_factor": 8 / 32,  # qk_rope_head_dim = 8
    "q_lora_rank": 32,
    "o_groups": 2,
    "o_lora_rank": 16,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "layer_types": [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ],
    # HCA rate shrunk from the production 128 to 8 so compression boundaries are
    # testable at tiny sequence lengths. CSA keeps the production rate m=4.
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    "mlp_layer_types": ["hash_moe", "moe", "moe"],
    "sliding_window": 8,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 3,
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 2,
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 512,
}


def tiny_v4_config(**overrides) -> DeepseekV4Config:
    """Build the tiny V4 config; keyword overrides replace TINY_V4_KWARGS entries."""
    kwargs = {**TINY_V4_KWARGS, **overrides}
    return DeepseekV4Config(**kwargs)


def build_tiny_model(
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    **config_overrides,
) -> DeepseekV4ForCausalLM:
    """Deterministically construct a randomly-initialized tiny V4 causal LM in eval mode."""
    torch.manual_seed(seed)
    config = tiny_v4_config(**config_overrides)
    model = DeepseekV4ForCausalLM(config)
    return model.to(device=device, dtype=dtype).eval()


def deterministic_input_ids(
    batch: int, seq_len: int, vocab_size: int = TINY_V4_KWARGS["vocab_size"], seed: int = 1
) -> torch.Tensor:
    """Reproducible unpadded input ids (no left padding — see V4_CACHE_ARCHITECTURE.md §6.2)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, seq_len), generator=generator)
