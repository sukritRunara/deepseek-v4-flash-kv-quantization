"""
src/calibration_data.py

C4 calibration data loader for latent KV cache quantization.

Loads a small subset of the C4 English corpus, tokenizes to a fixed
sequence length, and returns an iterable of input_ids tensors ready
for forward-pass hooking.

Usage:
    from src.calibration_data import load_calibration_data

    batches = load_calibration_data(
        tokenizer,
        n_samples=128,
        seq_len=512,
        seed=42,
    )
    for input_ids in batches:          # [1, seq_len] on CPU
        outputs = model(input_ids.to(device))
"""

from __future__ import annotations

import random
from typing import Iterator

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def load_calibration_data(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 128,
    seq_len: int = 512,
    seed: int = 42,
    dataset_name: str = "allenai/c4",
    dataset_split: str = "train",
    text_column: str = "text",
    streaming: bool = True,
) -> list[Tensor]:
    """Return a list of tokenized calibration samples.

    Each element is a LongTensor of shape [1, seq_len] (batch=1).
    Sequences are formed by concatenating documents until seq_len tokens
    are available, then slicing — no padding, no truncation mid-word.

    Args:
        tokenizer:    HuggingFace tokenizer matching the target model.
        n_samples:    Number of calibration sequences to return.
        seq_len:      Token length of each sequence.
        seed:         Random seed for reproducibility.
        dataset_name: HuggingFace dataset identifier.
        dataset_split: Dataset split to stream from.
        text_column:  Name of the text field in the dataset.
        streaming:    If True, stream the dataset (avoids full download).

    Returns:
        List of [1, seq_len] LongTensors on CPU.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required for calibration data. "
            "Install with: pip install datasets"
        ) from e

    random.seed(seed)

    print(f"[calibration] Loading {dataset_name} ({dataset_split}), streaming={streaming}")
    dataset = load_dataset(
        dataset_name,
        "en",
        split=dataset_split,
        streaming=streaming,
        trust_remote_code=True,
    )

    token_buffer: list[int] = []
    samples: list[Tensor] = []
    needed = n_samples * seq_len + seq_len  # a little extra to avoid short-falls

    for doc in dataset:
        if len(token_buffer) >= needed:
            break
        text = doc[text_column]
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(ids)

    if len(token_buffer) < n_samples * seq_len:
        raise RuntimeError(
            f"Not enough tokens in dataset after streaming "
            f"({len(token_buffer)} < {n_samples * seq_len}). "
            f"Increase the dataset size or reduce n_samples/seq_len."
        )

    # Randomly sample non-overlapping windows
    max_start = len(token_buffer) - seq_len
    starts = random.sample(range(max_start), n_samples)
    starts.sort()

    for start in starts:
        window = token_buffer[start : start + seq_len]
        samples.append(torch.tensor([window], dtype=torch.long))

    print(f"[calibration] Prepared {len(samples)} sequences × {seq_len} tokens")
    return samples


def calibration_iterator(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 128,
    seq_len: int = 512,
    seed: int = 42,
    **kwargs,
) -> Iterator[Tensor]:
    """Convenience generator wrapper around load_calibration_data."""
    samples = load_calibration_data(tokenizer, n_samples=n_samples, seq_len=seq_len, seed=seed, **kwargs)
    yield from samples