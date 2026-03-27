"""The text decoder: causal self-attention plus cross-attention to the image.

Both attention paths are written out (scaled dot-product with an explicit mask)
instead of using ``nn.MultiheadAttention``, because the mask semantics are the
part of a decoder that is easy to get subtly wrong and the tests pin them:
token *t* may attend to tokens ``<= t`` and to every image feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def causal_mask(length: int) -> torch.Tensor:
    """Boolean ``(T, T)`` mask where ``True`` means "may attend"."""
    if length <= 0:
        raise ValueError("length must be positive")
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def positional_encoding(length: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal encodings, shape ``(length, dim)``."""
    if length <= 0 or dim <= 0:
        raise ValueError("length and dim must be positive")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding
