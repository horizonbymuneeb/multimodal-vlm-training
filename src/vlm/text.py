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


def scaled_dot_product(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``softmax(QK^T / sqrt(d)) V`` with an optional additive-free bool mask."""
    if query.ndim != 4:
        raise ValueError(f"query must be (N, H, T, D), got {tuple(query.shape)}")
    if key.shape[-1] != query.shape[-1]:
        raise ValueError("query and key must share the head dimension")
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if mask is not None:
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match scores {tuple(scores.shape)}"
            )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention with separate q/k/v projections."""

    def __init__(self, dim: int, n_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        n, t, _ = x.shape
        return x.view(n, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        memory: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._split(self.q_proj(x))
        source = x if memory is None else memory
        key = self._split(self.k_proj(source))
        value = self._split(self.v_proj(source))
        out, weights = scaled_dot_product(query, key, value, mask)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dim)
        return self.out_proj(self.dropout(out)), weights
