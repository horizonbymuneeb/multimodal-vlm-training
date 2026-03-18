"""The projection head that maps vision features into the text space.

This is the only bridge between the frozen encoder and the decoder, so it is
kept deliberately small: two linear layers with a GELU in between and an
L2-normalised output.  Normalising here (rather than in the loss) means the
contrastive loss works on cosine similarity no matter which side produced the
vector, and retrieval can reuse the same embeddings.
"""

from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """``in_dim -> hidden -> out_dim`` with optional dropout and normalisation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        hidden = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.normalize = normalize

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"expected (N, D) features, got {tuple(features.shape)}")
        projected = self.net(features)
        return l2_normalize(projected) if self.normalize else projected


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise rows to unit length, safe for zero rows."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm
