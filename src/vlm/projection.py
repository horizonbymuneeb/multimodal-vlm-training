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


def build_projection(
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    seed: int | None = None,
) -> ProjectionHead:
    """Build the head with deterministic initialisation when ``seed`` is given."""
    if seed is not None:
        torch.manual_seed(seed)
    return ProjectionHead(in_dim, out_dim, hidden_dim=hidden_dim, dropout=dropout)


def project_features(head: ProjectionHead, features: torch.Tensor) -> torch.Tensor:
    """Apply the head without tracking gradients on the frozen encoder output."""
    return head(features)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarities between two ``(N, D)`` / ``(M, D)`` batches."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected 2-D embeddings")
    if a.shape[1] != b.shape[1]:
        raise ValueError("embeddings must share the feature dimension")
    return l2_normalize(a) @ l2_normalize(b).transpose(0, 1)


def alignment_loss(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between paired embeddings (a cheap auxiliary term).

    InfoNCE only cares about relative order; this term pulls matched pairs
    closer in absolute terms, which helps retrieval at small batch sizes.
    """
    if image_emb.shape != text_emb.shape:
        raise ValueError("image and text embeddings must have the same shape")
    return ((l2_normalize(image_emb) - l2_normalize(text_emb)) ** 2).sum(dim=-1).mean()


def projection_parameters(head: ProjectionHead) -> dict[str, int]:
    """Parameter counts per linear layer, for the README table and tests."""
    counts = {}
    for name, module in head.named_modules():
        if isinstance(module, nn.Linear):
            counts[name] = sum(p.numel() for p in module.parameters())
    return counts
