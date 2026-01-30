"""Evaluation: retrieval@k on the contrastive side, token accuracy on captions.

Retrieval is measured in both directions because they fail differently: a model
that ranks captions for an image well can still fail to pick the right image
for a caption when several images share a colour.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from src.vlm.losses import caption_accuracy
from src.vlm.projection import cosine_similarity_matrix, l2_normalize


def _gold_indices(n: int, targets: Sequence[int] | None, gallery_size: int) -> torch.Tensor:
    """Index of the correct gallery item per query (identity by default)."""
    if targets is None:
        return torch.arange(n)
    indices = torch.as_tensor(list(targets), dtype=torch.long)
    if indices.numel() != n:
        raise ValueError(f"targets must have {n} entries, got {indices.numel()}")
    if int(indices.min()) < 0 or int(indices.max()) >= gallery_size:
        raise ValueError("targets contain an index outside the gallery")
    return indices


def rank_matrix(
    query: torch.Tensor,
    gallery: torch.Tensor,
    targets: Sequence[int] | None = None,
) -> torch.Tensor:
    """For each query row, the rank of the matching gallery row (0 = best).

    The rank is the number of gallery items with a *strictly* higher similarity
    than the correct one.  That makes ties count as hits, which is the standard
    convention for retrieval metrics: with duplicate images in a gallery, a
    correct answer that ties for first should not be reported as a miss.
    """
    if query.ndim != 2 or gallery.ndim != 2:
        raise ValueError("query and gallery must be 2-D")
    if targets is None and query.shape[0] != gallery.shape[0]:
        raise ValueError(
            "paired retrieval needs the same number of queries and gallery items; "
            "pass targets= for a gallery of a different size"
        )
    with torch.no_grad():
        sim = cosine_similarity_matrix(query, gallery)
        gold = _gold_indices(sim.shape[0], targets, sim.shape[1])
        gold_scores = sim.gather(1, gold.unsqueeze(1)).squeeze(1).unsqueeze(1)
        ranks = (sim > gold_scores).sum(dim=1)
    return ranks


def retrieval_at_k(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k: int = 1,
    *,
    targets: Sequence[int] | None = None,
) -> float:
    """Fraction of queries whose match is in the top ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranks = rank_matrix(query, gallery, targets)
    return float((ranks < k).float().mean().item())


def mean_reciprocal_rank(
    query: torch.Tensor,
    gallery: torch.Tensor,
    targets: Sequence[int] | None = None,
) -> float:
    """Mean of ``1 / (rank + 1)`` over queries."""
    ranks = rank_matrix(query, gallery, targets)
    return float((1.0 / (ranks + 1.0)).mean().item())


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int = 0,
) -> float:
    """Teacher-forced token accuracy, padding excluded."""
    return caption_accuracy(logits, targets, pad_id=pad_id)
