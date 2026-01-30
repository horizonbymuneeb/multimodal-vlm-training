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


@torch.no_grad()
def embed_loader(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | list]:
    """Collect image embeddings, text embeddings and gold ids for one pass."""
    image_embs: list[torch.Tensor] = []
    text_embs: list[torch.Tensor] = []
    all_ids: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    for batch in loader:
        out = model(batch["image"], batch["ids"], mask=batch["mask"])
        image_embs.append(out["image_emb"])
        text_embs.append(out["text_emb"])
        all_ids.append(batch["ids"])
        all_logits.append(out["logits"])
    if not image_embs:
        raise ValueError("loader produced no batches")
    # Batches are padded to their own longest caption, so pad everything to the
    # widest sequence before concatenating.
    import torch.nn.functional as F

    width = max(ids.shape[1] for ids in all_ids)
    ids_padded = [F.pad(ids, (0, width - ids.shape[1]), value=0) for ids in all_ids]
    logits_padded = [
        F.pad(logits, (0, 0, 0, width - logits.shape[1])) for logits in all_logits
    ]
    return {
        "image_emb": torch.cat(image_embs, dim=0),
        "text_emb": torch.cat(text_embs, dim=0),
        "ids": torch.cat(ids_padded, dim=0),
        "logits": torch.cat(logits_padded, dim=0),
    }


def evaluate_retrieval(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    k_values: Sequence[int] = (1, 3),
) -> dict[str, float]:
    """Retrieval@k in both directions on a fixed dataset."""
    if not k_values:
        raise ValueError("k_values must not be empty")
    data = embed_loader(model, loader)
    image_emb = data["image_emb"]
    text_emb = data["text_emb"]
    metrics: dict[str, float] = {}
    for k in k_values:
        if k <= 0:
            raise ValueError("k values must be positive")
        metrics[f"i2t@{k}"] = retrieval_at_k(image_emb, text_emb, k)
        metrics[f"t2i@{k}"] = retrieval_at_k(text_emb, image_emb, k)
    metrics["i2t_mrr"] = mean_reciprocal_rank(image_emb, text_emb)
    metrics["t2i_mrr"] = mean_reciprocal_rank(text_emb, image_emb)
    return metrics
