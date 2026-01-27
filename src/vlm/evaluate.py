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
