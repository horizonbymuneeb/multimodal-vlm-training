"""Losses: hand-written InfoNCE, captioning cross-entropy, and their mixture.

InfoNCE is implemented from the definition (cosine logits, softmax over the
batch, cross-entropy against the identity) rather than calling a library helper,
because the temperature scaling and the in-batch negatives are the two things
that make contrastive training work and both are worth being explicit about.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.vlm.projection import l2_normalize


def temperature_logits(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cosine similarity scaled by ``1 / temperature``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_emb.shape != text_emb.shape:
        raise ValueError(
            f"image and text batches must match, got {tuple(image_emb.shape)} and "
            f"{tuple(text_emb.shape)}"
        )
    return (l2_normalize(image_emb) @ l2_normalize(text_emb).transpose(0, 1)) / temperature


def info_nce_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Image-to-text InfoNCE: each image must rank its own caption first.

    ``batch_size`` in-batch negatives are used, so the loss is only meaningful
    with at least two pairs; a batch of one has no negatives and the loss is
    exactly zero by construction.
    """
    logits = temperature_logits(image_emb, text_emb, temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def symmetric_infonce(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Average of image-to-text and text-to-image InfoNCE."""
    if image_emb.shape[0] < 2:
        raise ValueError("symmetric InfoNCE needs at least two pairs")
    logits = temperature_logits(image_emb, text_emb, temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    forward = F.cross_entropy(logits, targets)
    backward = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (forward + backward)
