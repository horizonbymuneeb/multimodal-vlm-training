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


def caption_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int = 0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Teacher-forced next-token cross-entropy, ignoring padding.

    ``logits`` is ``(N, T, V)`` and ``targets`` is ``(N, T)``: position ``t``
    predicts ``targets[t + 1]``, so the last logit and the first target are
    dropped.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (N, T, V), got {tuple(logits.shape)}")
    if targets.ndim != 2 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must be (N, T) with the same batch size")
    if targets.shape[1] > logits.shape[1]:
        raise ValueError("targets are longer than the logits sequence")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_targets = targets[:, 1:].reshape(-1)
    return F.cross_entropy(
        shift_logits,
        shift_targets,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def caption_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int = 0,
) -> float:
    """Fraction of non-pad next-token predictions that are correct."""
    if logits.ndim != 3:
        raise ValueError("logits must be (N, T, V)")
    predictions = logits[:, :-1, :].argmax(dim=-1)
    gold = targets[:, 1:]
    mask = gold != pad_id
    if mask.sum() == 0:
        return 0.0
    correct = (predictions == gold) & mask
    return float(correct.sum().item() / mask.sum().item())


def combine_losses(
    contrastive: torch.Tensor,
    caption: torch.Tensor,
    *,
    caption_weight: float = 0.5,
) -> torch.Tensor:
    """Weighted sum of the two objectives."""
    if caption_weight < 0:
        raise ValueError("caption_weight must be non-negative")
    return contrastive + caption_weight * caption


def batch_retrieval_accuracy(image_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
    """Fraction of pairs whose best match is the correct one (batch of 1 = 1.0)."""
    if image_emb.shape[0] != text_emb.shape[0]:
        raise ValueError("batches must match")
    with torch.no_grad():
        sim = l2_normalize(image_emb) @ l2_normalize(text_emb).transpose(0, 1)
        predicted = sim.argmax(dim=1)
        gold = torch.arange(sim.shape[0], device=sim.device)
        return float((predicted == gold).float().mean().item())


def loss_report(
    contrastive: float,
    caption: float,
    *,
    caption_weight: float = 0.5,
) -> dict[str, float]:
    """The numbers logged per step, in one place so train/eval agree."""
    return {
        "contrastive": float(contrastive),
        "caption": float(caption),
        "total": float(contrastive + caption_weight * caption),
    }
