"""Loss tests: InfoNCE behaviour, padding-aware caption loss, loss mixture."""

import pytest
import torch
import torch.nn.functional as F

from src.vlm.losses import (
    batch_retrieval_accuracy,
    caption_accuracy,
    caption_cross_entropy,
    combine_losses,
    info_nce_loss,
    loss_report,
    symmetric_infonce,
    temperature_logits,
)
from src.vlm.projection import l2_normalize


def aligned_pair(n: int = 4, dim: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    base = l2_normalize(torch.randn(n, dim))
    return base, base.clone()


def test_temperature_logits_scale_and_validate():
    a, b = aligned_pair()
    logits = temperature_logits(a, b, 0.5)
    assert logits.shape == (4, 4)
    assert torch.allclose(torch.diagonal(logits), torch.full((4,), 2.0), atol=1e-5)
    with pytest.raises(ValueError):
        temperature_logits(a, b, 0.0)
    with pytest.raises(ValueError):
        temperature_logits(a, b[:3], 0.5)


def test_info_nce_is_small_for_aligned_pairs():
    a, b = aligned_pair()
    loss = info_nce_loss(a, b, temperature=0.07)
    assert loss.item() < 0.05
    shuffled = info_nce_loss(a, torch.roll(b, shifts=1, dims=0), temperature=0.07)
    assert shuffled.item() > loss.item()
