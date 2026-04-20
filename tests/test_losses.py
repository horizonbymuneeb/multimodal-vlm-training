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


def test_info_nce_single_pair_is_zero():
    a = torch.randn(1, 6)
    loss = info_nce_loss(a, a.clone(), temperature=0.07)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_symmetric_infonce_is_the_average_of_both_directions():
    a, b = aligned_pair()
    logits = temperature_logits(a, b, 0.2)
    targets = torch.arange(4)
    expected = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    got = symmetric_infonce(a, b, temperature=0.2)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)
    with pytest.raises(ValueError):
        symmetric_infonce(torch.randn(1, 4), torch.randn(1, 4))


def test_caption_cross_entropy_matches_manual_shift():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.tensor([[1, 2, 3, 4, 5], [1, 6, 0, 0, 0]])
    manual = F.cross_entropy(logits[:, :-1, :].reshape(-1, 7), targets[:, 1:].reshape(-1), ignore_index=0)
    got = caption_cross_entropy(logits, targets, pad_id=0)
    assert got.item() == pytest.approx(manual.item(), rel=1e-6)
