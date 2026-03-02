"""Evaluation tests: retrieval@k, MRR, token accuracy and bootstrap CI."""

import pytest
import torch

from src.vlm.evaluate import (
    bootstrap_retrieval_ci,
    chance_level_retrieval,
    embed_loader,
    evaluate_all,
    evaluate_captioning,
    evaluate_retrieval,
    mean_reciprocal_rank,
    rank_matrix,
    retrieval_at_k,
    summarise_metrics,
    token_accuracy,
)


def aligned(n: int = 6, dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    base = torch.nn.functional.normalize(torch.randn(n, dim), dim=-1)
    return base, base.clone()


def test_rank_matrix_identity_for_aligned_embeddings():
    a, b = aligned()
    ranks = rank_matrix(a, b)
    assert ranks.tolist() == [0] * 6
    with pytest.raises(ValueError):
        rank_matrix(a, b[:3])
    with pytest.raises(ValueError):
        rank_matrix(torch.randn(3), torch.randn(3))


def test_retrieval_at_k_perfect_for_aligned_embeddings():
    a, b = aligned(4)
    assert retrieval_at_k(a, b, 1) == pytest.approx(1.0)
    assert mean_reciprocal_rank(a, b) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        retrieval_at_k(a, b, 0)
