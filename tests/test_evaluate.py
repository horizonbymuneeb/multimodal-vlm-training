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


def distractor_gallery():
    """Three queries where one gold item is beaten by a closer distractor."""
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    gallery = torch.tensor(
        [
            [1.0, 0.02],  # distractor: closer to query 0 than its gold item
            [0.9, 0.1],   # gold for query 0 -> rank 1
            [0.0, 1.0],   # gold for query 1 -> rank 0
            [1.0, 1.0],   # gold for query 2 -> rank 0
        ]
    )
    return query, gallery, [1, 2, 3]


def test_rank_matrix_counts_closer_items():
    query, gallery, targets = distractor_gallery()
    assert rank_matrix(query, gallery, targets).tolist() == [1, 0, 0]
    # Without explicit targets the paired 1:1 mapping is required.
    with pytest.raises(ValueError):
        rank_matrix(query, gallery)


def test_retrieval_at_k_and_mrr_with_distractors():
    query, gallery, targets = distractor_gallery()
    assert retrieval_at_k(query, gallery, 1, targets=targets) == pytest.approx(2 / 3)
    assert retrieval_at_k(query, gallery, 2, targets=targets) == pytest.approx(1.0)
    assert mean_reciprocal_rank(query, gallery, targets) == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    with pytest.raises(ValueError):
        retrieval_at_k(query, gallery, 1, targets=[0, 1])
    with pytest.raises(ValueError):
        retrieval_at_k(query, gallery, 1, targets=[0, 1, 9])
