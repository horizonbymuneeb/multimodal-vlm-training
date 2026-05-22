"""Projection head tests: normalisation, similarity and alignment term."""

import pytest
import torch

from src.vlm.projection import (
    ProjectionHead,
    alignment_loss,
    build_projection,
    cosine_similarity_matrix,
    l2_normalize,
    project_features,
    projection_parameters,
)


def test_projection_output_is_normalised():
    head = build_projection(8, 4, seed=0)
    out = head(torch.randn(6, 8))
    assert out.shape == (6, 4)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)


def test_projection_can_skip_normalisation():
    head = ProjectionHead(8, 4, normalize=False)
    out = head(torch.randn(5, 8))
    assert not torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-3)


def test_l2_normalize_handles_zero_rows():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    out = l2_normalize(x)
    assert torch.isfinite(out).all()
    assert out[0].tolist() == [0.0, 0.0]
    assert out[1].tolist() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError):
        l2_normalize(x, eps=0.0)


def test_projection_validation():
    with pytest.raises(ValueError):
        ProjectionHead(0, 4)
    with pytest.raises(ValueError):
        ProjectionHead(4, 0)
    head = build_projection(8, 4)
    with pytest.raises(ValueError):
        head(torch.randn(4))


def test_cosine_similarity_matrix_properties():
    a = torch.randn(5, 6)
    sim = cosine_similarity_matrix(a, a)
    assert sim.shape == (5, 5)
    diagonal = torch.diagonal(sim)
    assert torch.allclose(diagonal, torch.ones(5), atol=1e-5)
    assert torch.allclose(sim, sim.transpose(0, 1), atol=1e-5)
    with pytest.raises(ValueError):
        cosine_similarity_matrix(torch.randn(3, 4), torch.randn(3, 5))
    with pytest.raises(ValueError):
        cosine_similarity_matrix(torch.randn(3), torch.randn(3))


def test_alignment_loss_is_zero_for_identical_embeddings():
    x = torch.randn(4, 6)
    assert alignment_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert alignment_loss(x, -x).item() == pytest.approx(4.0, abs=1e-5)
    with pytest.raises(ValueError):
        alignment_loss(torch.randn(2, 3), torch.randn(3, 3))


def test_seeded_projection_is_reproducible():
    a = build_projection(8, 4, seed=2)
    b = build_projection(8, 4, seed=2)
    c = build_projection(8, 4, seed=3)
    x = torch.randn(3, 8)
    assert torch.allclose(a(x), b(x))
    assert not torch.allclose(a(x), c(x))


def test_project_features_and_parameter_counts():
    head = build_projection(8, 4, hidden_dim=6)
    out = project_features(head, torch.randn(2, 8))
    assert out.shape == (2, 4)
    counts = projection_parameters(head)
    assert set(counts) == {"net.0", "net.3"}
    assert counts["net.0"] == 8 * 6 + 6
    assert counts["net.3"] == 6 * 4 + 4
