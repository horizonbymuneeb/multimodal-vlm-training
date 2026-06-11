"""Vision tower tests: shapes, freezing and deterministic initialisation."""

import pytest
import torch

from src.vlm.vision import (
    ConvStem,
    VisionConfig,
    VisionEncoder,
    build_vision_encoder,
    count_parameters,
    encode_images,
    freeze_parameters,
    pooled_feature_dim,
    set_trainable,
    unfreeze_parameters,
)


def test_conv_stem_downsamples_by_eight():
    stem = ConvStem(channels=4, in_channels=3)
    out = stem(torch.zeros(2, 3, 16, 16))
    assert out.shape == (2, 16, 2, 2)
    with pytest.raises(ValueError):
        stem(torch.zeros(3, 16, 16))


def test_encoder_output_shape_and_finiteness():
    encoder = build_vision_encoder(out_dim=12, channels=4, image_size=16)
    out = encoder(torch.randn(5, 3, 16, 16))
    assert out.shape == (5, 12)
    assert torch.isfinite(out).all()


def test_encoder_feature_map_shape():
    encoder = build_vision_encoder(out_dim=8, channels=4, image_size=16)
    features = encoder.feature_map(torch.zeros(1, 3, 16, 16))
    assert features.shape == (1, 16, 2, 2)


def test_encoder_is_frozen_by_default():
    encoder = build_vision_encoder(out_dim=8, channels=4)
    assert count_parameters(encoder, trainable_only=True) == 0
    assert count_parameters(encoder) > 0
    assert unfreeze_parameters(encoder) == len(list(encoder.parameters()))
    assert count_parameters(encoder, trainable_only=True) == count_parameters(encoder)


def test_set_trainable_counts_parameters():
    encoder = build_vision_encoder(out_dim=8, channels=4, freeze=False)
    touched = set_trainable(encoder, False)
    assert touched == len(list(encoder.parameters()))
    assert freeze_parameters(encoder) == touched


def test_encode_images_runs_without_gradients():
    encoder = build_vision_encoder(out_dim=8, channels=4, freeze=False)
    images = torch.randn(2, 3, 16, 16, requires_grad=True)
    features = encode_images(encoder, images)
    assert not features.requires_grad
    assert encoder.training is False


def test_seeded_encoders_agree():
    a = build_vision_encoder(out_dim=8, channels=4, seed=3)
    b = build_vision_encoder(out_dim=8, channels=4, seed=3)
    c = build_vision_encoder(out_dim=8, channels=4, seed=4)
    x = torch.randn(2, 3, 16, 16)
    assert torch.allclose(a(x), b(x))
    assert not torch.allclose(a(x), c(x))


def test_vision_config_validation():
    with pytest.raises(ValueError):
        VisionConfig(out_dim=0)
    with pytest.raises(ValueError):
        VisionConfig(in_channels=2)
    with pytest.raises(ValueError):
        VisionConfig(image_size=4)
    assert pooled_feature_dim(VisionConfig(out_dim=32)) == 32


def test_encoder_handles_other_image_sizes():
    encoder = build_vision_encoder(out_dim=6, channels=4)
    for size in (8, 32):
        out = encoder(torch.zeros(1, 3, size, size))
        assert out.shape == (1, 6)
    assert isinstance(encoder, VisionEncoder)
