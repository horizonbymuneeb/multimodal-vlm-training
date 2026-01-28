"""Synthetic scene and caption dataset tests."""

import numpy as np
import pytest
import torch

from src.vlm.dataset import (
    COLORS,
    POSITIONS,
    SHAPES,
    SceneSpec,
    caption_for,
    caption_token_budget,
    collate_captions,
    dataset_vocabulary,
    generate_sample,
    make_dataloader,
    random_spec,
    render_scene,
)
from src.vlm.tokenizer import UNK_TOKEN


def test_render_scene_paints_the_right_place_and_colour():
    spec = SceneSpec(shape="circle", color="red", position="top-left", seed=0)
    image = render_scene(spec, size=16)
    assert image.shape == (3, 16, 16)
    assert image.min() >= 0.0 and image.max() <= 1.0
    row, col = (int(POSITIONS["top-left"][0] * 16), int(POSITIONS["top-left"][1] * 16))
    assert image[0, row, col] > 0.5  # red channel high
    assert image[2, row, col] < 0.4  # blue channel low
    assert image[0, 0, 0] < 0.2  # background is dark


def test_render_scene_supports_every_shape_and_position():
    for shape in SHAPES:
        for position in POSITIONS:
            image = render_scene(
                SceneSpec(shape=shape, color="green", position=position, seed=1), size=8
            )
            assert image.shape == (3, 8, 8)
            assert np.isfinite(image).all()


def test_render_scene_second_shape_changes_pixels():
    single = render_scene(SceneSpec("circle", "red", "center", seed=2), size=12)
    double = render_scene(
        SceneSpec("circle", "red", "center", seed=2, second_shape="square", second_color="blue"),
        size=12,
    )
    assert not np.allclose(single, double)
