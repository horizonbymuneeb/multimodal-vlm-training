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


def test_render_scene_rejects_tiny_images():
    with pytest.raises(ValueError):
        render_scene(SceneSpec("circle", "red", "center"), size=4)


def test_caption_for_is_deterministic_text():
    spec = SceneSpec("triangle", "blue", "bottom-right", seed=3)
    assert caption_for(spec) == "a blue triangle at bottom-right"
    spec2 = SceneSpec(
        "circle", "red", "center", seed=3, second_shape="square", second_color="yellow"
    )
    assert caption_for(spec2) == "a red circle at center and a yellow square"


def test_scene_spec_validation():
    with pytest.raises(ValueError):
        SceneSpec("hexagon", "red", "center")
    with pytest.raises(ValueError):
        SceneSpec("circle", "teal", "center")
    with pytest.raises(ValueError):
        SceneSpec("circle", "red", "middle")
    with pytest.raises(ValueError):
        SceneSpec("circle", "red", "center", second_shape="hexagon")


def test_random_spec_is_deterministic():
    a = random_spec(11)
    b = random_spec(11)
    c = random_spec(12)
    assert a == b
    assert a != c
    assert a.color in COLORS and a.shape in SHAPES


def test_generate_sample_returns_consistent_caption():
    image, caption, spec = generate_sample(5, size=10)
    assert image.shape == (3, 10, 10)
    assert caption == caption_for(spec)


def test_dataset_items_and_captions(vocab):
    from src.vlm.dataset import SyntheticCaptionDataset

    dataset = SyntheticCaptionDataset(6, vocab, image_size=8, max_len=14, seed=3)
    assert len(dataset) == 6
    item = dataset[0]
    assert item["image"].shape == (3, 8, 8)
    assert item["ids"].dtype == torch.long
    assert int(item["ids"][0]) == vocab.bos_id
    assert int(item["ids"][-1]) == vocab.eos_id
    assert dataset.captions()[0] == caption_for(dataset.spec_at(0))
    with pytest.raises(ValueError):
        SyntheticCaptionDataset(0, vocab)
    with pytest.raises(ValueError):
        SyntheticCaptionDataset(4, vocab, image_size=4)
