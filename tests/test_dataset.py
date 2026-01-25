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
