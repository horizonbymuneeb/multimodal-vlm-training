"""Synthetic image-caption pairs.

Downloading COCO is not an option for a test suite that must run offline in
seconds, so images are drawn from a tiny scene grammar: one or two shapes, a
colour and a position.  The caption is generated from the same scene spec, which
means retrieval has a real right answer and captioning has a real target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.vlm.tokenizer import BOS_ID, EOS_ID, PAD_ID, Vocab, pad_sequences, truncate

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.9, 0.15, 0.15),
    "green": (0.15, 0.8, 0.25),
    "blue": (0.2, 0.35, 0.95),
    "yellow": (0.95, 0.85, 0.15),
    "purple": (0.6, 0.2, 0.8),
}

SHAPES: tuple[str, ...] = ("circle", "square", "triangle")

POSITIONS: dict[str, tuple[float, float]] = {
    "top-left": (0.28, 0.28),
    "top-right": (0.28, 0.72),
    "bottom-left": (0.72, 0.28),
    "bottom-right": (0.72, 0.72),
    "center": (0.5, 0.5),
}


@dataclass(frozen=True)
class SceneSpec:
    """One scene: a shape, a colour and a position, plus an optional second shape."""

    shape: str
    color: str
    position: str
    seed: int = 0
    second_shape: str | None = None
    second_color: str | None = None

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}")
        if self.color not in COLORS:
            raise ValueError(f"unknown colour {self.color!r}")
        if self.position not in POSITIONS:
            raise ValueError(f"unknown position {self.position!r}")
        if self.second_shape is not None and self.second_shape not in SHAPES:
            raise ValueError(f"unknown second shape {self.second_shape!r}")
        if self.second_color is not None and self.second_color not in COLORS:
            raise ValueError(f"unknown second colour {self.second_color!r}")
