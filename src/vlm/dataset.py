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


def random_spec(seed: int) -> SceneSpec:
    """Deterministically draw a scene spec from a seed."""
    rng = np.random.default_rng(seed)
    shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
    color = list(COLORS)[int(rng.integers(0, len(COLORS)))]
    position = list(POSITIONS)[int(rng.integers(0, len(POSITIONS)))]
    second = None
    second_color = None
    if rng.random() < 0.35:
        second = SHAPES[int(rng.integers(0, len(SHAPES)))]
        second_color = list(COLORS)[int(rng.integers(0, len(COLORS)))]
    return SceneSpec(
        shape=str(shape),
        color=str(color),
        position=str(position),
        seed=seed,
        second_shape=str(second) if second else None,
        second_color=str(second_color) if second_color else None,
    )


def caption_for(spec: SceneSpec) -> str:
    """The sentence that describes a scene, in a fixed word order."""
    text = f"a {spec.color} {spec.shape} at {spec.position}"
    if spec.second_shape and spec.second_color:
        text += f" and a {spec.second_color} {spec.second_shape}"
    return text


def _paint(
    canvas: np.ndarray,
    shape: str,
    color: tuple[float, float, float],
    center: tuple[float, float],
    radius: float,
) -> None:
    height, width = canvas.shape[1:]
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = center[0] * height, center[1] * width
    if shape == "circle":
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    elif shape == "square":
        mask = (np.abs(yy - cy) <= radius) & (np.abs(xx - cx) <= radius)
    elif shape == "triangle":
        # Upright triangle: below the apex and inside the two sloping edges.
        mask = (yy >= cy - radius) & (yy <= cy + radius) & (
            np.abs(xx - cx) <= (yy - (cy - radius)) * 0.6
        )
    else:  # pragma: no cover - guarded by SceneSpec
        raise ValueError(f"unknown shape {shape!r}")
    for channel, value in enumerate(color):
        canvas[channel][mask] = value


def render_scene(spec: SceneSpec, size: int = 16) -> np.ndarray:
    """Render a scene to a ``(3, size, size)`` float array in ``[0, 1]``."""
    if size < 8:
        raise ValueError("size must be at least 8")
    canvas = np.full((3, size, size), 0.05, dtype=np.float32)
    radius = max(1.5, size * 0.16)
    _paint(canvas, spec.shape, COLORS[spec.color], POSITIONS[spec.position], radius)
    if spec.second_shape and spec.second_color:
        _paint(canvas, spec.second_shape, COLORS[spec.second_color], (0.5, 0.5), radius * 0.8)
    # Light per-pixel noise so identical scenes are not byte-identical batches.
    rng = np.random.default_rng(spec.seed + 991)
    canvas += rng.normal(0.0, 0.01, size=canvas.shape).astype(np.float32)
    return np.clip(canvas, 0.0, 1.0)


def generate_sample(seed: int, size: int = 16) -> tuple[np.ndarray, str, SceneSpec]:
    """Image, caption and the spec that produced them."""
    spec = random_spec(seed)
    return render_scene(spec, size), caption_for(spec), spec
