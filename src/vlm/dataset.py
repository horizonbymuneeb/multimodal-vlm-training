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


class SyntheticCaptionDataset(Dataset):
    """Dataset of rendered scenes with tokenised captions."""

    def __init__(
        self,
        n: int,
        vocab: Vocab,
        *,
        image_size: int = 16,
        max_len: int = 12,
        seed: int = 0,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if image_size < 8:
            raise ValueError("image_size must be at least 8")
        if max_len < 2:
            raise ValueError("max_len must be at least 2")
        self.n = int(n)
        self.vocab = vocab
        self.image_size = int(image_size)
        self.max_len = int(max_len)
        self.seed = int(seed)
        self.seeds = [seed * 100_000 + i for i in range(n)]

    def __len__(self) -> int:
        return self.n

    def spec_at(self, index: int) -> SceneSpec:
        return random_spec(self.seeds[index])

    def caption_at(self, index: int) -> str:
        return caption_for(self.spec_at(index))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.spec_at(index)
        image = render_scene(spec, self.image_size)
        ids = truncate(self.vocab.encode(caption_for(spec)), self.max_len, eos_id=self.vocab.eos_id)
        return {
            "image": torch.from_numpy(image),
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(len(ids), dtype=torch.long),
        }

    def captions(self) -> list[str]:
        return [self.caption_at(i) for i in range(self.n)]


def collate_captions(
    batch: Sequence[dict[str, torch.Tensor]],
    pad_id: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length captions into ``(N, T)`` ids plus a mask."""
    items = list(batch)
    if not items:
        raise ValueError("cannot collate an empty batch")
    images = torch.stack([item["image"] for item in items], dim=0)
    id_lists = [item["ids"].tolist() for item in items]
    padded, mask = pad_sequences(id_lists, pad_id=pad_id)
    return {
        "image": images,
        "ids": torch.tensor(padded, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
    }


def make_dataloader(
    n: int,
    vocab: Vocab,
    *,
    batch_size: int = 8,
    image_size: int = 16,
    max_len: int = 12,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader over the synthetic dataset with caption padding."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = SyntheticCaptionDataset(
        n, vocab, image_size=image_size, max_len=max_len, seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_captions(batch, pad_id=vocab.pad_id),
    )


def dataset_vocabulary(
    n: int,
    *,
    image_size: int = 16,
    seed: int = 0,
    min_freq: int = 1,
) -> Vocab:
    """Build the vocabulary that the dataset at ``seed`` will actually use."""
    from src.vlm.tokenizer import build_vocab

    captions = [caption_for(random_spec(seed * 100_000 + i)) for i in range(n)]
    return build_vocab(captions, min_freq=min_freq)


def caption_token_budget(captions: Iterable[str]) -> dict[str, float]:
    """Length statistics used to pick ``max_len`` instead of guessing."""
    lengths = [len(caption.split()) + 2 for caption in captions]  # +bos +eos
    lengths = list(lengths)
    if not lengths:
        raise ValueError("no captions")
    return {
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "mean": float(np.mean(lengths)),
    }
