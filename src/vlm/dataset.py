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
