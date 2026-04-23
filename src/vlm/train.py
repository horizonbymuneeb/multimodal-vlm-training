"""Training loop for the toy VLM.

One step optimises the sum of a symmetric InfoNCE term (image <-> caption) and a
teacher-forced captioning term.  Only the projection head and the LoRA adapters
are in the optimiser, which is why the loop can run on CPU in seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import yaml

from src.vlm.dataset import dataset_vocabulary, make_dataloader
from src.vlm.evaluate import evaluate_all, summarise_metrics
from src.vlm.losses import combine_losses, symmetric_infonce, caption_cross_entropy
from src.vlm.model import VLMConfig, build_model, parameter_summary, trainable_parameters


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and objective settings."""

    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-2
    weight_decay: float = 0.0
    temperature: float = 0.07
    caption_weight: float = 0.5
    seed: int = 5
    grad_clip: float = 1.0
    image_size: int = 16
    max_len: int = 12

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for in-batch negatives")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.caption_weight < 0:
            raise ValueError("caption_weight must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


def set_seed(seed: int) -> None:
    """Seed NumPy and Torch for reproducible CPU runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
