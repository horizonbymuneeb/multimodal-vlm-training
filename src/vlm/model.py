"""The assembled model: frozen vision tower -> projection -> LoRA decoder.

The composition is the whole point of the project, so the wiring is explicit:

1. the vision encoder is frozen and produces pooled features;
2. the projection head maps those features into the text embedding width;
3. the same projected vector is passed to the decoder as cross-attention
   memory, so captions are actually conditioned on the image;
4. only the projection and the LoRA adapters receive gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from src.vlm.lora import (
    apply_lora,
    count_trainable,
    iter_lora_modules,
    mark_only_lora_trainable,
)
from src.vlm.projection import ProjectionHead, build_projection
from src.vlm.text import TextConfig, TextDecoder
from src.vlm.vision import VisionConfig, VisionEncoder, build_vision_encoder


@dataclass(frozen=True)
class VLMConfig:
    """Everything needed to rebuild the model from a checkpoint."""

    vocab_size: int
    vision_dim: int = 48
    text_dim: int = 48
    n_heads: int = 4
    n_layers: int = 1
    max_len: int = 12
    dropout: float = 0.0
    image_size: int = 16
    vision_channels: int = 16
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size looks too small")
        if self.text_dim % self.n_heads != 0:
            raise ValueError("text_dim must be divisible by n_heads")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if not self.lora_targets:
            raise ValueError("lora_targets must not be empty")
        if self.max_len < 2:
            raise ValueError("max_len must be at least 2")

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["lora_targets"] = list(self.lora_targets)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "VLMConfig":
        data = dict(payload)
        if "lora_targets" in data:
            data["lora_targets"] = tuple(data["lora_targets"])
        return cls(**data)
