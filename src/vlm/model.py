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


class SmallVLM(nn.Module):
    """Frozen vision encoder + projection head + LoRA-adapted text decoder."""

    def __init__(self, cfg: VLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision: VisionEncoder = build_vision_encoder(
            cfg.vision_dim,
            channels=cfg.vision_channels,
            image_size=cfg.image_size,
            freeze=True,
        )
        self.projection: ProjectionHead = build_projection(
            cfg.vision_dim, cfg.text_dim, dropout=cfg.dropout
        )
        self.decoder = TextDecoder(
            TextConfig(
                vocab_size=cfg.vocab_size,
                dim=cfg.text_dim,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                max_len=cfg.max_len,
                dropout=cfg.dropout,
            )
        )
        adapted = apply_lora(
            self.decoder,
            targets=cfg.lora_targets,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            seed=0,
        )
        if adapted == 0:
            raise ValueError("no decoder projections matched the LoRA targets")
        mark_only_lora_trainable(self.decoder)

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Projected, L2-normalised image embeddings in the text space."""
        with torch.no_grad():
            features = self.vision(images)
        return self.projection(features)

    def text_embeddings(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pooled decoder token embeddings, used as the text side of InfoNCE."""
        embedded = self.decoder.embed_tokens(ids)
        if mask is None:
            pooled = embedded.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).to(embedded.dtype)
            pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(
        self,
        images: torch.Tensor,
        ids: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Logits plus both embedding spaces for the contrastive loss."""
        image_emb = self.encode_images(images)
        memory = image_emb.unsqueeze(1)
        out = self.decoder(ids, memory=memory)
        text_emb = self.text_embeddings(ids, mask)
        return {"logits": out["logits"], "image_emb": image_emb, "text_emb": text_emb}

    def trainable_parameter_count(self) -> int:
        return count_trainable(self)

    def lora_layers(self) -> int:
        return sum(1 for _ in iter_lora_modules(self.decoder))


def build_model(
    vocab_size: int,
    cfg: VLMConfig | None = None,
    *,
    seed: int | None = None,
) -> SmallVLM:
    """Build the model with deterministic initialisation when ``seed`` is set."""
    if cfg is None:
        cfg = VLMConfig(vocab_size=vocab_size)
    elif cfg.vocab_size != vocab_size:
        raise ValueError("cfg.vocab_size and vocab_size disagree")
    if seed is not None:
        torch.manual_seed(seed)
    return SmallVLM(cfg)
