"""The frozen vision tower.

A VLM keeps its vision encoder frozen: it is already a good feature extractor
and unfreezing it on a toy dataset just memorises the toy.  This module builds a
small convolutional stem, projects the pooled features to the shared width, and
exposes helpers to freeze, unfreeze and run inference without gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VisionConfig:
    """Shape of the convolutional stem and the output width."""

    out_dim: int = 48
    channels: int = 16
    in_channels: int = 3
    image_size: int = 16

    def __post_init__(self) -> None:
        if self.out_dim <= 0 or self.channels <= 0:
            raise ValueError("out_dim and channels must be positive")
        if self.in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8")


class ConvStem(nn.Module):
    """Three stride-2 conv blocks, so a 16x16 image becomes 2x2 features."""

    def __init__(self, channels: int = 16, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"expected (N, C, H, W) images, got {tuple(images.shape)}")
        return self.net(images)


class VisionEncoder(nn.Module):
    """Conv stem + pooled projection to ``out_dim``.

    The projection is a real layer rather than a reshape so the encoder output
    width is decoupled from the image size and the channel count.
    """

    def __init__(self, cfg: VisionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VisionConfig()
        self.stem = ConvStem(self.cfg.channels, self.cfg.in_channels)
        pooled = self.cfg.channels * 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(pooled, self.cfg.out_dim)
        self.norm = nn.LayerNorm(self.cfg.out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        pooled = self.pool(features).flatten(1)
        return self.norm(self.proj(pooled))

    def feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Spatial features before pooling, useful for debugging."""
        return self.stem(images)


def set_trainable(module: nn.Module, flag: bool) -> int:
    """Set ``requires_grad`` on every parameter; return how many were touched."""
    count = 0
    for param in module.parameters():
        param.requires_grad = flag
        count += 1
    return count


def freeze_parameters(module: nn.Module) -> int:
    return set_trainable(module, False)


def unfreeze_parameters(module: nn.Module) -> int:
    return set_trainable(module, True)
