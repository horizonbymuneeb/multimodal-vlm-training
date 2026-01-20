"""Shared fixtures: a tiny vocabulary, model and loader so tests stay fast.

The model used in tests is deliberately smaller than the config.yaml one
(16-wide, one layer, rank-2 adapters, 8x8 images) so the whole suite runs in a
few seconds on CPU.
"""

import pytest
import torch

from src.vlm.dataset import dataset_vocabulary, make_dataloader
from src.vlm.model import VLMConfig, build_model


@pytest.fixture(scope="session")
def vocab():
    return dataset_vocabulary(24, image_size=8, seed=1)


@pytest.fixture(scope="session")
def cfg(vocab):
    return VLMConfig(
        vocab_size=vocab.size,
        vision_dim=16,
        text_dim=16,
        n_heads=2,
        n_layers=1,
        max_len=14,
        image_size=8,
        vision_channels=4,
        lora_rank=2,
        lora_alpha=4.0,
    )


@pytest.fixture
def model(cfg):
    return build_model(cfg.vocab_size, cfg, seed=0)


@pytest.fixture(scope="session")
def loader(vocab):
    return make_dataloader(
        16, vocab, batch_size=4, image_size=8, max_len=14, seed=2, shuffle=False
    )


@pytest.fixture
def batch(loader):
    return next(iter(loader))


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)
