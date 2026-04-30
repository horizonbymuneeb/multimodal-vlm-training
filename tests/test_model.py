"""Assembled-model tests: wiring, freezing, parameter split and checkpoints."""

import pytest
import torch

from src.vlm.lora import count_trainable, iter_lora_modules
from src.vlm.model import (
    SmallVLM,
    VLMConfig,
    build_model,
    frozen_parameter_names,
    load_checkpoint,
    parameter_summary,
    save_checkpoint,
    trainable_parameters,
)


def test_forward_shapes(cfg, model, batch):
    out = model(batch["image"], batch["ids"], mask=batch["mask"])
    assert out["logits"].shape[:2] == batch["ids"].shape
    assert out["logits"].shape[2] == cfg.vocab_size
    assert out["image_emb"].shape == (batch["ids"].shape[0], cfg.text_dim)
    assert out["text_emb"].shape == out["image_emb"].shape
    norms = out["image_emb"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_vision_tower_is_frozen_and_decoder_is_lora_only(model):
    vision_params = list(model.vision.parameters())
    assert vision_params and all(not p.requires_grad for p in vision_params)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(("projection" in name) or ("lora_" in name) for name in trainable)
    assert any("projection" in name for name in trainable)


def test_parameter_summary_adds_up(cfg, model):
    summary = parameter_summary(model)
    assert summary["total"] == summary["trainable"] + summary["frozen"]
    assert summary["trainable"] == count_trainable(model)
    assert summary["lora_layers"] == len(list(iter_lora_modules(model.decoder)))
    assert summary["trainable"] < summary["total"]
