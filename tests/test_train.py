"""Training-loop tests: config, determinism, one step and a short run."""

from pathlib import Path

import pytest
import torch

from src.vlm.model import build_model
from src.vlm.train import (
    TrainConfig,
    build_experiment,
    config_to_vlm,
    fit,
    load_config,
    set_seed,
    train_step,
)
from src.vlm.model import trainable_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> dict:
    return {
        "data": {"image_size": 8, "train_samples": 12, "val_samples": 4, "seed": 2},
        "model": {
            "vision_dim": 16,
            "text_dim": 16,
            "n_heads": 2,
            "n_layers": 1,
            "max_len": 14,
            "dropout": 0.0,
            "vocab_min_freq": 1,
        },
        "lora": {"rank": 2, "alpha": 4.0, "targets": ["q_proj", "v_proj"]},
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.01,
            "weight_decay": 0.0,
            "temperature": 0.07,
            "caption_weight": 0.5,
            "seed": 3,
        },
    }


def test_train_config_validation():
    TrainConfig()
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=1)
    with pytest.raises(ValueError):
        TrainConfig(lr=0.0)
    with pytest.raises(ValueError):
        TrainConfig(temperature=0.0)
    with pytest.raises(ValueError):
        TrainConfig(caption_weight=-0.1)
    with pytest.raises(ValueError):
        TrainConfig(grad_clip=0.0)


def test_set_seed_is_deterministic():
    set_seed(7)
    a = torch.randn(4)
    set_seed(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_repo_config_loads_and_converts():
    config = load_config(REPO_ROOT / "config.yaml")
    vlm_cfg, train_cfg = config_to_vlm(config)
    assert vlm_cfg.text_dim == 48
    assert train_cfg.batch_size == 8
    assert vlm_cfg.lora_targets == ("q_proj", "v_proj")
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "missing.yaml")


def test_load_config_rejects_missing_sections(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data: {}\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_train_step_reports_loss_components(model, batch):
    cfg = TrainConfig(epochs=1, batch_size=4, lr=0.01, seed=1)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr)
    report = train_step(model, batch, optimizer, cfg)
    assert set(report) == {"contrastive", "caption", "total"}
    assert report["total"] == pytest.approx(report["contrastive"] + cfg.caption_weight * report["caption"])
    assert all(value > 0 for value in report.values())
