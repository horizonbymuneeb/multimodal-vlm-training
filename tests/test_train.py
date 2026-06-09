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


def test_train_step_updates_only_trainable_weights(model, batch):
    cfg = TrainConfig(epochs=1, batch_size=4, lr=0.05, seed=1)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr)
    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    trainable_before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    train_step(model, batch, optimizer, cfg)
    for name, param in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(param, frozen_before[name]), f"{name} changed while frozen"
    changed = [
        name
        for name, param in model.named_parameters()
        if name in trainable_before and not torch.equal(param, trainable_before[name])
    ]
    assert changed, "no trainable parameter changed"


def test_fit_records_history_and_reduces_loss():
    config = tiny_config()
    bundle = build_experiment(config)
    cfg = TrainConfig(epochs=2, batch_size=4, lr=0.02, seed=1)
    history = fit(bundle["model"], bundle["train_loader"], cfg)
    assert len(history["epochs"]) == 2
    assert len(history["steps"]) == 2 * len(bundle["train_loader"])
    assert all(value == value for value in history["epochs"])  # not NaN
    assert history["last_loss"] is not None
    assert history["last_loss"] < history["first_loss"]
    assert history["seconds"] >= 0.0


def test_build_experiment_wires_vocab_and_loaders():
    config = tiny_config()
    bundle = build_experiment(config)
    vocab = bundle["vocab"]
    assert bundle["model"].vocab_size == vocab.size
    assert len(bundle["train_loader"].dataset) == 12
    assert len(bundle["val_loader"].dataset) == 4
    batch = next(iter(bundle["train_loader"]))
    assert batch["ids"].max() < vocab.size
    assert batch["image"].shape == (4, 3, 8, 8)


def test_build_experiment_is_reproducible():
    config = tiny_config()
    first = build_experiment(config)
    second = build_experiment(config)
    a = next(iter(first["train_loader"]))
    b = next(iter(second["train_loader"]))
    assert torch.equal(a["ids"], b["ids"])
    assert torch.allclose(a["image"], b["image"])


def test_fit_requires_trainable_parameters():
    config = tiny_config()
    bundle = build_experiment(config)
    model = bundle["model"]
    for param in model.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError):
        fit(model, bundle["train_loader"], TrainConfig(epochs=1, batch_size=4))
