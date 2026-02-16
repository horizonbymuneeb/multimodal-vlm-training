"""Demo tests: the end-to-end script must run and report real numbers."""

import json
from pathlib import Path

import pytest
import yaml

from src.vlm.demo import describe_data, describe_losses, describe_model, main, run_demo

REPO_ROOT = Path(__file__).resolve().parents[1]


def demo_config() -> dict:
    return {
        "data": {"image_size": 8, "train_samples": 8, "val_samples": 4, "seed": 2},
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


def test_describe_data_reports_shapes_and_captions():
    report = describe_data(demo_config())
    assert report["samples"] == 12
    assert report["image_shape"] == [3, 8, 8]
    assert report["sample_caption"].startswith("a ")
    assert report["caption_budget"]["min"] >= 3.0


def test_describe_model_and_losses():
    from src.vlm.train import build_experiment

    bundle = build_experiment(demo_config())
    model_report = describe_model(bundle)
    assert model_report["vocab_size"] == bundle["vocab"].size
    assert model_report["parameters"]["total"] > model_report["parameters"]["trainable"] > 0
    assert 0.0 < model_report["trainable_fraction"] < 1.0
    losses = describe_losses(bundle)
    assert losses["untrained_infonce"] > 0.0
    assert 0.0 <= losses["untrained_i2t@1"] <= 1.0
