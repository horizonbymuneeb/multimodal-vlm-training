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


def load_config(path: str | Path) -> dict:
    """Read ``config.yaml`` and check the sections the code needs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, Mapping):
        raise ValueError("config must be a mapping")
    for section in ("data", "model", "lora", "train"):
        if section not in data:
            raise ValueError(f"config is missing the {section!r} section")
    return dict(data)


def config_to_vlm(data: Mapping) -> tuple[VLMConfig, TrainConfig]:
    """Turn the YAML sections into typed configs (vocabulary size is added later)."""
    model_section = dict(data.get("model", {}))
    lora_section = dict(data.get("lora", {}))
    train_section = dict(data.get("train", {}))
    data_section = dict(data.get("data", {}))
    train_cfg = TrainConfig(
        epochs=int(train_section.get("epochs", 3)),
        batch_size=int(train_section.get("batch_size", 8)),
        lr=float(train_section.get("lr", 1e-2)),
        weight_decay=float(train_section.get("weight_decay", 0.0)),
        temperature=float(train_section.get("temperature", 0.07)),
        caption_weight=float(train_section.get("caption_weight", 0.5)),
        seed=int(train_section.get("seed", 5)),
        image_size=int(data_section.get("image_size", 16)),
        max_len=int(model_section.get("max_len", 12)),
    )
    vlm_cfg = VLMConfig(
        vocab_size=int(model_section.get("vocab_size", 0)) or 8,
        vision_dim=int(model_section.get("vision_dim", 48)),
        text_dim=int(model_section.get("text_dim", 48)),
        n_heads=int(model_section.get("n_heads", 4)),
        n_layers=int(model_section.get("n_layers", 1)),
        max_len=int(model_section.get("max_len", 12)),
        dropout=float(model_section.get("dropout", 0.0)),
        image_size=int(data_section.get("image_size", 16)),
        lora_rank=int(lora_section.get("rank", 4)),
        lora_alpha=float(lora_section.get("alpha", 8.0)),
        lora_targets=tuple(lora_section.get("targets", ["q_proj", "v_proj"])),
    )
    return vlm_cfg, train_cfg


def train_step(model, batch: Mapping[str, torch.Tensor], optimizer, cfg: TrainConfig) -> dict[str, float]:
    """One optimisation step; returns the loss components."""
    model.train()
    out = model(batch["image"], batch["ids"], mask=batch["mask"])
    contrastive = symmetric_infonce(
        out["image_emb"], out["text_emb"], temperature=cfg.temperature
    )
    caption = caption_cross_entropy(out["logits"], batch["ids"], pad_id=0)
    total = combine_losses(contrastive, caption, caption_weight=cfg.caption_weight)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), cfg.grad_clip)
    optimizer.step()
    return {
        "contrastive": float(contrastive.item()),
        "caption": float(caption.item()),
        "total": float(total.item()),
    }


def fit(model, loader, cfg: TrainConfig, *, verbose: bool = False) -> dict:
    """Train for ``cfg.epochs`` epochs and return the loss history."""
    set_seed(cfg.seed)
    params = trainable_parameters(model)
    if not params:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: dict = {"steps": [], "epochs": [], "seconds": 0.0}
    start = time.time()
    for epoch in range(cfg.epochs):
        epoch_losses: list[float] = []
        for batch in loader:
            step = train_step(model, batch, optimizer, cfg)
            history["steps"].append(step)
            epoch_losses.append(step["total"])
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["epochs"].append(mean_loss)
        if verbose:
            print(f"epoch {epoch + 1:02d} loss={mean_loss:.4f}")
    history["seconds"] = time.time() - start
    history["first_loss"] = history["steps"][0]["total"] if history["steps"] else None
    history["last_loss"] = history["steps"][-1]["total"] if history["steps"] else None
    return history


def build_experiment(config: Mapping) -> dict:
    """Build vocabulary, model, loaders and configs from a parsed config."""
    vlm_cfg, train_cfg = config_to_vlm(config)
    data_section = dict(config.get("data", {}))
    train_samples = int(data_section.get("train_samples", 64))
    val_samples = int(data_section.get("val_samples", 16))
    seed = int(data_section.get("seed", 3))
    min_freq = int(config.get("model", {}).get("vocab_min_freq", 1))
    vocab = dataset_vocabulary(
        train_samples + val_samples, image_size=train_cfg.image_size, seed=seed, min_freq=min_freq
    )
    vlm_cfg = VLMConfig(**{**vlm_cfg.to_dict(), "vocab_size": vocab.size})
    model = build_model(vocab.size, vlm_cfg, seed=train_cfg.seed)
    train_loader = make_dataloader(
        train_samples,
        vocab,
        batch_size=train_cfg.batch_size,
        image_size=train_cfg.image_size,
        max_len=train_cfg.max_len,
        seed=seed,
    )
    val_loader = make_dataloader(
        val_samples,
        vocab,
        batch_size=train_cfg.batch_size,
        image_size=train_cfg.image_size,
        max_len=train_cfg.max_len,
        seed=seed + 1,
        shuffle=False,
    )
    return {
        "vocab": vocab,
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_cfg": train_cfg,
        "vlm_cfg": vlm_cfg,
    }


def main(argv: Iterable[str] | None = None) -> int:
    """CLI: ``python -m src.vlm.train --config config.yaml``."""
    parser = argparse.ArgumentParser(description="Train the toy vision-language model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="artifacts/run.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    bundle = build_experiment(config)
    model = bundle["model"]
    summary = parameter_summary(model)
    history = fit(model, bundle["train_loader"], bundle["train_cfg"], verbose=True)
    metrics = evaluate_all(model, bundle["val_loader"], k_values=(1, 3), pad_id=0)
    report = {
        "vocab_size": bundle["vocab"].size,
        "parameters": summary,
        "train": {
            "epochs": history["epochs"],
            "first_loss": history["first_loss"],
            "last_loss": history["last_loss"],
            "seconds": round(float(history["seconds"]), 2),
        },
        "val": metrics,
        "val_summary": summarise_metrics(metrics),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
