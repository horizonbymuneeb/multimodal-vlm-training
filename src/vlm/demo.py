"""End-to-end demo: dataset, model wiring, a short training run, evaluation.

Run with ``python -m src.vlm.demo --config config.yaml``.  It is the fastest way
to see the shape of the project without reading every module: vocabulary size,
parameter split between frozen and trainable, a few training steps and the
retrieval/captioning numbers that come out the other end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from src.vlm.dataset import caption_for, caption_token_budget, generate_sample, random_spec
from src.vlm.evaluate import evaluate_all, summarise_metrics
from src.vlm.losses import batch_retrieval_accuracy, info_nce_loss
from src.vlm.model import parameter_summary
from src.vlm.train import build_experiment, fit, load_config


def describe_data(config: dict) -> dict:
    """Vocabulary size, caption length budget and one rendered sample."""
    data = dict(config.get("data", {}))
    n = int(data.get("train_samples", 64)) + int(data.get("val_samples", 16))
    seed = int(data.get("seed", 3))
    image_size = int(data.get("image_size", 16))
    captions = [caption_for(random_spec(seed * 100_000 + i)) for i in range(n)]
    image, caption, spec = generate_sample(seed, image_size)
    return {
        "samples": n,
        "image_shape": list(image.shape),
        "caption_budget": caption_token_budget(captions),
        "sample_caption": caption,
        "sample_spec": {
            "shape": spec.shape,
            "color": spec.color,
            "position": spec.position,
            "second_shape": spec.second_shape,
        },
    }


def describe_model(bundle: dict) -> dict:
    """Parameter split for the assembled model."""
    summary = parameter_summary(bundle["model"])
    return {
        "vocab_size": bundle["vocab"].size,
        "parameters": summary,
        "trainable_fraction": round(summary["trainable"] / max(1, summary["total"]), 4),
    }


def describe_losses(bundle: dict) -> dict:
    """Contrastive loss on an untrained batch, for reference."""
    model = bundle["model"]
    batch = next(iter(bundle["train_loader"]))
    with torch.no_grad():
        out = model(batch["image"], batch["ids"], mask=batch["mask"])
        loss = info_nce_loss(out["image_emb"], out["text_emb"], temperature=0.07)
        accuracy = batch_retrieval_accuracy(out["image_emb"], out["text_emb"])
    return {"untrained_infonce": float(loss.item()), "untrained_i2t@1": accuracy}


def run_demo(config: dict, *, verbose: bool = False) -> dict:
    """Build, train briefly and evaluate; returns a JSON-friendly report."""
    bundle = build_experiment(config)
    before = describe_losses(bundle)
    history = fit(bundle["model"], bundle["train_loader"], bundle["train_cfg"], verbose=verbose)
    metrics = evaluate_all(bundle["model"], bundle["val_loader"], k_values=(1, 3), pad_id=0)
    return {
        "data": describe_data(config),
        "model": describe_model(bundle),
        "losses": before,
        "train": {
            "epochs": history["epochs"],
            "first_loss": history["first_loss"],
            "last_loss": history["last_loss"],
            "seconds": round(float(history["seconds"]), 3),
        },
        "val": metrics,
        "val_summary": summarise_metrics(metrics),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the toy VLM demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_demo(load_config(args.config))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0
