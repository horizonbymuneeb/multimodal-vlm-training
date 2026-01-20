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
