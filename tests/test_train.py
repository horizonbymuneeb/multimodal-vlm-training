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
