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
