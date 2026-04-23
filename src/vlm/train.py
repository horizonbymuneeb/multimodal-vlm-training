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
