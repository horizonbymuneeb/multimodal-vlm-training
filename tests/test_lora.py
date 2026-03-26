"""LoRA adapter tests: zero-init, freezing, merge/unmerge and accounting."""

import pytest
import torch
from torch import nn

from src.vlm.lora import (
    LoRALinear,
    apply_lora,
    count_trainable,
    iter_lora_modules,
    lora_parameters,
    lora_rank_summary,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora,
)
from src.vlm.text import TextConfig, TextDecoder
