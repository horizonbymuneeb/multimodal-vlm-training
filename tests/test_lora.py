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


def base_layer(in_features: int = 6, out_features: int = 5) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(in_features, out_features)


def test_lora_starts_as_an_exact_copy_of_the_base_layer():
    base = base_layer()
    layer = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 6)
    assert torch.allclose(layer(x), base(x))
    assert torch.allclose(layer.delta_weight(), torch.zeros_like(base.weight))
    assert layer.scaling == pytest.approx(2.0)


def test_lora_only_trains_the_adapter():
    base = base_layer()
    layer = LoRALinear(base, rank=2)
    assert all(not p.requires_grad for p in layer.base.parameters())
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad
    assert count_trainable(layer) == layer.lora_a.numel() + layer.lora_b.numel()
    assert count_trainable(layer) == 2 * 6 + 5 * 2
