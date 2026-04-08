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


def test_lora_update_matches_the_hand_formula():
    base = base_layer()
    layer = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.ones_like(layer.lora_a))
        layer.lora_b.copy_(torch.ones_like(layer.lora_b))
    x = torch.randn(2, 6)
    expected = base(x) + 2.0 * (x @ torch.ones(6, 2)) @ torch.ones(2, 5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_merge_and_unmerge_preserve_outputs():
    base = base_layer()
    layer = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(3, 6)
    before = layer(x).clone()
    layer.merge()
    assert layer.merged
    assert torch.allclose(layer(x), before, atol=1e-5)
    weight_after_merge = layer.base.weight.detach().clone()
    layer.merge()  # idempotent
    assert torch.allclose(layer.base.weight, weight_after_merge)
    layer.unmerge()
    assert not layer.merged
    assert torch.allclose(layer(x), before, atol=1e-5)


def test_lora_validation():
    with pytest.raises(ValueError):
        LoRALinear(base_layer(), rank=0)
    with pytest.raises(ValueError):
        LoRALinear(base_layer(), rank=2, alpha=0.0)
    with pytest.raises(ValueError):
        LoRALinear(base_layer(2, 2), rank=4)
