"""LoRA-style low-rank adapters for the decoder projections.

The decoder is frozen; instead of training ``q_proj``/``v_proj`` directly, each
one gets a frozen base weight plus a trainable low-rank update ``B @ A``.  ``B``
is initialised to zero, so at step 0 the adapted model is *exactly* the frozen
model; the tests assert that, because a non-zero init silently changes the
baseline every experiment is compared against.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if base.out_features < rank or base.in_features < rank:
            raise ValueError("rank must not exceed the layer dimensions")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5, generator=generator)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        update = (x @ self.lora_a.transpose(0, 1)) @ self.lora_b.transpose(0, 1)
        return self.base(x) + update * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The effective weight update, ``scaling * B @ A``."""
        return self.scaling * (self.lora_b @ self.lora_a)

    def merge(self) -> None:
        """Fold the update into the base weight (inference-time optimisation)."""
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight += self.delta_weight()
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight -= self.delta_weight()
        self.merged = False

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}"


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def apply_lora(
    model: nn.Module,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    *,
    rank: int = 4,
    alpha: float = 8.0,
    seed: int | None = None,
) -> int:
    """Replace matching ``nn.Linear`` submodules with :class:`LoRALinear`.

    A target matches when the last dotted component of the module name equals
    one of ``targets``.  Returns the number of replaced layers.
    """
    if not targets:
        raise ValueError("targets must not be empty")
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    offset = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                layer_seed = None if seed is None else seed + offset
                offset += 1
                replacements.append(
                    (parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, seed=layer_seed))
                )
    for parent, child_name, layer in replacements:
        setattr(parent, child_name, layer)
    return len(replacements)


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze everything, then unfreeze only the adapter matrices."""
    for param in model.parameters():
        param.requires_grad = False
    count = 0
    for _, module in iter_lora_modules(model):
        module.lora_a.requires_grad = True
        module.lora_b.requires_grad = True
        count += 2
    return count


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in iter_lora_modules(model):
        params.extend([module.lora_a, module.lora_b])
    return params


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model: nn.Module) -> int:
    """Merge every adapter into its base layer; returns the number merged."""
    merged = 0
    for _, module in iter_lora_modules(model):
        module.merge()
        merged += 1
    return merged


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors, keyed by module name."""
    payload: dict[str, torch.Tensor] = {}
    for name, module in iter_lora_modules(model):
        payload[f"{name}.lora_a"] = module.lora_a.detach().clone()
        payload[f"{name}.lora_b"] = module.lora_b.detach().clone()
    return payload


def lora_rank_summary(model: nn.Module) -> dict[str, int]:
    """Rank and parameter count per adapted layer."""
    return {
        name: module.lora_a.numel() + module.lora_b.numel()
        for name, module in iter_lora_modules(model)
    }
