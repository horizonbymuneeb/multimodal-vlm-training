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


def test_forward_shapes(cfg, model, batch):
    out = model(batch["image"], batch["ids"], mask=batch["mask"])
    assert out["logits"].shape[:2] == batch["ids"].shape
    assert out["logits"].shape[2] == cfg.vocab_size
    assert out["image_emb"].shape == (batch["ids"].shape[0], cfg.text_dim)
    assert out["text_emb"].shape == out["image_emb"].shape
    norms = out["image_emb"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_vision_tower_is_frozen_and_decoder_is_lora_only(model):
    vision_params = list(model.vision.parameters())
    assert vision_params and all(not p.requires_grad for p in vision_params)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(("projection" in name) or ("lora_" in name) for name in trainable)
    assert any("projection" in name for name in trainable)


def test_parameter_summary_adds_up(cfg, model):
    summary = parameter_summary(model)
    assert summary["total"] == summary["trainable"] + summary["frozen"]
    assert summary["trainable"] == count_trainable(model)
    assert summary["lora_layers"] == len(list(iter_lora_modules(model.decoder)))
    assert summary["trainable"] < summary["total"]


def test_trainable_parameters_are_the_optimiser_view(model):
    params = trainable_parameters(model)
    assert params and all(p.requires_grad for p in params)
    assert sum(p.numel() for p in params) == count_trainable(model)


def test_lora_layers_are_wired_into_the_decoder(model):
    names = [name for name, _ in iter_lora_modules(model.decoder)]
    assert names
    assert all(name.endswith(("q_proj", "v_proj")) for name in names)


def test_checkpoint_roundtrip_preserves_logits(cfg, model, batch, tmp_path):
    model.eval()
    with torch.no_grad():
        before = model(batch["image"], batch["ids"], mask=batch["mask"])["logits"]
    path = save_checkpoint(model, tmp_path / "vlm.pt")
    loaded, meta = load_checkpoint(path)
    with torch.no_grad():
        after = loaded(batch["image"], batch["ids"], mask=batch["mask"])["logits"]
    assert meta["vocab_size"] == cfg.vocab_size
    assert torch.allclose(before, after, atol=1e-6)
    assert path.exists()


def test_load_checkpoint_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_config_validation_and_serialisation():
    with pytest.raises(ValueError):
        VLMConfig(vocab_size=20, text_dim=8, n_heads=3)
    with pytest.raises(ValueError):
        VLMConfig(vocab_size=20, lora_rank=0)
    with pytest.raises(ValueError):
        VLMConfig(vocab_size=20, lora_targets=())
    with pytest.raises(ValueError):
        VLMConfig(vocab_size=4)
    cfg = VLMConfig(vocab_size=20)
    restored = VLMConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_build_model_validates_vocab_size():
    cfg = VLMConfig(vocab_size=20)
    with pytest.raises(ValueError):
        build_model(21, cfg)
    with pytest.raises(ValueError):
        build_model(20, VLMConfig(vocab_size=20, lora_targets=("not_a_layer",)))


def test_text_embeddings_respect_padding_mask(model):
    ids = torch.tensor([[1, 5, 6, 2, 0, 0], [1, 7, 2, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    pooled = model.text_embeddings(ids, mask)
    manual_first = model.decoder.embed_tokens(ids[0, :4]).mean(dim=0)
    assert torch.allclose(pooled[0], manual_first, atol=1e-6)


def test_frozen_parameter_names_cover_the_vision_tower(model):
    names = list(frozen_parameter_names(model))
    assert any(name.startswith("vision.") for name in names)
    assert all("lora_" not in name for name in names)
