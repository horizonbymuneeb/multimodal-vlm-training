"""Decoder tests: causal masking, cross-attention and logits shape."""

import pytest
import torch

from src.vlm.text import (
    DecoderBlock,
    MultiHeadAttention,
    TextConfig,
    TextDecoder,
    causal_mask,
    next_token_logits,
    positional_encoding,
    scaled_dot_product,
)


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    assert mask[0].tolist() == [True, False, False, False]
    assert mask[3].tolist() == [True, True, True, True]
    with pytest.raises(ValueError):
        causal_mask(0)


def test_positional_encoding_shape_and_first_row():
    encoding = positional_encoding(6, 8)
    assert encoding.shape == (6, 8)
    assert encoding[0, 0] == pytest.approx(0.0)
    assert encoding[0, 1] == pytest.approx(1.0)
    assert encoding[1, 0] == pytest.approx(torch.sin(torch.tensor(1.0)).item(), rel=1e-5)
    with pytest.raises(ValueError):
        positional_encoding(0, 8)


def test_scaled_dot_product_masks_future_positions():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    out, weights = scaled_dot_product(q, k, v, causal_mask(3))
    assert out.shape == (1, 2, 3, 4)
    assert weights.shape == (1, 2, 3, 3)
    upper = weights.triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 3), atol=1e-6)


def test_scaled_dot_product_validation():
    with pytest.raises(ValueError):
        scaled_dot_product(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3))
    q = torch.zeros(1, 1, 3, 4)
    with pytest.raises(ValueError):
        scaled_dot_product(q, torch.zeros(1, 1, 3, 5), torch.zeros(1, 1, 3, 4))
    with pytest.raises(ValueError):
        scaled_dot_product(q, q, q, torch.ones(2, 2, dtype=torch.bool))


def test_multihead_attention_shapes_and_head_validation():
    attn = MultiHeadAttention(8, 2)
    x = torch.randn(2, 5, 8)
    out, weights = attn(x, mask=causal_mask(5))
    assert out.shape == (2, 5, 8)
    assert weights.shape == (2, 2, 5, 5)
    with pytest.raises(ValueError):
        MultiHeadAttention(8, 3)


def test_decoder_block_cross_attention_uses_memory():
    block = DecoderBlock(8, 2)
    x = torch.randn(2, 4, 8)
    memory = torch.randn(2, 1, 8)
    out, weights = block(x, mask=causal_mask(4), memory=memory)
    assert out.shape == (2, 4, 8)
    assert "self" in weights and "cross" in weights
    assert weights["cross"].shape == (2, 2, 4, 1)


def test_text_decoder_logits_shape_and_attention_masks():
    cfg = TextConfig(vocab_size=20, dim=8, n_heads=2, n_layers=1, max_len=6)
    decoder = TextDecoder(cfg)
    ids = torch.randint(0, 20, (2, 5))
    out = decoder(ids)
    assert out["logits"].shape == (2, 5, 20)
    assert out["hidden"].shape == (2, 5, 8)
    attention = out["attentions"]["0.self"]
    assert attention.shape == (2, 2, 5, 5)
    upper = attention.triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)


def test_text_decoder_conditions_on_memory():
    torch.manual_seed(0)
    cfg = TextConfig(vocab_size=20, dim=8, n_heads=2, n_layers=1, max_len=6)
    decoder = TextDecoder(cfg)
    ids = torch.randint(0, 20, (2, 4))
    plain = decoder(ids)["logits"]
    conditioned = decoder(ids, memory=torch.randn(2, 1, 8))["logits"]
    assert plain.shape == conditioned.shape
    assert not torch.allclose(plain, conditioned)


def test_text_decoder_rejects_long_sequences_and_bad_configs():
    cfg = TextConfig(vocab_size=20, dim=8, n_heads=2, n_layers=1, max_len=4)
    decoder = TextDecoder(cfg)
    with pytest.raises(ValueError):
        decoder(torch.zeros(1, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        decoder(torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        TextConfig(vocab_size=20, dim=8, n_heads=3)
    with pytest.raises(ValueError):
        TextConfig(vocab_size=4)
