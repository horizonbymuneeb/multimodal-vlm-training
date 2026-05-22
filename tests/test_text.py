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
