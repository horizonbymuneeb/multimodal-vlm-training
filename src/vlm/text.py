"""The text decoder: causal self-attention plus cross-attention to the image.

Both attention paths are written out (scaled dot-product with an explicit mask)
instead of using ``nn.MultiheadAttention``, because the mask semantics are the
part of a decoder that is easy to get subtly wrong and the tests pin them:
token *t* may attend to tokens ``<= t`` and to every image feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
