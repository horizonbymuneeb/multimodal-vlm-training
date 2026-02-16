"""Losses: hand-written InfoNCE, captioning cross-entropy, and their mixture.

InfoNCE is implemented from the definition (cosine logits, softmax over the
batch, cross-entropy against the identity) rather than calling a library helper,
because the temperature scaling and the in-batch negatives are the two things
that make contrastive training work and both are worth being explicit about.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.vlm.projection import l2_normalize
