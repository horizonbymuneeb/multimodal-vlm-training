"""The projection head that maps vision features into the text space.

This is the only bridge between the frozen encoder and the decoder, so it is
kept deliberately small: two linear layers with a GELU in between and an
L2-normalised output.  Normalising here (rather than in the loss) means the
contrastive loss works on cosine similarity no matter which side produced the
vector, and retrieval can reuse the same embeddings.
"""

from __future__ import annotations

import torch
from torch import nn
