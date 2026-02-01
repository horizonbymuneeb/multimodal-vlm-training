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
