"""Loss tests: InfoNCE behaviour, padding-aware caption loss, loss mixture."""

import pytest
import torch
import torch.nn.functional as F

from src.vlm.losses import (
    batch_retrieval_accuracy,
    caption_accuracy,
    caption_cross_entropy,
    combine_losses,
    info_nce_loss,
    loss_report,
    symmetric_infonce,
    temperature_logits,
)
from src.vlm.projection import l2_normalize
