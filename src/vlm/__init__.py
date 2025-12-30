"""A small vision-language model, trainable end to end on a laptop CPU.

The interesting part of a VLM is not the size of the decoder but the seam
between the two modalities: a frozen vision encoder, a projection that maps
image features into the text embedding space, and low-rank adapters that make
the decoder trainable without touching its frozen weights.  The package keeps
each of those pieces in its own module so they can be tested separately.
"""

from src.vlm.losses import caption_cross_entropy, info_nce_loss, symmetric_infonce
from src.vlm.lora import LoRALinear, apply_lora, count_trainable, merge_lora
from src.vlm.model import SmallVLM, VLMConfig, build_model
from src.vlm.projection import ProjectionHead, l2_normalize
from src.vlm.tokenizer import Vocab, build_vocab
from src.vlm.vision import VisionEncoder, build_vision_encoder

__all__ = [
    "LoRALinear",
    "ProjectionHead",
    "SmallVLM",
    "VLMConfig",
    "VisionEncoder",
    "Vocab",
    "apply_lora",
    "build_model",
    "build_vocab",
    "build_vision_encoder",
    "caption_cross_entropy",
    "count_trainable",
    "info_nce_loss",
    "l2_normalize",
    "merge_lora",
    "symmetric_infonce",
]
