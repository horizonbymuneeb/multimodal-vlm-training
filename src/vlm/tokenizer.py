"""A word-level vocabulary with explicit special tokens.

A pretrained tokenizer would mean a download, so the project uses a word-level
vocabulary built from the generated captions.  The vocabulary is an object with
stable ids rather than a dictionary floating around, because the padding id has
to be identical at training, evaluation and inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
