"""The frozen vision tower.

A VLM keeps its vision encoder frozen: it is already a good feature extractor
and unfreezing it on a toy dataset just memorises the toy.  This module builds a
small convolutional stem, projects the pooled features to the shared width, and
exposes helpers to freeze, unfreeze and run inference without gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
