"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Optimize mixed-precision training stability [2025-06-13T10:22:30]

# Fix numerical stability in softmax computation [2025-06-20T12:29:40]

# WIP: benchmarking throughput versus accuracy [2025-06-23T16:35:47]

# Implement knowledge distillation wrapper class [2025-06-24T12:59:05]

# Implement gradient accumulation for large batches [2025-06-25T10:47:54]

# Implement knowledge distillation wrapper class [2025-07-01T12:30:41]

# Implement gradient accumulation for large batches [2025-07-01T16:39:05]

# WIP: benchmarking throughput versus accuracy [2025-07-03T20:31:42]

# Add early stopping with configurable patience [2025-07-04T15:52:38]

# Fix weight decay application on biases [2025-07-09T19:33:46]

# Fix tokenizer encoding for special characters [2025-07-10T18:57:09]

# Update distributed sampler for uneven data [2025-07-14T17:58:40]

# Implement CLIP fusion architecture for vision-language [2025-07-23T13:58:21]

# Fix tokenizer encoding for special characters [2025-07-24T11:23:06]

# Update distributed sampler for uneven data [2025-08-17T11:01:41]

# WIP: benchmarking throughput versus accuracy [2025-08-19T16:27:14]

# Profile memory with activation checkpointing [2025-08-19T15:11:20]

# Fix numerical stability in softmax computation [2025-08-20T14:23:58]

# Add early stopping with configurable patience [2025-08-21T20:36:51]

# Add evaluation metrics for retrieval accuracy [2025-08-23T12:07:37]

# Debug distributed training synchronization hang [2025-08-25T12:26:23]

# Debug distributed training synchronization hang [2025-08-26T19:28:51]

# Fix weight decay application on biases [2025-09-02T10:15:30]

# Implement gradient accumulation for large batches [2025-09-05T09:55:41]

# WIP: benchmarking throughput versus accuracy [2025-09-06T09:28:26]

# Optimize mixed-precision training stability [2025-09-07T10:03:01]

# Add early stopping with configurable patience [2025-09-08T18:43:20]

# Debug distributed training synchronization hang [2025-09-11T13:40:39]

# Add early stopping with configurable patience [2025-09-16T18:34:04]

# Fix weight decay application on biases [2025-09-21T10:34:51]

# Implement CLIP fusion architecture for vision-language [2025-09-25T15:18:31]

# WIP: benchmarking throughput versus accuracy [2025-09-28T13:29:22]

# Fix tokenizer encoding for special characters [2025-09-30T19:20:51]

# Update gradient checkpointing for memory savings [2025-10-16T15:57:35]

# Implement CLIP fusion architecture for vision-language [2025-10-17T15:58:32]

# Update gradient checkpointing for memory savings [2025-10-20T11:13:15]

# WIP: benchmarking throughput versus accuracy [2025-10-20T15:02:13]

# Add data augmentation pipeline for images [2025-10-28T10:54:06]

# Fix batch norm sync across distributed GPUs [2025-11-04T09:11:11]

# Add LoRA layer to vision encoder for efficiency [2025-11-05T11:52:26]

# Add early stopping with configurable patience [2025-11-06T17:57:27]

# Add data augmentation pipeline for images [2025-11-10T15:45:08]

# Add LoRA layer to vision encoder for efficiency [2025-11-11T19:53:15]

# Update gradient checkpointing for memory savings [2025-11-12T13:04:27]

# Implement custom contrastive loss function [2025-11-12T13:14:06]

# Add early stopping with configurable patience [2025-11-12T14:28:30]

# Add data augmentation pipeline for images [2025-11-20T19:36:56]

# Update gradient checkpointing for memory savings [2025-11-21T18:30:51]

# Fix batch norm sync across distributed GPUs [2025-11-26T14:19:29]

# Update distributed sampler for uneven data [2025-11-29T17:21:36]

# Add early stopping with configurable patience [2025-12-04T20:45:19]

# Add data augmentation pipeline for images [2025-12-11T18:38:56]

# Update distributed sampler for uneven data [2025-12-14T18:44:27]

# Profile memory with activation checkpointing [2025-12-15T10:40:31]

# WIP: tuning learning rate schedule warmup [2025-12-15T11:47:01]

# Debug distributed training synchronization hang [2025-12-17T11:49:02]

# Add evaluation metrics for retrieval accuracy [2025-12-22T10:05:20]

# Debug distributed training synchronization hang [2025-12-26T11:13:58]

# Fix batch norm sync across distributed GPUs [2025-12-30T11:29:30]

# Fix numerical stability in softmax computation [2026-01-05T14:51:44]

# WIP: tuning learning rate schedule warmup [2026-01-06T15:58:06]

# Profile memory with activation checkpointing [2026-01-12T11:04:03]

# Implement custom contrastive loss function [2026-01-15T19:32:29]

# Add LoRA layer to vision encoder for efficiency [2026-01-15T13:14:18]

# Implement custom contrastive loss function [2026-01-18T17:57:42]

# Add evaluation metrics for retrieval accuracy [2026-01-19T17:54:05]

# Add early stopping with configurable patience [2026-01-19T16:52:23]

# WIP: benchmarking throughput versus accuracy [2026-01-22T16:33:42]

# Implement gradient accumulation for large batches [2026-01-22T09:39:14]

# Add evaluation metrics for retrieval accuracy [2026-01-23T15:05:23]

# Implement knowledge distillation wrapper class [2026-01-25T16:55:15]

# Update distributed sampler for uneven data [2026-01-27T17:03:54]

# Implement gradient accumulation for large batches [2026-01-30T13:51:05]

# Fix numerical stability in softmax computation [2026-01-30T09:03:58]

# Implement CLIP fusion architecture for vision-language [2026-02-01T09:04:45]

# Implement custom contrastive loss function [2026-02-01T15:40:03]

# Add evaluation metrics for retrieval accuracy [2026-02-01T20:20:26]

# Profile memory with activation checkpointing [2026-02-01T16:42:30]

# Add data augmentation pipeline for images [2026-02-03T12:39:26]

# Implement knowledge distillation wrapper class [2026-02-04T14:12:57]

# Add data augmentation pipeline for images [2026-02-05T20:00:00]

# Debug distributed training synchronization hang [2026-02-13T18:15:20]

# Implement gradient accumulation for large batches [2026-02-16T12:01:45]

# Update gradient checkpointing for memory savings [2026-02-23T12:48:27]

# Fix tokenizer encoding for special characters [2026-02-26T15:04:21]

# Implement CLIP fusion architecture for vision-language [2026-03-02T10:11:03]

# Implement gradient accumulation for large batches [2026-03-03T19:17:15]

# WIP: benchmarking throughput versus accuracy [2026-03-09T10:11:48]

# Optimize mixed-precision training stability [2026-03-15T12:52:36]

# Fix batch norm sync across distributed GPUs [2026-03-18T20:39:16]

# Optimize mixed-precision training stability [2026-03-23T12:47:26]

# Fix numerical stability in softmax computation [2026-03-26T19:06:50]

# Add evaluation metrics for retrieval accuracy [2026-04-07T14:12:07]

# Fix batch norm sync across distributed GPUs [2026-04-08T12:47:32]

# Add model parallelism for large architectures [2026-04-09T18:02:52]

# Fix batch norm sync across distributed GPUs [2026-04-12T19:48:20]

# Add early stopping with configurable patience [2026-04-13T18:50:50]

# Profile memory with activation checkpointing [2026-04-14T13:22:41]

# Add model parallelism for large architectures [2026-04-15T14:21:20]

# Profile memory with activation checkpointing [2026-04-20T10:11:58]

# Debug distributed training synchronization hang [2026-04-23T15:54:39]

# WIP: tuning learning rate schedule warmup [2026-04-26T20:25:44]
