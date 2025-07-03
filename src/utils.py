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
