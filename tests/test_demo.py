"""Demo tests: the end-to-end script must run and report real numbers."""

import json
from pathlib import Path

import pytest
import yaml

from src.vlm.demo import describe_data, describe_losses, describe_model, main, run_demo
