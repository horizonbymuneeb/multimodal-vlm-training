"""Evaluation tests: retrieval@k, MRR, token accuracy and bootstrap CI."""

import pytest
import torch

from src.vlm.evaluate import (
    bootstrap_retrieval_ci,
    chance_level_retrieval,
    embed_loader,
    evaluate_all,
    evaluate_captioning,
    evaluate_retrieval,
    mean_reciprocal_rank,
    rank_matrix,
    retrieval_at_k,
    summarise_metrics,
    token_accuracy,
)
