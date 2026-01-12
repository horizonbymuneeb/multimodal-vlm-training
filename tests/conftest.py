"""Shared fixtures: a tiny vocabulary, model and loader so tests stay fast.

The model used in tests is deliberately smaller than the config.yaml one
(16-wide, one layer, rank-2 adapters, 8x8 images) so the whole suite runs in a
few seconds on CPU.
"""
