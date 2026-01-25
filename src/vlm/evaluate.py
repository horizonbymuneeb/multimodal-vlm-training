"""Evaluation: retrieval@k on the contrastive side, token accuracy on captions.

Retrieval is measured in both directions because they fail differently: a model
that ranks captions for an image well can still fail to pick the right image
for a caption when several images share a colour.
"""
