from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .registry import LOSSES, register


@register(LOSSES, "cross_entropy")
def build_cross_entropy(cfg: Dict[str, Any]) -> nn.Module:
    lcfg = cfg.get("loss", {})
    label_smoothing = float(lcfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


@register(LOSSES, "cosine_embedding")
def build_cosine_embedding(cfg: Dict[str, Any]) -> nn.Module:
    # Example embedding loss. Needs target pairs; for your tasks you can swap to SupCon/ArcFace/etc.
    return nn.CosineEmbeddingLoss(margin=float(cfg.get("loss", {}).get("margin", 0.0)))


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg.get("loss", {}).get("name", "cross_entropy")
    builder = LOSSES.get(name)
    if builder is None:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES.keys())}")
    return builder(cfg)