from __future__ import annotations

from typing import Dict
import torch

from .registry import METRICS, register


@register(METRICS, "accuracy_top1")
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()


@register(METRICS, "none")
def none_metric(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=logits.device)


def build_metric(cfg: Dict) -> str:
    return cfg.get("eval", {}).get("metric", "accuracy_top1")