from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import math
import torch


def build_optimizer(cfg: Dict[str, Any], params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    ocfg = cfg["optimizer"]
    name = ocfg.get("name", "AdamW")
    lr = float(ocfg.get("lr", 3e-4))
    weight_decay = float(ocfg.get("weight_decay", 0.0))
    betas = tuple(ocfg.get("betas", [0.9, 0.999]))

    if name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name.lower() == "sgd":
        momentum = float(ocfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)

    raise ValueError(f"Unknown optimizer '{name}'")


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                t = (step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
                t = min(max(t, 0.0), 1.0)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> Any:
    scfg = cfg.get("scheduler", {"name": "none"})
    name = scfg.get("name", "none").lower()

    if name == "none":
        return None

    if name == "step":
        step_size = int(scfg.get("step_size", 1))
        gamma = float(scfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        epochs = int(cfg["train"]["epochs"])
        total_steps = epochs * steps_per_epoch
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        warmup_steps = warmup_epochs * steps_per_epoch
        min_lr = float(scfg.get("min_lr", 1e-6))
        return WarmupCosineLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)

    raise ValueError(f"Unknown scheduler '{name}'")