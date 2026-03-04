from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: int = 0,
    best_metric: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        ckpt["scaler"] = scaler.state_dict()
    if extra:
        ckpt["extra"] = extra

    torch.save(ckpt, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    path = Path(path)
    ckpt = torch.load(str(path), map_location=map_location)

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt