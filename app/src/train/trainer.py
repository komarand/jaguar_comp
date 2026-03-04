from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.losses import build_loss
from ..models.metrics import METRICS, build_metric
from ..train.optim import build_optimizer, build_scheduler
from ..utils.checkpointing import save_checkpoint
from ..utils.logging import JsonlLogger


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metric_name: str,
    device: torch.device,
    amp: bool,
) -> Tuple[float, float]:
    model.eval()
    metric_fn = METRICS.get(metric_name, None)
    if metric_fn is None:
        raise ValueError(f"Unknown metric '{metric_name}'. Available: {list(METRICS.keys())}")

    total_loss = 0.0
    total_metric = 0.0
    n = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        x = batch["image"]
        y = batch.get("target", None)

        with autocast(enabled=amp):
            out = model(x)
            if y is None:
                # no labels; skip
                continue
            loss = loss_fn(out, y)

        metric = metric_fn(out, y)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_metric += float(metric.item()) * bs
        n += bs

        pbar.set_postfix(loss=total_loss / max(1, n), metric=total_metric / max(1, n))

    return total_loss / max(1, n), total_metric / max(1, n)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp: bool,
    scaler: GradScaler,
    scheduler: Any,
    grad_clip_norm: float,
    accumulation_steps: int,
    log_every: int,
    epoch: int,
    logger: JsonlLogger | None = None,
) -> float:
    model.train()

    total_loss = 0.0
    n = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"train e{epoch}", leave=False)

    for step, batch in pbar:
        batch = _to_device(batch, device)
        x = batch["image"]
        y = batch.get("target", None)
        if y is None:
            continue

        with autocast(enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                # If cosine warmup scheduler is step-based, call each optimizer step
                if scheduler.__class__.__name__ == "WarmupCosineLR":
                    scheduler.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs * accumulation_steps
        n += bs

        if (step + 1) % max(1, log_every) == 0:
            lr = optimizer.param_groups[0]["lr"]
            if logger:
                logger.log({"epoch": epoch, "step": step, "train_loss": total_loss / max(1, n), "lr": lr})

        pbar.set_postfix(loss=total_loss / max(1, n), lr=optimizer.param_groups[0]["lr"])

    # If epoch-based scheduler (e.g., StepLR), step per epoch
    if scheduler is not None and scheduler.__class__.__name__ != "WarmupCosineLR":
        scheduler.step()

    return total_loss / max(1, n)


def fit(cfg: Dict[str, Any], model: torch.nn.Module, train_loader, val_loader) -> None:
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg)
    amp = bool(cfg.get("device", {}).get("mixed_precision", True)) and (device.type == "cuda")

    loss_fn = build_loss(cfg).to(device)

    # Only train params with requires_grad=True (important when freezing EVA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg, trainable_params)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    scaler = GradScaler(enabled=amp)

    logger = JsonlLogger(out_dir / "train_log.jsonl")
    metric_name = build_metric(cfg)

    best_metric = None
    best_mode = cfg.get("eval", {}).get("best_mode", "max")
    save_best = bool(cfg.get("eval", {}).get("save_best", True))

    for epoch in range(int(cfg["train"]["epochs"])):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            amp=amp,
            scaler=scaler,
            scheduler=scheduler,
            grad_clip_norm=float(cfg["train"].get("grad_clip_norm", 0.0)),
            accumulation_steps=int(cfg["train"].get("accumulation_steps", 1)),
            log_every=int(cfg["train"].get("log_every", 50)),
            epoch=epoch,
            logger=logger,
        )

        val_loss, val_metric = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            metric_name=metric_name,
            device=device,
            amp=amp,
        )

        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metric": val_metric,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(payload)

        # save last
        save_checkpoint(out_dir / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_metric=best_metric)

        # save best
        if save_best:
            current = val_metric if metric_name != "none" else (-val_loss)
            if best_metric is None:
                best_metric = current
                save_checkpoint(out_dir / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_metric=best_metric)
            else:
                improved = (current > best_metric) if best_mode == "max" else (current < best_metric)
                if improved:
                    best_metric = current
                    save_checkpoint(out_dir / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_metric=best_metric)


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    acc = cfg.get("device", {}).get("accelerator", "cuda").lower()
    if acc == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if acc == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")