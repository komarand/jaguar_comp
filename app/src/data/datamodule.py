from __future__ import annotations

from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader

from .augmentations import build_stage_transforms
from .dataset import CsvImageDataset


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]

    train_tfms = build_stage_transforms(cfg, "train")
    val_tfms = build_stage_transforms(cfg, "val")

    train_ds = CsvImageDataset(
        csv_path=dcfg["train_csv"],
        img_root=dcfg["img_root"],
        image_col=dcfg.get("image_col", "path"),
        label_col=dcfg.get("label_col", "label"),
        transforms=train_tfms,
    )
    val_ds = CsvImageDataset(
        csv_path=dcfg["val_csv"],
        img_root=dcfg["img_root"],
        image_col=dcfg.get("image_col", "path"),
        label_col=dcfg.get("label_col", "label"),
        transforms=val_tfms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(dcfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=bool(dcfg.get("pin_memory", True)),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(dcfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=bool(dcfg.get("pin_memory", True)),
        drop_last=False,
    )
    return train_loader, val_loader