from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.data.augmentations import build_stage_transforms
from src.data.dataset import CsvImageDataset
from src.models.factory import build_model
from src.infer.predict import load_model_weights, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--base", type=str, default=None)
    ap.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pt")
    ap.add_argument("--csv", type=str, required=True, help="csv to infer")
    ap.add_argument("--out", type=str, required=True, help="output .npy path")
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, base_path=args.base)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # device
    acc = cfg.get("device", {}).get("accelerator", "cuda").lower()
    if acc == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif acc == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(cfg, device=device).to(device)
    load_model_weights(model, args.ckpt, device=device)

    dcfg = cfg["data"]
    tfms = build_stage_transforms(cfg, "val")  # usually val/test tfms

    ds = CsvImageDataset(
        csv_path=args.csv,
        img_root=dcfg["img_root"],
        image_col=dcfg.get("image_col", "path"),
        label_col=None,  # inference
        transforms=tfms,
    )

    bs = args.batch_size if args.batch_size is not None else int(dcfg.get("batch_size", 64))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=bool(dcfg.get("pin_memory", True)),
        drop_last=False,
    )

    amp = bool(cfg.get("device", {}).get("mixed_precision", True))
    res = predict(model, loader, device=device, amp=amp, return_paths=True)

    np.save(str(out_path), res["outputs"])
    # paths can be saved separately if needed:
    # (out_path.parent / (out_path.stem + "_paths.txt")).write_text("\n".join(res["paths"]), encoding="utf-8")


if __name__ == "__main__":
    main()