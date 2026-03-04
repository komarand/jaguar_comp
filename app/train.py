from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.utils.config import load_config, save_config
from src.utils.seed import seed_everything
from src.data.datamodule import build_loaders
from src.models.factory import build_model
from src.train.trainer import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to experiment yaml")
    ap.add_argument("--base", type=str, default=None, help="optional base yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, base_path=args.base)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    seed_everything(seed)

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    # Resolve device early
    device_cfg = cfg.get("device", {})
    acc = device_cfg.get("accelerator", "cuda").lower()
    if acc == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif acc == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(cfg, device=device).to(device)

    fit(cfg, model, train_loader, val_loader)


if __name__ == "__main__":
    main()