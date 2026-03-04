from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..utils.checkpointing import load_checkpoint


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    return_paths: bool = True,
) -> Dict[str, Any]:
    model.eval()

    outputs: List[np.ndarray] = []
    paths: List[str] = []

    pbar = tqdm(loader, desc="predict", leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        x = batch["image"]
        with autocast(enabled=amp and device.type == "cuda"):
            out = model(x)
        out = out.float().detach().cpu().numpy()
        outputs.append(out)

        if return_paths and "path" in batch:
            # path is list of strings (not tensor)
            paths.extend(list(batch["path"]))

    y = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)
    res = {"outputs": y}
    if return_paths:
        res["paths"] = paths
    return res


def load_model_weights(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device) -> Dict[str, Any]:
    return load_checkpoint(ckpt_path, model=model, map_location=str(device), strict=True)