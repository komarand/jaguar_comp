from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import timm

from .heads import HEADS  # ensure registration
from .registry import HEADS as HEADS_REG


class ModelWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, mode: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # timm create_model with num_classes=0 usually returns features (B, C)
        out = self.head(feats)
        return out


def _infer_backbone_feat_dim(backbone: nn.Module, img_size: int, in_chans: int, device: torch.device) -> int:
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(2, in_chans, img_size, img_size, device=device)
        y = backbone(x)
        if y.ndim != 2:
            # some models return (B, C, ...) if misconfigured
            y = y.flatten(1)
        return int(y.shape[1])


def freeze_backbone(model: nn.Module, freeze_regex: str = "") -> None:
    if freeze_regex:
        rgx = re.compile(freeze_regex)
        for n, p in model.named_parameters():
            if rgx.search(n):
                p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = False


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    tcfg = cfg["task"]
    mcfg = cfg["model"]

    mode = tcfg.get("mode", "classification")
    backbone_name = mcfg["backbone"]
    pretrained = bool(mcfg.get("pretrained", True))
    img_size = int(mcfg.get("img_size", 224))
    in_chans = int(mcfg.get("in_chans", 3))
    drop_rate = float(mcfg.get("drop_rate", 0.0))
    drop_path_rate = float(mcfg.get("drop_path_rate", 0.0))

    # Make a timm backbone that outputs features (num_classes=0)
    backbone = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
        in_chans=in_chans,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    feat_dim = _infer_backbone_feat_dim(backbone, img_size=img_size, in_chans=in_chans, device=device)

    head_cfg = mcfg.get("head", {})
    head_name = head_cfg.get("name", "linear")

    if mode == "classification":
        out_dim = int(head_cfg.get("out_dim", tcfg["num_classes"]))
    else:
        out_dim = int(head_cfg.get("out_dim", 512))

    head_builder = HEADS_REG.get(head_name)
    if head_builder is None:
        raise ValueError(f"Unknown head '{head_name}'. Available: {list(HEADS_REG.keys())}")

    head = head_builder(
        in_dim=feat_dim,
        out_dim=out_dim,
        hidden_dim=int(head_cfg.get("hidden_dim", 1024)),
        dropout=float(head_cfg.get("dropout", 0.0)),
    )

    model = ModelWithHead(backbone=backbone, head=head, mode=mode)

    if bool(mcfg.get("freeze_backbone", False)):
        freeze_backbone(model.backbone)

    fr = str(mcfg.get("freeze_backbone_regex", "")).strip()
    if fr:
        freeze_backbone(model, freeze_regex=fr)

    return model