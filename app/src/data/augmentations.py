from __future__ import annotations

from typing import Any, Dict, List

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(augs_cfg: List[Dict[str, Any]]) -> A.Compose:
    """
    Build Albumentations pipeline from YAML list.

    Supported ops (examples):
      - RandomResizedCrop, Resize, CenterCrop
      - HorizontalFlip, VerticalFlip
      - ColorJitter
      - RandomBrightnessContrast
      - HueSaturationValue
      - RGBShift
      - Blur, MotionBlur, GaussianBlur
      - GaussNoise
      - CoarseDropout
      - Normalize
      - ToTensorV2

    Notes:
      - Albumentations works with numpy images (H, W, C) in uint8 usually.
      - Use Normalize + ToTensorV2 for PyTorch tensors.
    """
    ops = []

    for a in augs_cfg:
        name = a["name"]

        if name == "RandomResizedCrop":
            # Albumentations API: height, width
            size = int(a["size"])
            scale = a.get("scale", [0.8, 1.0])
            ratio = a.get("ratio", [0.75, 1.3333333333])
            p = float(a.get("p", 1.0))
            ops.append(
                A.RandomResizedCrop(
                    height=size,
                    width=size,
                    scale=tuple(scale),
                    ratio=tuple(ratio),
                    p=p,
                )
            )

        elif name == "Resize":
            size = a["size"]
            p = float(a.get("p", 1.0))
            if isinstance(size, list):
                h, w = int(size[0]), int(size[1])
            else:
                h = w = int(size)
            ops.append(A.Resize(height=h, width=w, p=p))

        elif name == "CenterCrop":
            size = int(a["size"])
            p = float(a.get("p", 1.0))
            ops.append(A.CenterCrop(height=size, width=size, p=p))

        elif name == "HorizontalFlip":
            p = float(a.get("p", 0.5))
            ops.append(A.HorizontalFlip(p=p))

        elif name == "VerticalFlip":
            p = float(a.get("p", 0.5))
            ops.append(A.VerticalFlip(p=p))

        elif name == "ColorJitter":
            p = float(a.get("p", 0.3))
            ops.append(
                A.ColorJitter(
                    brightness=float(a.get("brightness", 0.2)),
                    contrast=float(a.get("contrast", 0.2)),
                    saturation=float(a.get("saturation", 0.2)),
                    hue=float(a.get("hue", 0.1)),
                    p=p,
                )
            )

        elif name == "RandomBrightnessContrast":
            p = float(a.get("p", 0.2))
            ops.append(
                A.RandomBrightnessContrast(
                    brightness_limit=float(a.get("brightness_limit", 0.2)),
                    contrast_limit=float(a.get("contrast_limit", 0.2)),
                    p=p,
                )
            )

        elif name == "HueSaturationValue":
            p = float(a.get("p", 0.2))
            ops.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(a.get("hue_shift_limit", 10)),
                    sat_shift_limit=int(a.get("sat_shift_limit", 15)),
                    val_shift_limit=int(a.get("val_shift_limit", 10)),
                    p=p,
                )
            )

        elif name == "RGBShift":
            p = float(a.get("p", 0.1))
            ops.append(
                A.RGBShift(
                    r_shift_limit=int(a.get("r_shift_limit", 10)),
                    g_shift_limit=int(a.get("g_shift_limit", 10)),
                    b_shift_limit=int(a.get("b_shift_limit", 10)),
                    p=p,
                )
            )

        elif name == "Blur":
            p = float(a.get("p", 0.1))
            ops.append(A.Blur(blur_limit=int(a.get("blur_limit", 3)), p=p))

        elif name == "MotionBlur":
            p = float(a.get("p", 0.1))
            ops.append(A.MotionBlur(blur_limit=int(a.get("blur_limit", 7)), p=p))

        elif name == "GaussianBlur":
            p = float(a.get("p", 0.1))
            ops.append(A.GaussianBlur(blur_limit=int(a.get("blur_limit", 3)), p=p))

        elif name == "GaussNoise":
            p = float(a.get("p", 0.1))
            ops.append(
                A.GaussNoise(
                    var_limit=tuple(a.get("var_limit", [10.0, 50.0])),
                    p=p,
                )
            )

        elif name == "CoarseDropout":
            p = float(a.get("p", 0.1))
            ops.append(
                A.CoarseDropout(
                    max_holes=int(a.get("max_holes", 8)),
                    max_height=int(a.get("max_height", 32)),
                    max_width=int(a.get("max_width", 32)),
                    min_holes=int(a.get("min_holes", 1)),
                    min_height=int(a.get("min_height", 8)),
                    min_width=int(a.get("min_width", 8)),
                    fill_value=a.get("fill_value", 0),
                    p=p,
                )
            )

        elif name == "Normalize":
            mean = a.get("mean", [0.485, 0.456, 0.406])
            std = a.get("std", [0.229, 0.224, 0.225])
            p = float(a.get("p", 1.0))
            ops.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=p))

        elif name in ("ToTensor", "ToTensorV2"):
            # allow both names in YAML for convenience
            ops.append(ToTensorV2())

        else:
            raise ValueError(f"Unknown Albumentations op: {name}")

    return A.Compose(ops)


def build_stage_transforms(cfg: Dict[str, Any], stage: str) -> A.Compose:
    augs_cfg = cfg.get("augmentations", {}).get(stage, [])
    if not augs_cfg:
        # sensible default: normalize + tensor
        return A.Compose([A.Normalize(), ToTensorV2()])
    return build_transforms(augs_cfg)