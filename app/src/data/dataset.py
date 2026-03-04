from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class CsvImageDataset(Dataset):
    csv_path: str
    img_root: str
    image_col: str = "path"
    label_col: Optional[str] = "label"
    transforms: Any = None

    def __post_init__(self):
        self.csv_path = str(self.csv_path)
        self.img_root = str(self.img_root)
        self.df = pd.read_csv(self.csv_path)
        if self.image_col not in self.df.columns:
            raise ValueError(f"image_col '{self.image_col}' not found in csv columns: {self.df.columns.tolist()}")
        if self.label_col is not None and self.label_col not in self.df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in csv columns: {self.df.columns.tolist()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = str(row[self.image_col])
        path = Path(self.img_root) / rel_path

        img = Image.open(path).convert("RGB")
        img = np.array(img)  # HWC, uint8

        if self.transforms is not None:
            # Albumentations returns dict with "image"
            img = self.transforms(image=img)["image"]
        else:
            # fallback: manual tensor conversion
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        item: Dict[str, Any] = {"image": img, "index": idx, "path": rel_path}

        if self.label_col is not None:
            label = row[self.label_col]
            if isinstance(label, (int, float)) and not pd.isna(label):
                item["target"] = torch.tensor(int(label), dtype=torch.long)
            else:
                item["target_raw"] = label

        return item