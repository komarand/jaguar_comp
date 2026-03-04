from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import copy
import yaml


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict base with upd."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def load_config(path: str | Path, base_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load config from YAML.
    If base_path provided, merges base <- exp (exp overrides).
    """
    exp = load_yaml(path)
    if base_path is None:
        return exp
    base = load_yaml(base_path)
    return _deep_update(base, exp)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)