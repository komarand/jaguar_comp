from __future__ import annotations

from typing import Callable, Dict, TypeVar

T = TypeVar("T")

BACKBONES: Dict[str, Callable] = {}
HEADS: Dict[str, Callable] = {}
LOSSES: Dict[str, Callable] = {}
METRICS: Dict[str, Callable] = {}


def register(registry: Dict[str, Callable], name: str):
    def deco(fn: Callable) -> Callable:
        registry[name] = fn
        return fn
    return deco