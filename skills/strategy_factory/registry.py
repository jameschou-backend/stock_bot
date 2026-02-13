from __future__ import annotations

from typing import Dict, List

from .base import Strategy
from .strategies import DefensiveLowVol, MeanReversion, MomentumTrend


_REGISTRY: Dict[str, Strategy] = {}


def register(strategy: Strategy) -> None:
    _REGISTRY[strategy.name] = strategy


def get(name: str) -> Strategy:
    return _REGISTRY[name]


def list_strategies() -> List[str]:
    return sorted(_REGISTRY.keys())


def register_defaults() -> None:
    for s in [MomentumTrend(), MeanReversion(), DefensiveLowVol()]:
        register(s)
