from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Protocol

import pandas as pd


@dataclass
class RuleContext:
    """策略規則執行的上下文。

    - df: 必須包含 stock_id, trading_date, close 等欄位
    - now: 當日日期（date）
    - extra: 可選的額外資料（如 market regime）
    """

    df: pd.DataFrame
    now: pd.Timestamp
    extra: Dict[str, object] = field(default_factory=dict)


class Rule(Protocol):
    name: str

    def apply(self, ctx: RuleContext) -> pd.Series:
        """回傳 bool Series（index 對齊 df）"""


class Strategy(Protocol):
    name: str
    entry_rules: List[Rule]
    filter_rules: List[Rule]
    exit_rules: List[Rule]
    rank_col: str
    rank_ascending: bool
    stoploss_fixed_pct: float

    def select_candidates(self, ctx: RuleContext) -> pd.DataFrame:
        """回傳符合 entry+filter 的候選股票（含 stock_id）"""


class FunctionalRule:
    """最小可用 Rule 實作：用函式產生布林序列。"""

    def __init__(self, name: str, fn: Callable[[RuleContext], pd.Series]):
        self.name = name
        self._fn = fn

    def apply(self, ctx: RuleContext) -> pd.Series:
        return self._fn(ctx)


def apply_rules_all(rules: Iterable[Rule], ctx: RuleContext) -> pd.Series:
    """AND 結合多個規則。"""
    if not rules:
        return pd.Series([True] * len(ctx.df), index=ctx.df.index)
    mask = None
    for rule in rules:
        r = rule.apply(ctx).astype(bool)
        mask = r if mask is None else (mask & r)
    return mask


def apply_rules_any(rules: Iterable[Rule], ctx: RuleContext) -> pd.Series:
    """OR 結合多個規則。"""
    if not rules:
        return pd.Series([False] * len(ctx.df), index=ctx.df.index)
    mask = None
    for rule in rules:
        r = rule.apply(ctx).astype(bool)
        mask = r if mask is None else (mask | r)
    return mask
