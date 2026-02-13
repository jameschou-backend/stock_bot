from __future__ import annotations

import pandas as pd

from .base import FunctionalRule, RuleContext


def entry_close_above_ma(col_ma: str = "ma_60"):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df["close"] > ctx.df[col_ma]

    return FunctionalRule(f"close_above_{col_ma}", _fn)


def entry_ret_above(col_ret: str = "ret_20", threshold: float = 0.1):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df[col_ret] > threshold

    return FunctionalRule(f"{col_ret}_gt_{threshold}", _fn)


def entry_rsi_below(col_rsi: str = "rsi_14", threshold: float = 30):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df[col_rsi] < threshold

    return FunctionalRule(f"{col_rsi}_lt_{threshold}", _fn)


def entry_below_bollinger(col_lower: str = "bb_lower"):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df["close"] < ctx.df[col_lower]

    return FunctionalRule(f"close_below_{col_lower}", _fn)


def entry_volume_above(col_vol: str = "volume_20", col_base: str = "volume_60_mean"):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df[col_vol] > ctx.df[col_base]

    return FunctionalRule(f"{col_vol}_gt_{col_base}", _fn)
