from __future__ import annotations

import pandas as pd

from .base import FunctionalRule, RuleContext


def filter_min_turnover(col_turnover: str = "avg_turnover_20", threshold: float = 0.5):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df[col_turnover] >= threshold

    return FunctionalRule(f"{col_turnover}_gte_{threshold}", _fn)


def filter_low_vol_percentile(col_vol_pct: str = "vol_pct_60", threshold: float = 0.3):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df[col_vol_pct] <= threshold

    return FunctionalRule(f"{col_vol_pct}_lte_{threshold}", _fn)
