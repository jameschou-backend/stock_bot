from __future__ import annotations

import pandas as pd

from .base import FunctionalRule, RuleContext


def exit_below_ma(col_ma: str = "ma_20"):
    def _fn(ctx: RuleContext) -> pd.Series:
        return ctx.df["close"] < ctx.df[col_ma]

    return FunctionalRule(f"close_below_{col_ma}", _fn)


def _position_return(ctx: RuleContext) -> pd.Series:
    positions = ctx.extra.get("positions", {})
    returns = []
    for _, row in ctx.df.iterrows():
        pos = positions.get(row["stock_id"])
        if not pos:
            returns.append(0.0)
        else:
            if pos.avg_cost and pos.avg_cost > 0:
                returns.append(float(row["close"]) / pos.avg_cost - 1)
            else:
                returns.append(0.0)
    return pd.Series(returns, index=ctx.df.index)


def exit_take_profit_fixed(threshold: float = 0.2):
    def _fn(ctx: RuleContext) -> pd.Series:
        return _position_return(ctx) >= threshold

    return FunctionalRule(f"takeprofit_fixed_{threshold}", _fn)


def exit_stop_loss_fixed(threshold: float = -0.07):
    def _fn(ctx: RuleContext) -> pd.Series:
        return _position_return(ctx) <= threshold

    return FunctionalRule(f"stoploss_fixed_{threshold}", _fn)


def exit_trailing_stop(trailing_pct: float = 0.1):
    def _fn(ctx: RuleContext) -> pd.Series:
        positions = ctx.extra.get("positions", {})
        flags = []
        for _, row in ctx.df.iterrows():
            pos = positions.get(row["stock_id"])
            if not pos or not pos.max_price:
                flags.append(False)
                continue
            flags.append(float(row["close"]) <= pos.max_price * (1 - trailing_pct))
        return pd.Series(flags, index=ctx.df.index)

    return FunctionalRule(f"trailing_stop_{trailing_pct}", _fn)


def exit_time_stop(days: int = 10):
    def _fn(ctx: RuleContext) -> pd.Series:
        positions = ctx.extra.get("positions", {})
        current_index = ctx.extra.get("current_day_index")
        flags = []
        for _, row in ctx.df.iterrows():
            pos = positions.get(row["stock_id"])
            if not pos or pos.entry_date is None:
                flags.append(False)
                continue
            if current_index is not None and pos.entry_index is not None:
                held_days = current_index - pos.entry_index
            else:
                held_days = (ctx.now.date() - pos.entry_date.date()).days
            flags.append(held_days >= days)
        return pd.Series(flags, index=ctx.df.index)

    return FunctionalRule(f"time_stop_{days}", _fn)
