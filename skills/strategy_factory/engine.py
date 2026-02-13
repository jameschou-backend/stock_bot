from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from .base import RuleContext, Strategy, apply_rules_all, apply_rules_any
from .portfolio import Portfolio, execute_order, risk_position_size


@dataclass
class StrategyAllocation:
    strategy: Strategy
    weight: float


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    transaction_cost_pct: float = 0.00585
    slippage_pct: float = 0.001
    risk_per_trade: float = 0.01
    max_positions: int = 6
    rebalance_freq: str = "D"  # D/W/M
    stoploss_fixed_pct: float = 0.07


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Dict] = []
        self.positions_snapshots: List[Dict] = []

    def _rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        if self.config.rebalance_freq == "D":
            return trading_dates
        dates = sorted(trading_dates)
        if self.config.rebalance_freq == "W":
            return [d for d in dates if d.weekday() == 0]
        # 月初第一個交易日
        rebal = []
        last_month = None
        for d in dates:
            ym = (d.year, d.month)
            if ym != last_month:
                rebal.append(d)
                last_month = ym
        return rebal

    def run(self, price_df: pd.DataFrame, allocations: List[StrategyAllocation]) -> Dict:
        df = price_df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df = df.sort_values(["trading_date", "stock_id"])

        trading_dates = sorted(df["trading_date"].unique())
        rebalance_dates = set(self._rebalance_dates(trading_dates))

        portfolio = Portfolio(self.config.initial_capital, self.config.initial_capital)
        equity_curve = []

        for current_date in trading_dates:
            day_df = df[df["trading_date"] == current_date]
            price_map = dict(zip(day_df["stock_id"], day_df["close"]))
            portfolio.update_prices(price_map)

            # 先出場
            if portfolio.positions:
                held_df = day_df[day_df["stock_id"].isin(portfolio.positions.keys())].copy()
                ctx = RuleContext(df=held_df, now=current_date, extra={"positions": portfolio.positions})
                exit_mask = None
                for alloc in allocations:
                    if not alloc.strategy.exit_rules:
                        continue
                    m = apply_rules_any(alloc.strategy.exit_rules, ctx)
                    exit_mask = m if exit_mask is None else (exit_mask | m)
                if exit_mask is not None and not held_df.empty:
                    for _, row in held_df[exit_mask].iterrows():
                        sid = row["stock_id"]
                        price = float(row["close"]) * (1 - self.config.slippage_pct)
                        pos = portfolio.positions.get(sid)
                        if not pos:
                            continue
                        qty = pos.qty
                        fee = qty * price * self.config.transaction_cost_pct
                        execute_order(portfolio, sid, "SELL", qty, price, fee, trade_date=current_date)
                        self.trades.append(
                            {
                                "trading_date": current_date.date(),
                                "stock_id": sid,
                                "action": "SELL",
                                "qty": qty,
                                "price": price,
                                "fee": fee,
                                "reason": "exit_rule",
                            }
                        )

            # 進場（僅在再平衡日）
            if current_date in rebalance_dates:
                for alloc in allocations:
                    if portfolio.can_open_more(self.config.max_positions) is False:
                        break
                    ctx = RuleContext(df=day_df, now=current_date, extra={"positions": portfolio.positions})
                    entry_mask = apply_rules_all(alloc.strategy.entry_rules, ctx)
                    filter_mask = apply_rules_all(alloc.strategy.filter_rules, ctx)
                    candidates = day_df[entry_mask & filter_mask]
                    if candidates.empty:
                        continue

                    # 可用資金分配給策略
                    target_value = portfolio.total_value() * alloc.weight
                    cash_budget = min(portfolio.cash, target_value)
                    if cash_budget <= 0:
                        continue

                    for _, row in candidates.iterrows():
                        if portfolio.can_open_more(self.config.max_positions) is False:
                            break
                        sid = row["stock_id"]
                        if sid in portfolio.positions:
                            continue
                        entry_price = float(row["close"]) * (1 + self.config.slippage_pct)
                        stop_price = entry_price * (1 - self.config.stoploss_fixed_pct)
                        qty = risk_position_size(cash_budget, self.config.risk_per_trade, entry_price, stop_price)
                        if qty <= 0:
                            continue
                        fee = qty * entry_price * self.config.transaction_cost_pct
                        execute_order(portfolio, sid, "BUY", qty, entry_price, fee, trade_date=current_date)
                        self.trades.append(
                            {
                                "trading_date": current_date.date(),
                                "stock_id": sid,
                                "action": "BUY",
                                "qty": qty,
                                "price": entry_price,
                                "fee": fee,
                                "reason": "entry_rule",
                            }
                        )

            equity_curve.append(
                {
                    "trading_date": current_date.date(),
                    "equity": portfolio.total_value(),
                    "cash": portfolio.cash,
                    "positions": len(portfolio.positions),
                }
            )

            # 持倉快照
            for pos in portfolio.positions.values():
                self.positions_snapshots.append(
                    {
                        "trading_date": current_date.date(),
                        "stock_id": pos.stock_id,
                        "qty": pos.qty,
                        "avg_cost": pos.avg_cost,
                        "market_value": pos.market_value(),
                        "unrealized_pnl": pos.unrealized_pnl(),
                    }
                )

        return {
            "equity_curve": equity_curve,
            "trades": self.trades,
            "positions": self.positions_snapshots,
        }
