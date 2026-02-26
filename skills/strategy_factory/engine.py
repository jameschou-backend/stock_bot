from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Dict, List, Optional

import pandas as pd

from .base import RuleContext, Strategy, apply_rules_all, apply_rules_any
from .portfolio import Portfolio, apply_pyramiding, execute_order, risk_position_size


@dataclass
class StrategyAllocation:
    strategy: Strategy
    weight: float


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    transaction_cost_pct: float = 0.001425
    min_fee: float = 20.0
    slippage_pct: float = 0.001
    risk_per_trade: float = 0.01
    position_size_multiplier: float = 1.0
    target_exposure_pct: float = 1.0
    max_positions: int = 6
    rebalance_freq: str = "D"  # D/W/M
    board_lot_shares: int = 1000
    min_trade_shares: int = 100
    min_hold_days: int = 1
    stoploss_fixed_pct: float = 0.07
    min_notional_per_trade: float = 1_000.0
    max_pyramiding_level: int = 1


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

    def _calc_fee(self, qty: float, price: float) -> float:
        notional = qty * price
        if notional <= 0:
            return 0.0
        fee = math.ceil(notional * self.config.transaction_cost_pct)
        min_fee = math.ceil(self.config.min_fee)
        return float(max(fee, min_fee))

    def _price_tick(self, price: float) -> float:
        if price < 100:
            return 0.01
        if price < 1000:
            return 0.1
        return 5.0

    def _round_to_valid_price(self, price: float, action: str) -> float:
        if price <= 0:
            return 0.0
        tick = self._price_tick(price)
        ratio = price / tick
        if action in {"BUY", "ADD"}:
            stepped = math.ceil(ratio - 1e-12) * tick
        else:
            stepped = math.floor(ratio + 1e-12) * tick
        if tick == 0.01:
            return round(stepped, 2)
        if tick == 0.1:
            return round(stepped, 1)
        return round(stepped, 0)

    def _slipped_price(self, raw_price: float, action: str) -> float:
        if action in {"BUY", "ADD"}:
            slipped = raw_price * (1 + self.config.slippage_pct)
        else:
            slipped = raw_price * (1 - self.config.slippage_pct)
        return self._round_to_valid_price(slipped, action)

    def _normalize_qty(self, qty: float) -> int:
        q = int(qty)
        if q <= 0:
            return 0
        lot = max(int(self.config.board_lot_shares), 1)
        min_shares = max(int(self.config.min_trade_shares), 1)
        min_shares = min(min_shares, lot)

        if q < min_shares:
            return 0
        # 優先整張；不足一張時退而求其次使用 100 股單位。
        if q >= lot:
            return (q // lot) * lot
        return (q // min_shares) * min_shares

    def run(self, price_df: pd.DataFrame, allocations: List[StrategyAllocation]) -> Dict:
        df = price_df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df = df.sort_values(["trading_date", "stock_id"])

        trading_dates = sorted(df["trading_date"].unique())
        day_index_map = {d: idx for idx, d in enumerate(trading_dates)}
        next_day_map = {
            trading_dates[i]: trading_dates[i + 1] for i in range(len(trading_dates) - 1)
        }
        rebalance_dates = set(self._rebalance_dates(trading_dates))

        portfolio = Portfolio(self.config.initial_capital, self.config.initial_capital)
        equity_curve = []
        pending_buys: Dict[pd.Timestamp, List[Dict]] = {}

        for current_date in trading_dates:
            current_day_index = day_index_map[current_date]
            day_df = df[df["trading_date"] == current_date]
            # 先執行「前一日訊號 -> 本日開盤」的買進單
            for order in pending_buys.pop(current_date, []):
                sid = order["stock_id"]
                if portfolio.has_stock(sid):
                    continue
                if portfolio.can_open_more(self.config.max_positions) is False:
                    break
                match = day_df[day_df["stock_id"] == sid]
                if match.empty:
                    continue
                open_px = float(match.iloc[0].get("open", match.iloc[0]["close"]))
                entry_price = self._slipped_price(open_px, "BUY")
                if entry_price <= 0:
                    continue
                qty = int(order["qty"])
                qty = self._normalize_qty(qty)
                if qty <= 0:
                    continue
                if qty * entry_price < self.config.min_notional_per_trade:
                    continue
                fee = self._calc_fee(qty, entry_price)
                if portfolio.cash < qty * entry_price + fee:
                    continue
                execute_order(
                    portfolio,
                    order["strategy_name"],
                    sid,
                    "BUY",
                    qty,
                    entry_price,
                    fee,
                    trade_date=current_date,
                    entry_index=current_day_index,
                )
                self.trades.append(
                    {
                        "trading_date": current_date.date(),
                        "stock_id": sid,
                        "strategy_name": order["strategy_name"],
                        "action": "BUY",
                        "qty": qty,
                        "price": entry_price,
                        "fee": fee,
                        "reason": order["reason"],
                    }
                )

            price_map = dict(zip(day_df["stock_id"], day_df["close"]))
            prev_max = {key: pos.max_price for key, pos in portfolio.positions.items()}
            portfolio.update_prices(price_map)

            # 先出場
            if portfolio.positions:
                for alloc in allocations:
                    positions = portfolio.positions_for_strategy(alloc.strategy.name)
                    if not positions:
                        continue
                    eligible_stock_ids = []
                    for sid, pos in positions.items():
                        if pos.entry_index is not None:
                            held_days = current_day_index - pos.entry_index
                        elif pos.entry_date is not None:
                            held_days = (current_date.date() - pos.entry_date.date()).days
                        else:
                            held_days = self.config.min_hold_days
                        if held_days >= self.config.min_hold_days:
                            eligible_stock_ids.append(sid)
                    if not eligible_stock_ids:
                        continue
                    held_df = day_df[day_df["stock_id"].isin(eligible_stock_ids)].copy()
                    if held_df.empty:
                        continue
                    ctx = RuleContext(
                        df=held_df,
                        now=current_date,
                        extra={
                            "positions": positions,
                            "current_day_index": current_day_index,
                        },
                    )
                    if not alloc.strategy.exit_rules:
                        continue
                    exit_reason_map: Dict[str, List[str]] = {}
                    exit_mask = pd.Series([False] * len(held_df), index=held_df.index)
                    for rule in alloc.strategy.exit_rules:
                        rule_mask = rule.apply(ctx).astype(bool)
                        exit_mask = exit_mask | rule_mask
                        for idx, flag in rule_mask.items():
                            if not flag:
                                continue
                            sid = held_df.loc[idx, "stock_id"]
                            exit_reason_map.setdefault(sid, []).append(rule.name)
                    if exit_mask is not None and not held_df.empty:
                        for _, row in held_df[exit_mask].iterrows():
                            sid = row["stock_id"]
                            price = self._slipped_price(float(row["close"]), "SELL")
                            if price <= 0:
                                continue
                            pos = positions.get(sid)
                            if not pos:
                                continue
                            qty = pos.qty
                            fee = self._calc_fee(qty, price)
                            avg_cost_before_sell = pos.avg_cost
                            realized_pnl = (price - avg_cost_before_sell) * qty - fee
                            execute_order(
                                portfolio,
                                alloc.strategy.name,
                                sid,
                                "SELL",
                                qty,
                                price,
                                fee,
                                trade_date=current_date,
                            )
                            self.trades.append(
                                {
                                    "trading_date": current_date.date(),
                                    "stock_id": sid,
                                    "strategy_name": alloc.strategy.name,
                                    "action": "SELL",
                                    "qty": qty,
                                    "price": price,
                                    "fee": fee,
                                    "realized_pnl": realized_pnl,
                                    "avg_cost": avg_cost_before_sell,
                                    "reason": {
                                        "code": "exit_rule",
                                        "detail": exit_reason_map.get(sid, ["exit_rule"]),
                                    },
                                }
                            )

            # 加碼（pyramiding）
            if portfolio.positions:
                for alloc in allocations:
                    positions = portfolio.positions_for_strategy(alloc.strategy.name)
                    if not positions:
                        continue
                    for sid, pos in positions.items():
                        if pos.pyramiding_level >= self.config.max_pyramiding_level:
                            continue
                        last_high = prev_max.get((alloc.strategy.name, sid)) or pos.max_price or pos.last_price
                        add_ratio = apply_pyramiding(pos, last_high, pos.last_price)
                        if add_ratio <= 0:
                            continue
                        add_qty = pos.base_qty * add_ratio
                        add_qty = self._normalize_qty(add_qty)
                        if add_qty <= 0:
                            continue
                        add_price = self._slipped_price(float(pos.last_price), "ADD")
                        if add_price <= 0:
                            continue
                        if add_qty * add_price < self.config.min_notional_per_trade:
                            continue
                        fee = self._calc_fee(add_qty, add_price)
                        if portfolio.cash < add_qty * add_price + fee:
                            continue
                        execute_order(
                            portfolio,
                            alloc.strategy.name,
                            sid,
                            "ADD",
                            add_qty,
                            add_price,
                            fee,
                            trade_date=current_date,
                        )
                        self.trades.append(
                            {
                                "trading_date": current_date.date(),
                                "stock_id": sid,
                                "strategy_name": alloc.strategy.name,
                                "action": "ADD",
                                "qty": add_qty,
                                "price": add_price,
                                "fee": fee,
                                "reason": {
                                    "code": "pyramiding",
                                    "detail": [f"level_{pos.pyramiding_level}"],
                                },
                            }
                        )

            # 進場（僅在再平衡日）
            if current_date in rebalance_dates:
                next_date = next_day_map.get(current_date)
                if next_date is not None:
                    total_value = portfolio.total_value()
                    current_exposure = max(total_value - portfolio.cash, 0.0)
                    target_exposure_value = total_value * max(min(self.config.target_exposure_pct, 1.0), 0.0)
                    remaining_total_budget = max(target_exposure_value - current_exposure, 0.0)
                    if remaining_total_budget <= 0:
                        continue
                    for alloc in allocations:
                        reserved_orders = pending_buys.get(next_date, [])
                        reserved_sids = {order["stock_id"] for order in reserved_orders}
                        slots_left = self.config.max_positions - len(portfolio.positions) - len(reserved_sids)
                        if slots_left <= 0:
                            break
                        ctx = RuleContext(df=day_df, now=current_date, extra={"positions": portfolio.positions})
                        entry_mask = apply_rules_all(alloc.strategy.entry_rules, ctx)
                        filter_mask = apply_rules_all(alloc.strategy.filter_rules, ctx)
                        candidates = day_df[entry_mask & filter_mask].copy()
                        if candidates.empty:
                            continue

                        # 可用資金分配給策略
                        target_value = target_exposure_value * alloc.weight
                        current_value = portfolio.value_for_strategy(alloc.strategy.name)
                        cash_budget = min(
                            portfolio.cash,
                            remaining_total_budget,
                            max(target_value - current_value, 0.0),
                        )
                        if cash_budget <= 0:
                            continue

                        rank_col = getattr(alloc.strategy, "rank_col", None)
                        rank_ascending = getattr(alloc.strategy, "rank_ascending", False)
                        if rank_col and rank_col in candidates.columns:
                            candidates = candidates.sort_values(rank_col, ascending=rank_ascending)

                        for _, row in candidates.iterrows():
                            if slots_left <= 0:
                                break
                            sid = row["stock_id"]
                            if portfolio.has_stock(sid):
                                continue
                            if sid in reserved_sids:
                                continue
                            estimate_entry_price = self._slipped_price(float(row["close"]), "BUY")
                            if estimate_entry_price <= 0:
                                continue
                            stoploss_pct = getattr(alloc.strategy, "stoploss_fixed_pct", self.config.stoploss_fixed_pct)
                            stop_price = estimate_entry_price * (1 - stoploss_pct)
                            qty = risk_position_size(
                                cash_budget,
                                self.config.risk_per_trade,
                                estimate_entry_price,
                                stop_price,
                            )
                            qty *= max(self.config.position_size_multiplier, 0.0)
                            max_affordable = cash_budget / estimate_entry_price
                            qty = min(qty, max_affordable)
                            qty = self._normalize_qty(qty)
                            if qty <= 0:
                                continue
                            if qty * estimate_entry_price < self.config.min_notional_per_trade:
                                continue
                            estimate_fee = self._calc_fee(qty, estimate_entry_price)
                            cash_budget -= qty * estimate_entry_price + estimate_fee

                            pending_buys.setdefault(next_date, []).append(
                                {
                                    "strategy_name": alloc.strategy.name,
                                    "stock_id": sid,
                                    "qty": qty,
                                    "reason": {
                                        "code": "entry_rule",
                                        "detail": [r.name for r in alloc.strategy.entry_rules],
                                    },
                                }
                            )
                            reserved_sids.add(sid)
                            slots_left -= 1

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
                        "strategy_name": pos.strategy_name,
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
