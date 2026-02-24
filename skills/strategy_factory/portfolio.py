from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Position:
    stock_id: str
    strategy_name: str
    qty: float
    avg_cost: float
    last_price: float
    entry_date: Optional[pd.Timestamp] = None
    entry_index: Optional[int] = None
    max_price: Optional[float] = None
    base_qty: float = 0.0
    pyramiding_level: int = 0

    def market_value(self) -> float:
        return self.qty * self.last_price

    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.qty


@dataclass
class Portfolio:
    initial_capital: float
    cash: float
    positions: Dict[tuple, Position] = field(default_factory=dict)

    def total_value(self) -> float:
        return self.cash + sum(p.market_value() for p in self.positions.values())

    def update_prices(self, price_map: Dict[str, float]) -> None:
        for key, pos in self.positions.items():
            if pos.stock_id in price_map:
                pos.last_price = price_map[pos.stock_id]
                if pos.max_price is None or pos.last_price > pos.max_price:
                    pos.max_price = pos.last_price

    def can_open_more(self, max_positions: int) -> bool:
        return len(self.positions) < max_positions

    def has_stock(self, stock_id: str) -> bool:
        return any(pos.stock_id == stock_id for pos in self.positions.values())

    def positions_for_strategy(self, strategy_name: str) -> Dict[str, Position]:
        return {pos.stock_id: pos for pos in self.positions.values() if pos.strategy_name == strategy_name}

    def value_for_strategy(self, strategy_name: str) -> float:
        return sum(pos.market_value() for pos in self.positions.values() if pos.strategy_name == strategy_name)


def risk_position_size(
    cash: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """用風險部位計算可買數量（以風險金額/每股風險）。"""
    risk_amount = cash * risk_per_trade
    per_share_risk = max(entry_price - stop_price, 0.0)
    if per_share_risk <= 0:
        return 0.0
    return risk_amount / per_share_risk


def apply_pyramiding(
    pos: Position,
    last_high: float,
    current_price: float,
    thresholds: Tuple[float, float] = (0.05, 0.10),
) -> float:
    """判斷是否加碼，回傳加碼比例。

    規則：
    - +5% 且突破前高 -> 加 30%
    - 再 +5% -> 加 20%
    - 僅在獲利狀態加碼
    """
    if pos.avg_cost <= 0:
        return 0.0
    if current_price <= pos.avg_cost:
        return 0.0
    gain = current_price / pos.avg_cost - 1
    if pos.pyramiding_level == 0 and gain >= thresholds[0] and current_price > last_high:
        pos.pyramiding_level = 1
        return 0.30
    if pos.pyramiding_level == 1 and gain >= thresholds[1]:
        pos.pyramiding_level = 2
        return 0.20
    return 0.0


def execute_order(
    portfolio: Portfolio,
    strategy_name: str,
    stock_id: str,
    action: str,
    qty: float,
    price: float,
    fee: float,
    trade_date: Optional[pd.Timestamp] = None,
    entry_index: Optional[int] = None,
) -> None:
    key = (strategy_name, stock_id)
    if action == "BUY" or action == "ADD":
        cost = qty * price + fee
        portfolio.cash -= cost
        if key in portfolio.positions:
            pos = portfolio.positions[key]
            total_cost = pos.avg_cost * pos.qty + qty * price
            pos.qty += qty
            if pos.qty <= 0:
                # 防呆：避免異常資料導致除以 0；清掉異常持倉。
                del portfolio.positions[key]
                return
            pos.avg_cost = total_cost / pos.qty
            pos.last_price = price
            if pos.max_price is None or price > pos.max_price:
                pos.max_price = price
        else:
            portfolio.positions[key] = Position(
                stock_id=stock_id,
                strategy_name=strategy_name,
                qty=qty,
                avg_cost=price,
                last_price=price,
                entry_date=trade_date,
                entry_index=entry_index,
                max_price=price,
                base_qty=qty,
            )
    elif action == "SELL":
        if key not in portfolio.positions:
            return
        pos = portfolio.positions[key]
        qty = min(qty, pos.qty)
        proceeds = qty * price - fee
        portfolio.cash += proceeds
        pos.qty -= qty
        pos.last_price = price
        if pos.qty <= 0:
            del portfolio.positions[key]
