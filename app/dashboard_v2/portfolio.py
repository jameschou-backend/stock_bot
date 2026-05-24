"""Portfolio tracking helper：讀 portfolio.json + 計算實時 P&L。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PORTFOLIO_PATH = PROJECT_ROOT / "portfolio.json"


def load_portfolio() -> Optional[dict]:
    """從 portfolio.json 讀持倉。沒檔回 None。"""
    if not PORTFOLIO_PATH.exists():
        return None
    try:
        return json.load(open(PORTFOLIO_PATH, encoding="utf-8"))
    except Exception:
        return None


def compute_pnl(portfolio: dict, price_map: dict) -> dict:
    """算每檔持倉的當前 P&L + 整體統計。

    Args:
        portfolio: load_portfolio() 結果
        price_map: stock_id → latest close

    Returns:
        dict 含 positions (DataFrame), totals (dict)
    """
    positions = portfolio.get("positions", {})
    cash = float(portfolio.get("cash", 0))
    rows = []
    total_cost = 0.0
    total_value = 0.0
    for sid, info in positions.items():
        shares = int(info["shares"])
        entry = float(info["entry_price"])
        cost = shares * entry
        current = price_map.get(str(sid))
        if current is None:
            mkt_val = cost
            unreal_pnl = 0
            ret_pct = 0
        else:
            mkt_val = shares * float(current)
            unreal_pnl = mkt_val - cost
            ret_pct = unreal_pnl / cost if cost > 0 else 0
        total_cost += cost
        total_value += mkt_val
        rows.append({
            "stock_id": sid,
            "shares": shares,
            "entry_price": entry,
            "current_price": float(current) if current else None,
            "cost": cost,
            "market_value": mkt_val,
            "unreal_pnl": unreal_pnl,
            "return_pct": ret_pct,
            "entry_date": info.get("entry_date", ""),
        })

    pos_df = pd.DataFrame(rows)
    totals = {
        "n_positions": len(rows),
        "total_cost": total_cost,
        "total_market_value": total_value,
        "cash": cash,
        "total_assets": total_value + cash,
        "unreal_pnl": total_value - total_cost,
        "unreal_pnl_pct": (total_value - total_cost) / total_cost if total_cost > 0 else 0,
    }
    return {"positions": pos_df, "totals": totals}


def compute_picks_alignment(portfolio: dict, picks_df: pd.DataFrame) -> dict:
    """計算手上持倉跟今日 picks 的對齊度。

    Returns:
        in_picks (持倉且在 today picks)
        missing (持倉但不在 today picks → 該賣)
        new_picks (在 today picks 但未持倉 → 該買)
    """
    holdings = set(str(s) for s in portfolio.get("positions", {}).keys())
    today_picks = set(picks_df["stock_id"].astype(str).tolist())

    in_picks = sorted(holdings & today_picks)
    missing = sorted(holdings - today_picks)  # 該賣
    new_picks = sorted(today_picks - holdings)  # 該買
    return {
        "in_picks": in_picks,
        "missing": missing,
        "new_picks": new_picks,
        "alignment_pct": len(in_picks) / max(len(holdings), 1),
    }
