"""Portfolio tracking helper：讀 portfolio.json + 計算實時 P&L + 持有期換股提醒。"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PORTFOLIO_PATH = PROJECT_ROOT / "portfolio.json"

# 策略本質：持有 20 個交易日 (~ 1 個月)
HOLD_TARGET_TRADING_DAYS = 20
HOLD_TARGET_CALENDAR_DAYS = 28  # 緩衝（含週末，~4 calendar weeks）


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
    today = date.today()
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

        # 持有期計算（calendar days，approx trading_days = calendar_days × 0.72）
        entry_dt_str = info.get("entry_date", "")
        holding_days = 0
        hold_status = "—"
        if entry_dt_str:
            try:
                entry_dt = datetime.strptime(entry_dt_str, "%Y-%m-%d").date()
                holding_days = (today - entry_dt).days
                if holding_days >= HOLD_TARGET_CALENDAR_DAYS:
                    hold_status = "🔄 到期可換股"
                elif holding_days >= HOLD_TARGET_CALENDAR_DAYS - 5:
                    hold_status = f"⏰ 接近換股期 ({holding_days}d)"
                else:
                    days_left = HOLD_TARGET_CALENDAR_DAYS - holding_days
                    hold_status = f"⏳ 持有 {holding_days}d (剩 ~{days_left}d)"
            except Exception:
                hold_status = entry_dt_str

        rows.append({
            "stock_id": sid,
            "shares": shares,
            "entry_price": entry,
            "current_price": float(current) if current else None,
            "cost": cost,
            "market_value": mkt_val,
            "unreal_pnl": unreal_pnl,
            "return_pct": ret_pct,
            "entry_date": entry_dt_str,
            "holding_days": holding_days,
            "hold_status": hold_status,
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
