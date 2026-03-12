#!/usr/bin/env python3
"""
trade_dashboard.py — Flask 交易明細 Dashboard

啟動:
    python app/trade_dashboard.py
    或
    make trade-dashboard

網址: http://localhost:5001

API:
    GET /           → HTML 頁面
    GET /trades     → trade_log.csv JSON
    GET /nav        → 資產曲線 + 期間績效 JSON
    GET /candles/<stock_id>/<entry_date>  → 日K OHLCV JSON
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from flask import Flask, jsonify, render_template

from app.db import get_session
from app.models import RawPrice

# ── 路徑設定 ───────────────────────────────────────────────────────────────
TRADE_LOG_CSV = ROOT / "artifacts" / "trade_log.csv"
BACKTEST_JSON = ROOT / "artifacts" / "backtest" / "test_v1a_with_sf.json"

app = Flask(__name__, template_folder=str(ROOT / "templates"))

# ── 記憶體快取（首次載入後固定，無需 reload） ──────────────────────────────
_trades_cache: list[dict] | None = None
_nav_cache: dict | None = None


def _load_trades() -> list[dict]:
    global _trades_cache
    if _trades_cache is None:
        df = pd.read_csv(TRADE_LOG_CSV)
        df = df.where(pd.notnull(df), None)  # NaN → None（JSON 友好）
        _trades_cache = df.to_dict(orient="records")
    return _trades_cache


def _load_nav() -> dict:
    global _nav_cache
    if _nav_cache is None:
        with open(BACKTEST_JSON, encoding="utf-8") as f:
            data = json.load(f)
        _nav_cache = {
            "equity_curve": data.get("equity_curve", []),
            "benchmark_curve": data.get("benchmark_curve", []),
            "periods": data.get("periods", []),
            "summary": data.get("summary", {}),
        }
    return _nav_cache


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("trade_dashboard.html")


@app.route("/trades")
def get_trades():
    if not TRADE_LOG_CSV.exists():
        return jsonify({
            "error": "trade_log.csv not found. "
                     "Run: python scripts/generate_trade_log.py --output artifacts/trade_log.csv"
        }), 404
    return jsonify(_load_trades())


@app.route("/nav")
def get_nav():
    if not BACKTEST_JSON.exists():
        return jsonify({"error": f"Backtest JSON not found: {BACKTEST_JSON}"}), 404
    return jsonify(_load_nav())


@app.route("/candles/<stock_id>/<entry_date>")
def get_candles(stock_id: str, entry_date: str):
    """查詢個股日K，範圍：進場前45天 ~ 出場後45天"""
    if not TRADE_LOG_CSV.exists():
        return jsonify({"error": "trade_log.csv not found"}), 404

    trades = _load_trades()
    matched = [
        t for t in trades
        if str(t.get("stock_id", "")) == stock_id and t.get("entry_date") == entry_date
    ]
    if not matched:
        return jsonify({"error": f"No trade: {stock_id} @ {entry_date}"}), 404

    trade = matched[0]
    exit_date_str = str(trade.get("exit_date") or entry_date)
    entry_price = float(trade.get("entry_price") or 0)
    exit_price = float(trade.get("exit_price") or 0)
    pnl = float(trade.get("realized_pnl_pct") or 0)
    name = str(trade.get("name") or "")

    try:
        entry_dt = datetime.strptime(entry_date, "%Y-%m-%d").date()
        exit_dt = datetime.strptime(exit_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format (expected YYYY-MM-DD)"}), 400

    start_dt = entry_dt - timedelta(days=45)
    end_dt = exit_dt + timedelta(days=45)

    with get_session() as session:
        rows = (
            session.query(RawPrice)
            .filter(
                RawPrice.stock_id == stock_id,
                RawPrice.trading_date >= start_dt,
                RawPrice.trading_date <= end_dt,
            )
            .order_by(RawPrice.trading_date)
            .all()
        )
        # 在 session 關閉前取出所有欄位值，避免 DetachedInstanceError
        candles = [
            {
                "time": str(r.trading_date),
                "open": float(r.open) if r.open is not None else None,
                "high": float(r.high) if r.high is not None else None,
                "low": float(r.low) if r.low is not None else None,
                "close": float(r.close) if r.close is not None else None,
                "volume": int(r.volume) if r.volume is not None else 0,
            }
            for r in rows
            if r.open is not None  # 過濾無效資料
        ]

    return jsonify({
        "candles": candles,
        "stock_id": stock_id,
        "name": name,
        "entry_date": entry_date,
        "exit_date": exit_date_str,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pnl_pct": pnl,
    })


# ── 啟動 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  台股 ML 交易明細 Dashboard")
    print(f"  trade_log : {TRADE_LOG_CSV}")
    print(f"  backtest  : {BACKTEST_JSON}")
    print("  URL       : http://localhost:5001")
    print("=" * 60)
    if not TRADE_LOG_CSV.exists():
        print(f"\n⚠️  警告：找不到 {TRADE_LOG_CSV}")
        print("   請先執行: python scripts/generate_trade_log.py --output artifacts/trade_log.csv\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
