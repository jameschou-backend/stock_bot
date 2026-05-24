#!/usr/bin/env python
"""Stage J：Daily alert via Telegram。

通知內容：
  1. 月初 picks（每月第 1 個交易日，含 30 檔清單）
  2. 月底前 5 天提醒 rebalance（提早準備換股）
  3. 大盤 regime 切換（200ma cross）
  4. MDD rolling 警告（達 -20% / -30%）

設定方式（.env 加 2 個 env var）：
  TELEGRAM_BOT_TOKEN=xxx  （從 @BotFather 取得）
  TELEGRAM_CHAT_ID=yyy    （從 @userinfobot 取得）

用法：
  python scripts/daily_alert.py             # 跑所有 check + send
  python scripts/daily_alert.py --dry-run   # 不真的 send，只 print
  python scripts/daily_alert.py --test      # 跑 test 訊息
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def send_telegram(text_msg: str, dry_run: bool = False) -> bool:
    """送 Telegram message。dry_run=True 時只 print。"""
    if dry_run:
        print("─" * 60)
        print(text_msg)
        print("─" * 60)
        return True
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID 未設定")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text_msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.error(f"Telegram send failed: {exc}")
        return False


def check_today_is_rebalance_day(engine) -> tuple[bool, int]:
    """檢查今天是否為「月初第 1 個交易日」(rebalance day)。

    Returns:
        (is_rebalance, days_to_next_rebalance)
    """
    today = date.today()
    # 取本月所有 trading_date
    q = text("""
        SELECT DISTINCT trading_date FROM raw_prices
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL 60 DAY)
        ORDER BY trading_date
    """)
    df = pd.read_sql(q, engine)
    if df.empty:
        return False, -1
    tds = pd.to_datetime(df["trading_date"]).dt.date.tolist()
    tds_set = set(tds)

    # 是否本月第一個 trading_date
    first_of_month = [d for d in tds if d.month == today.month and d.year == today.year]
    is_rebalance = bool(first_of_month) and today == first_of_month[0]

    # 距下次 rebalance（下月第 1 個 trading_date）天數
    next_month = today.replace(day=28) + timedelta(days=4)
    next_month = next_month.replace(day=1)
    next_first = [d for d in tds if d.year == next_month.year and d.month == next_month.month]
    days_to_next = (next_first[0] - today).days if next_first else -1

    return is_rebalance, days_to_next


def fetch_latest_picks(engine) -> pd.DataFrame:
    q = text("""
        SELECT p.stock_id, p.score, p.pick_date, s.name, s.industry_category
        FROM picks p
        LEFT JOIN stocks s ON s.stock_id = p.stock_id
        WHERE p.pick_date = (SELECT MAX(pick_date) FROM picks)
        ORDER BY p.score DESC
    """)
    return pd.read_sql(q, engine)


def check_regime_change(engine) -> Optional[str]:
    """檢查近 5 日大盤 regime 是否切換（200ma cross）。"""
    q = text("""
        SELECT trading_date, AVG(close) AS mkt_close
        FROM raw_prices
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL 250 DAY)
          AND stock_id REGEXP '^[0-9]{4}$'
        GROUP BY trading_date
        ORDER BY trading_date
    """)
    df = pd.read_sql(q, engine)
    if df.empty or len(df) < 200:
        return None
    s = df["mkt_close"].astype(float)
    ma200 = s.rolling(200, min_periods=40).mean()
    above = s > ma200
    # 檢查近 5 日有沒有 cross
    recent = above.iloc[-5:]
    if recent.nunique() > 1:
        if recent.iloc[-1]:
            return "🟢 大盤剛站上 200ma（轉 BULL）"
        return "🔴 大盤剛跌破 200ma（轉 BEAR）"
    return None


def check_mdd_warning(engine, threshold_pct: float = -0.20) -> Optional[str]:
    """檢查 portfolio 近期 MDD 是否觸警告線。"""
    # 用 picks 表的歷史結果近似
    q = text("""
        SELECT pick_date, COUNT(*) AS n
        FROM picks
        WHERE pick_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
        GROUP BY pick_date
        ORDER BY pick_date
    """)
    # 暫時 stub：未來接 strategy_c_trades 算實際 P&L
    return None


def format_picks_message(picks_df: pd.DataFrame, capital: int = 1_000_000) -> str:
    """格式化 picks 為 Telegram 訊息。"""
    if picks_df.empty:
        return "⚠️ 今日無 picks"
    n = len(picks_df)
    per_stock = capital / n

    msg = f"📈 *月初 Picks ({picks_df.iloc[0]['pick_date']})*\n\n"
    msg += f"資金 ${capital:,} → {n} 檔等權，每檔 ${per_stock:,.0f}\n\n"
    msg += "```\n#  代碼  名稱             產業\n"
    for i, (_, p) in enumerate(picks_df.head(30).iterrows(), 1):
        name = (p.get("name") or "")[:8]
        ind = (p.get("industry_category") or "")[:8]
        msg += f"{i:>2} {p['stock_id']:<5} {name:<8} {ind}\n"
    msg += "```\n\n"
    msg += "🔔 *操作提醒*：當日收盤前 5 分鐘等權買入。"
    return msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="不真的 send，只 print")
    p.add_argument("--test", action="store_true", help="送測試訊息確認 token")
    p.add_argument("--capital", type=int, default=1_000_000)
    args = p.parse_args()

    if args.test:
        send_telegram(
            "🤖 *Stock Bot 測試訊息*\n\n如果你看到這條，代表 Telegram bot 設定成功！",
            dry_run=args.dry_run,
        )
        return

    engine = get_engine()
    alerts = []

    # 1. Rebalance day check
    is_reb, days_to_next = check_today_is_rebalance_day(engine)
    if is_reb:
        picks = fetch_latest_picks(engine)
        alerts.append(format_picks_message(picks, args.capital))
    elif 0 < days_to_next <= 5:
        alerts.append(
            f"⏰ *Rebalance 提醒*\n\n"
            f"距下次月初再平衡日剩 *{days_to_next} 天*。\n"
            f"請預先準備資金 + 檢查目前持倉。"
        )

    # 2. Regime change check
    regime_msg = check_regime_change(engine)
    if regime_msg:
        alerts.append(f"📊 *市場狀態警示*\n\n{regime_msg}")

    # 3. MDD warning（stub）
    mdd_msg = check_mdd_warning(engine)
    if mdd_msg:
        alerts.append(f"⚠️ *MDD 警示*\n\n{mdd_msg}")

    if not alerts:
        logger.info("無 alert 觸發")
        if args.dry_run:
            print("無 alert 觸發")
        return

    for a in alerts:
        send_telegram(a, dry_run=args.dry_run)
    logger.info(f"送出 {len(alerts)} 則 alert")


if __name__ == "__main__":
    main()
