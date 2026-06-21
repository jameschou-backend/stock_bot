#!/usr/bin/env python3
"""回補「曾存在但已下市」股票的歷史價格，修正 survivorship bias。

問題：raw_prices 以 fetch_stock_list()（今日 TaiwanStockInfo 現存清單）回補，2016–2021
已下市股票（日月光 2311、矽品 2325、樂陞 3662 等）完全缺失 → 回測 universe / label /
benchmark 全建立在「倖存者」上，使早年（2016–2021，高報酬年段）績效系統性高估。

解法：FinMind「按日期 bulk 全市場」抓取（data_id=None → 回該日所有交易股票，自然含
當時存在、後來下市者），upsert 進 raw_prices（冪等：現存股一併更新、缺失下市股新增）。

用法：
  python scripts/backfill_delisted_prices.py --diagnose
      抽樣每年年中一個交易日，抓全市場 stock_id 比對 DB，報告缺失下市股數量（不寫入）。
  python scripts/backfill_delisted_prices.py --start 2016-01-01 --end 2021-12-31
      回補指定區間全市場價格（建議分年跑；FinMind sponsor 6/24 到期前完成 2016–2021）。

注意：
  - 全市場 bulk 抓取資料量大，chunk-days 預設 7（單次回傳過大 FinMind 會空回，可調小）。
  - rate limit 走 config.finmind_requests_per_hour；冪等，中斷後重跑安全（重複 chunk 只重抓）。
  - 回補後須：(1) 全量重建 features/labels（FORCE_RECOMPUTE_DAYS=3650）
            (2) 重跑 10y 回測基準，更新 CLAUDE.md / memory 數字。
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.finmind import fetch_dataset
from app.models import RawPrice
from skills.ingest_prices import _normalize_prices


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fetch_market_day(config, d: date) -> set:
    """抓單一日期全市場 TaiwanStockPrice（data_id=None），回 stock_id 集合。"""
    df = fetch_dataset(
        "TaiwanStockPrice", d, d, data_id=None,
        token=config.finmind_token,
        requests_per_hour=config.finmind_requests_per_hour,
        max_retries=config.finmind_retry_max,
        backoff_seconds=config.finmind_retry_backoff,
    )
    if df.empty:
        return set()
    df = _normalize_prices(df)
    # 只看四碼普通股（[1-9] 開頭排除 00xx ETF；6 碼權證/ETN 不在此範圍），對齊生產 universe
    return {s for s in df["stock_id"].astype(str) if re.fullmatch(r"[1-9]\d{3}", s)}


def diagnose(config, probe_dates) -> None:
    print("=== Survivorship 缺口診斷（FinMind 全市場 vs DB raw_prices）===")
    total_missing = 0
    with get_session() as session:
        for d in probe_dates:
            market = _fetch_market_day(config, d)
            if not market:
                print(f"  {d}: FinMind 無資料（非交易日？）")
                continue
            rows = session.execute(
                select(RawPrice.stock_id).distinct().where(RawPrice.trading_date == d)
            ).all()
            in_db = {str(r[0]) for r in rows}
            missing = market - in_db
            total_missing += len(missing)
            extra = f"（例：{sorted(missing)[:8]}）" if missing else ""
            print(f"  {d}: FinMind {len(market)} 檔 / DB {len(in_db)} 檔 / 缺 {len(missing)} 檔{extra}")
    print(f"\n抽樣缺失合計 {total_missing} 檔（去重前）。缺口主要來自已下市股，"
          "回補後早年 universe / label / benchmark 才完整。")


def backfill(config, start: date, end: date, chunk_days: int = 1) -> None:
    # FinMind 全市場（data_id=None）多日區間會空回（資料量太大），必須「逐交易日」抓。
    # 交易日清單取自既有 raw_prices（現存股的交易日 = 全市場交易日，下市股也在這些日交易），
    # 順帶自動跳過非交易日，不浪費 API call。chunk_days 參數保留向後相容但已不使用。
    try:
        with get_session() as session:
            td_rows = session.execute(
                select(RawPrice.trading_date)
                .where(RawPrice.trading_date >= start)
                .where(RawPrice.trading_date <= end)
                .distinct()
            ).all()
    except Exception as exc:
        print(f"❌ 無法連線 DB（MySQL 127.0.0.1:3307 是否啟動？docker container poyue.mysql）：{exc}")
        sys.exit(1)
    trading_days = sorted({r[0] for r in td_rows})
    if not trading_days:
        with get_session() as session:
            _mn = session.execute(select(func.min(RawPrice.trading_date))).scalar()
            _mx = session.execute(select(func.max(RawPrice.trading_date))).scalar()
        print(f"❌ raw_prices 在 {start}~{end} 無交易日。實際資料範圍 {_mn}~{_mx}，請改用此範圍內的日期。")
        return
    print(f"回補全市場 {start} ~ {end}：逐交易日抓（{len(trading_days)} 個交易日，data_id=None 含下市股）...")
    total = 0
    skipped = 0
    with get_session() as session:
        for i, d in enumerate(trading_days, 1):
            try:
                df = fetch_dataset(
                    "TaiwanStockPrice", d, d, data_id=None,
                    token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,
                    max_retries=config.finmind_retry_max,
                    backoff_seconds=config.finmind_retry_backoff,
                    timeout=120,
                )
            except Exception as exc:
                print(f"  [{d}] fetch 失敗，跳過: {exc}")
                skipped += 1
                continue
            if df.empty:
                skipped += 1
                continue
            df = _normalize_prices(df)
            # 只回補四碼普通股（排除 ETF/權證/ETN），對齊生產 universe，避免灌入數萬權證
            df = df[df["stock_id"].str.fullmatch(r"[1-9]\d{3}")]
            records = df.to_dict("records")
            if not records:
                continue
            stmt = insert(RawPrice).values(records)
            stmt = stmt.on_duplicate_key_update(
                open=stmt.inserted.open, high=stmt.inserted.high,
                low=stmt.inserted.low, close=stmt.inserted.close,
                volume=stmt.inserted.volume,
            )
            session.execute(stmt)
            session.commit()
            total += len(records)
            if i % 50 == 0 or i == len(trading_days):
                print(f"  [{d}] {i}/{len(trading_days)} 累計 upsert {total:,} 列")
    print(f"\n完成：upsert {total:,} 列（{len(trading_days)} 交易日，{skipped} 日空回/失敗）。")
    print("下一步：(1) make rebuild-features 全量重建 features/labels")
    print("        (2) 重跑 10y 回測基準，更新 CLAUDE.md / memory 數字")


def main() -> None:
    p = argparse.ArgumentParser(description="回補下市股歷史價格（修正 survivorship bias）")
    p.add_argument("--start", type=_parse_date, default=date(2016, 1, 1))
    p.add_argument("--end", type=_parse_date, default=date.today())
    p.add_argument("--chunk-days", type=int, default=7)
    p.add_argument("--diagnose", action="store_true", help="只診斷缺口不回補")
    args = p.parse_args()

    config = load_config()
    if args.diagnose:
        probe = [date(y, 6, 1) for y in range(2016, 2026)]
        diagnose(config, probe)
    else:
        backfill(config, args.start, args.end, args.chunk_days)


if __name__ == "__main__":
    main()
