#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List, Set

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.finmind import fetch_dataset_by_stocks, fetch_stock_list
from app.market_calendar import get_recent_trading_days
from app.models import RawInstitutional, Stock
from scripts.data_quality_report import generate_report, write_outputs
from skills.ingest_institutional import _normalize_institutional


def _chunks(days: List[date], size: int) -> List[List[date]]:
    return [days[i:i + size] for i in range(0, len(days), size)]


def _allowed_stock_ids(session) -> Set[str]:
    rows = session.execute(
        select(Stock.stock_id).where(Stock.is_listed == True).where(Stock.security_type == "stock")
    ).fetchall()
    return {str(row[0]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw_institutional using trading calendar")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    config = load_config()
    with get_session() as session:
        asof = date.today()
        trading_days = list(reversed(get_recent_trading_days(session, asof, args.days)))
        if not trading_days:
            print("no trading days to backfill")
            return
        stock_ids = fetch_stock_list(
            config.finmind_token,
            requests_per_hour=config.finmind_requests_per_hour,
            max_retries=config.finmind_retry_max,
            backoff_seconds=config.finmind_retry_backoff,
        )
        allowed = _allowed_stock_ids(session)
        total_rows = 0
        for idx, chunk_days in enumerate(_chunks(trading_days, 20), start=1):
            start_date = min(chunk_days)
            end_date = max(chunk_days)
            print(f"[institutional] chunk {idx}: {start_date} ~ {end_date}")
            last_exc = None
            for attempt in range(1, args.retries + 1):
                try:
                    df = fetch_dataset_by_stocks(
                        "TaiwanStockInstitutionalInvestorsBuySell",
                        start_date,
                        end_date,
                        stock_ids,
                        token=config.finmind_token,
                        batch_size=500,
                        use_batch_query=True,
                        requests_per_hour=config.finmind_requests_per_hour,
                        max_retries=config.finmind_retry_max,
                        backoff_seconds=config.finmind_retry_backoff,
                    )
                    if df.empty:
                        break
                    df = _normalize_institutional(df, allowed_stock_ids=allowed or None)
                    records = df.to_dict("records")
                    if not records:
                        break
                    stmt = insert(RawInstitutional).values(records)
                    stmt = stmt.on_duplicate_key_update(
                        foreign_buy=stmt.inserted.foreign_buy,
                        foreign_sell=stmt.inserted.foreign_sell,
                        foreign_net=stmt.inserted.foreign_net,
                        trust_buy=stmt.inserted.trust_buy,
                        trust_sell=stmt.inserted.trust_sell,
                        trust_net=stmt.inserted.trust_net,
                        dealer_buy=stmt.inserted.dealer_buy,
                        dealer_sell=stmt.inserted.dealer_sell,
                        dealer_net=stmt.inserted.dealer_net,
                    )
                    session.execute(stmt)
                    session.commit()
                    total_rows += len(records)
                    break
                except Exception as exc:
                    session.rollback()
                    last_exc = exc
                    print(f"  attempt {attempt}/{args.retries} failed: {exc}")
            if last_exc is not None and attempt == args.retries:
                raise last_exc

    payload = generate_report(days=args.days, asof=date.today())
    json_path, md_path = write_outputs(payload)
    print(f"institutional backfill rows={total_rows}")
    print(f"quality report: {json_path}")
    print(f"quality report: {md_path}")


if __name__ == "__main__":
    main()
