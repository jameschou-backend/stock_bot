#!/usr/bin/env python3
"""Read-only listing chronology audit; never reinterpret observation dates as IPO dates."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sqlalchemy import text
from app.db import get_session
from scripts.research_exit_scenarios import read, write, sha


def run(source, output):
    if output.exists():
        raise ValueError('Use a new evidence path')
    data = read(source)
    listings = data['historical_universe']['absent_delisting_records']
    rows = []
    with get_session() as session:
        for item in listings:
            sid, end = item['stock_id'], item['date']
            master = session.execute(text('SELECT name,market,listed_date,delisted_date,is_listed '
                'FROM stocks WHERE stock_id=:sid'), dict(sid=sid)).mappings().first()
            prices = session.execute(text('SELECT COUNT(*) AS rows_count, MIN(trading_date) AS first_date, '
                'MAX(trading_date) AS last_date FROM raw_prices WHERE stock_id=:sid AND trading_date>=:start '
                'AND trading_date<:end'), dict(sid=sid, start='2021-01-01', end=end)).mappings().one()
            after = session.execute(text('SELECT trading_date,close,volume FROM raw_prices '
                'WHERE stock_id=:sid AND trading_date>=:end ORDER BY trading_date LIMIT 10'),
                dict(sid=sid, end=end)).mappings().all()
            after_count = session.execute(text('SELECT COUNT(*) FROM raw_prices WHERE stock_id=:sid '
                'AND trading_date>=:end'), dict(sid=sid, end=end)).scalar_one()
            history = session.execute(text('SELECT effective_date,status_type FROM stock_status_history '
                'WHERE stock_id=:sid ORDER BY effective_date'), dict(sid=sid)).mappings().all()
            def serial(row):
                return {k: v if v is None or isinstance(v, (str, int, bool, float)) else str(v)
                        for k, v in dict(row).items()}
            rows.append(dict(stock_id=sid, delisting_reference=end,
                master=serial(master) if master else None, prices_before_delisting=serial(prices),
                prices_on_or_after_delisting_count=after_count,
                prices_on_or_after_delisting_examples=[serial(x) for x in after],
                recorded_status_history=[serial(x) for x in history],
                eligible_for_automatic_historical_cohort=False,
                reason='Need verified listing intervals, transfers and code reuse before adding to historical universe'))
    verification = ROOT / '.cache/cash-risk-data-audit-20260913/2456-20260617-vendor-check.json'
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), rows=rows,
        data_source_sha256=sha(source), code_sha256=sha(__file__),
        companies_with_pre_delisting_prices=sum(x['prices_before_delisting']['rows_count'] > 0 for x in rows),
        companies_missing_listing_date=sum(not x['master'] or x['master']['listed_date'] is None for x in rows),
        companies_with_post_delisting_rows=sum(x['prices_on_or_after_delisting_count'] > 0 for x in rows),
        live_qualified=False, database_mutations=0,
        conclusion='Do not append current stock metadata to historical cohorts: missing IPO dates and chronology conflicts remain',
        vendor_spot_check=read(verification) if verification.exists() else None,
        vendor_spot_check_sha256=sha(verification) if verification.exists() else None)
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r = run(args.source, args.output)
    print({k: r[k] for k in ('companies_with_pre_delisting_prices', 'companies_missing_listing_date',
                             'companies_with_post_delisting_rows', 'database_mutations')})
