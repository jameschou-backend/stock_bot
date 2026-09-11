#!/usr/bin/env python3
"""Reconcile cached raw evidence into an isolated validated table. Zero requests."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy.dialects.mysql import insert
from app.db import get_session, run_migrations
from app.finmind import FinMindError
from app.models import ValidatedHoldingDist, Stock
from skills.holding_validation import aggregate


def run(apply=False):
    started = time.monotonic()
    paths = sorted((ROOT/'.cache/chip-inputs/raw/TaiwanStockHoldingSharesPer').glob('*.parquet'))
    extra = ROOT/'.cache/holder-case-2492/TaiwanStockHoldingSharesPer.parquet'
    if extra.exists():
        paths.append(extra)
    if not paths:
        raise ValueError('No raw holder cache. Prepare raw evidence first.')
    with get_session() as session:
        allowed = {x[0] for x in session.query(Stock.stock_id).filter(Stock.security_type == 'stock')}
    rows, rejected, fingerprints = {}, [], {}
    for path in paths:
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = pd.read_parquet(path)
        for (sid, day), group in raw.groupby(['stock_id', 'date']):
            try:
                frame = aggregate(group, allowed)
            except (FinMindError, ValueError, TypeError) as exc:
                rejected.append(dict(stock_id=sid, date=str(day), reason=str(exc)))
                continue
            for row in frame.to_dict('records'):
                key = (row['stock_id'], row['trading_date'])
                if key in rows and rows[key] != row:
                    raise ValueError(f'Conflicting source revisions for {key}')
                rows[key] = row
    if apply:
        run_migrations(ROOT/'storage/migrations/017_validated_holding.sql')
        with get_session() as session:
            values = list(rows.values())
            for i in range(0, len(values), 1000):
                stmt = insert(ValidatedHoldingDist).values(values[i:i+1000])
                stmt = stmt.on_duplicate_key_update({c: stmt.inserted[c] for c in
                    ('available_date', 'large_holder_pct', 'small_holder_pct', 'top_level_pct', 'holder_count')})
                session.execute(stmt)
            session.commit()
    report = dict(applied=apply, requests=0, rows=len(rows), stocks=len({k[0] for k in rows}),
                  rejected=rejected, sources=fingerprints, elapsed_seconds=round(time.monotonic()-started, 2),
                  legacy_table_untouched=True, full_market_coverage=False)
    dest = ROOT/'.cache/holding-repair.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','rejected')}|{'rejected':len(rejected)}))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    run(parser.parse_args().apply)
