#!/usr/bin/env python3
"""Recover local quotes truncated at a later market-transfer listing date."""
from datetime import date, datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from app.config import load_config
from app.db import get_session
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read, write, sha
from scripts.prepare_historical_cohort_supplement import summarize, FIELDS
from scripts.prepare_historical_selector_quality import validate

OUTPUT = ROOT / '.cache/historical-listing-prefix-v2-20260925'
IDENTITY = ROOT / '.cache/historical-universe-followup-v2-20260925/report-final.json'
COMPANIES = ROOT / '.cache/million-replay-signals/companies.parquet'


def plan():
    report, companies = read(IDENTITY), pd.read_parquet(COMPANIES)
    rows, unconfirmed = [], []
    for company in companies.itertuples():
        episodes = [e for e in report['episodes'] if e['stock_id'] == company.stock_id and e['category'] == '股票']
        if not episodes:
            continue
        first = min(e['start'] for e in episodes)
        old = str(company.listed_date.date())
        if first < old and old > '2021-01-01':
            prior = [e for e in episodes if e['end'] == old and e['market'] == 'TPEx'
                     and e['start_evidence'] in ('reviewed_primary_archive', 'reviewed_primary_archive_v2')]
            if len(prior) != 1:
                unconfirmed.append(dict(stock_id=company.stock_id, old_start=old, snapshot_start=first))
                continue
            rows.append(dict(stock_id=company.stock_id, start=max(prior[0]['start'], '2021-01-01'), end_exclusive=old))
    return dict(rows=rows, unconfirmed_discrepancies=unconfirmed, maximum_calls=20, max_retries=0,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in (IDENTITY, COMPANIES)},
        raw_source='local_mysql_read_only', adjusted_source='FinMind TaiwanStockPriceAdj')


def verify():
    result = read(OUTPUT / 'manifest.json')
    if (sha(OUTPUT / 'manifest.json') != (OUTPUT / 'manifest.sha256').read_text().strip()
            or result['plan'] != plan()):
        raise ValueError('Historical listing-prefix identity changed')
    for name, digest in result['files_sha256'].items():
        if sha(OUTPUT / name) != digest:
            raise ValueError('Historical listing-prefix source changed: ' + name)
    return result


def prepare():
    if (OUTPUT / 'manifest.json').exists():
        return verify()
    fixed = plan()
    if (OUTPUT / 'plan.json').exists() and read(OUTPUT / 'plan.json') != fixed:
        raise ValueError('Prefix plan changed')
    write(OUTPUT / 'plan.json', fixed)
    raw_path = OUTPUT / 'quotes.parquet'
    if not raw_path.exists():
        frames = []
        with get_session() as session:
            if session.execute(text('SELECT DATABASE()')).scalar_one() != 'stock_bot':
                raise ValueError('Unexpected source database')
            for row in fixed['rows']:
                records = session.execute(text('SELECT stock_id,trading_date AS date,open,high,low,close,volume '
                    'FROM raw_prices WHERE stock_id=:stock_id AND trading_date>=:start AND trading_date<:end_exclusive '
                    'ORDER BY trading_date'), row).mappings().all()
                frames.append(pd.DataFrame(records))
        frame = pd.concat(frames, ignore_index=True)
        if frame.empty:
            raise ValueError('Historical prefixes are unavailable; no synthetic data is allowed')
        frame['date'] = pd.to_datetime(frame.date)
        for col in ('open', 'high', 'low', 'close', 'volume'):
            frame[col] = pd.to_numeric(frame[col], errors='raise')
        frame[FIELDS].to_parquet(raw_path, index=False)
    raw = pd.read_parquet(raw_path)
    info = summarize(raw, [r['stock_id'] for r in fixed['rows']], allow_quarantine=True)
    attempts_path = OUTPUT / 'attempts.json'
    attempts = read(attempts_path) if attempts_path.exists() else []
    for row in fixed['rows']:
        sid = row['stock_id']
        target = OUTPUT / (sid + '.parquet')
        if target.exists():
            validate(pd.read_parquet(target), sid)
            continue
        if len(attempts) >= fixed['maximum_calls']:
            raise ValueError('Listing prefix request ceiling reached')
        attempts.append(dict(stock_id=sid, at=datetime.now(timezone.utc).isoformat()))
        write(attempts_path, attempts)
        frame = fetch_dataset('TaiwanStockPriceAdj', date(2021, 1, 1), date(2026, 9, 9), data_id=sid,
            token=load_config().finmind_token, requests_per_hour=5400, max_retries=0, timeout=30)
        validate(frame, sid)
        frame.to_parquet(target, index=False)
        print(sid, 'adjusted', len(frame), flush=True)
    result = dict(plan=fixed, summary=info, calls_reserved=len(attempts), database_writes=0,
        observed_at=datetime.now(timezone.utc).isoformat(), source_is_revised_snapshot=True,
        files_sha256={p.name: sha(p) for p in OUTPUT.iterdir() if p.suffix in ('.json', '.parquet')})
    write(OUTPUT / 'manifest.json', result)
    (OUTPUT / 'manifest.sha256').write_text(sha(OUTPUT / 'manifest.json') + '\n')
    return verify()


if __name__ == '__main__':
    with file_lock(OUTPUT / '.prepare.lock', timeout=0):
        result = prepare()
    print('raw_rows', result['summary']['quote_rows'], 'quarantined', result['summary']['quarantined_rows'],
          'calls_reserved', result['calls_reserved'])
