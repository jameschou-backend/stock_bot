#!/usr/bin/env python3
"""Read-only evidence and research quarantine; absence alone is not a delisting."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from app.db import get_session
from scripts.research_exit_scenarios import read, write, sha

FIELDS = ['open', 'high', 'low', 'close', 'volume']
PAIRS = [('2026-06-09', '2017-09-18'), ('2026-06-11', '2017-12-18'), ('2026-06-17', '2018-05-18')]
CACHE = ROOT / '.cache/price-chronology-20260913'


def normalize_source(frame, day):
    if frame.empty or 'date' not in frame or set(frame.date.astype(str)) != {day}:
        raise ValueError('Missing or wrong source date: '+day)
    frame = frame.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}).copy()
    if not {'stock_id', *FIELDS}.issubset(frame):
        raise ValueError('Missing price fields')
    frame.stock_id = frame.stock_id.astype(str)
    frame = frame[frame.stock_id.str.fullmatch(r'\d{4}')]
    if frame.empty or frame.stock_id.duplicated().any():
        raise ValueError('Empty or duplicate stock snapshot')
    for field in FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors='raise')
    return frame.set_index('stock_id')[FIELDS]


def compare_snapshot(current, source, historical):
    common = current.index.intersection(source.index)
    a, b = current.loc[common, FIELDS].astype(float), source.loc[common, FIELDS].astype(float)
    equal = np.isclose(a, b, rtol=0, atol=.000001, equal_nan=True).all(axis=1)
    missing = sorted(set(current.index)-set(source.index))
    conflicts = sorted(set(common[~equal]) | set(missing))
    copies = []
    for sid in conflicts:
        if sid in historical.index:
            x, y = (f.loc[sid, FIELDS].astype(float).to_numpy() for f in (current, historical))
            # Zero/missing volume fingerprints are insufficient evidence of a copied quote.
            if np.isfinite(x).all() and x[-1] > 0 and np.array_equal(x, y):
                copies.append(sid)
    return dict(common_rows=len(common), different_rows=common[~equal].tolist(),
                absent_rows=missing, conflicts=conflicts, exact_older_ohlcv_matches=copies)


def run(output):
    if output.exists():
        raise ValueError('Choose a new audit output')
    refs = read(ROOT / '.cache/cash-allocation-inputs/manifest.json')['references']
    from scripts import research_cash_allocation as cash
    data = cash.load_inputs()
    pool = set(data.quotes.stock_id)
    rows, quarantine, inputs = [], [], {}
    for day, old_day in PAIRS:
        vendor_path = CACHE / (day+'.parquet')
        source = normalize_source(pd.read_parquet(vendor_path), day)
        inputs[str(vendor_path.relative_to(ROOT))] = sha(vendor_path)
        frames = []
        for label, date in [('db', day), ('old', old_day)]:
            snapshot = CACHE / f'{label}-{day}.parquet'
            if not snapshot.exists():
                with get_session() as session:
                    result = session.execute(text('SELECT stock_id,open,high,low,close,volume '
                        'FROM raw_prices WHERE trading_date=:d'), dict(d=date)).mappings().all()
                frame = pd.DataFrame([dict(x) for x in result])
                for field in FIELDS:
                    frame[field] = frame[field].astype(float)
                frame.to_parquet(snapshot, index=False)
            inputs[str(snapshot.relative_to(ROOT))] = sha(snapshot)
            frames.append(pd.read_parquet(snapshot).set_index('stock_id'))
        result = compare_snapshot(frames[0], source, frames[1])
        selected = sorted(set(result['conflicts']) & pool)
        frozen = data.quotes[data.quotes.date.eq(pd.Timestamp(day))].set_index('stock_id')
        common = frozen.index.intersection(source.index)
        frozen_differences = common[~np.isclose(frozen.loc[common, FIELDS].astype(float),
            source.loc[common, FIELDS], rtol=0, atol=.000001, equal_nan=True).all(axis=1)].tolist()
        if set(frozen_differences)-set(result['conflicts']):
            raise ValueError('Frozen quote has an additional conflict; extend explicit audit first')
        result.update(date=day, older_date=old_day, db_rows=len(frames[0]), vendor_rows=len(source),
                      frozen_pool_conflicts=selected, frozen_common_differences=frozen_differences)
        rows.append(result)
        for sid in result['conflicts']:
            quarantine.append(dict(stock_id=sid, date=day, in_replay_pool=sid in pool,
                exact_older_copy=sid in result['exact_older_ohlcv_matches'],
                original={k:float(v) for k,v in frames[0].loc[sid,FIELDS].items()},
                reason='Missing from fresh daily source' if sid in result['absent_rows'] else 'OHLCV revision conflict'))
    paths = {ref['path']: ref['sha256'] for ref in refs.values()}
    for name, digest in {**inputs, **paths}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Input changed: '+name)
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), rows=rows, quarantine=quarantine,
        input_sha256={**inputs, **paths}, code_sha256=sha(__file__), database_mutations=0,
        live_qualified=False, root_cause='Older-day OHLCV copies found; responsible historical writer not yet established',
        scope='Three suspect dates only; not a full-history certification')
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r = run(args.output)
    print([{k: row[k] for k in ('date','db_rows','vendor_rows','frozen_pool_conflicts')}
           | {'conflicts':len(row['conflicts']), 'older_copies':len(row['exact_older_ohlcv_matches'])}
           for row in r['rows']])
