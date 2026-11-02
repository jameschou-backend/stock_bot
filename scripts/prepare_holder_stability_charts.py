#!/usr/bin/env python3
"""Append-only, offline display data derived from the existing holder study."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.holder_stability_web import read_report, study_view, EXPECTED_SHA256
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _prepare
from skills.verified_backtest_tool import offline_only


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build(output):
    if output.exists() or output.with_suffix('.sha256').exists():
        raise ValueError('Choose new output paths; archived chart data is immutable')
    report = read_report()
    members = {row['stock_id']: row for row in study_view()['candidates']}
    bundle = ROOT / '.cache/scanner-20261030-close/inputs-v1'
    weekly_path = ROOT / '.cache/holder-stability-600-20261102/run-a/weekly.parquet'
    refs = {k: v for k, v in report['source_sha256'].items() if k.startswith(str(bundle.relative_to(ROOT)) + '/')}
    refs[str(weekly_path.relative_to(ROOT))] = report['research_outputs'][str(weekly_path.relative_to(ROOT))]
    refs[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))

    def verify():
        for path, expected in refs.items():
            if digest(ROOT / path) != expected:
                raise ValueError('Frozen chart input changed: ' + path)

    with offline_only():
        verify()
        days = pd.DatetimeIndex(pd.read_parquet(bundle / 'close-official.parquet', columns=['date']).date)
        days = days[days <= pd.Timestamp(report['end'])][-240:]
        data = load_bundle(bundle, str(days[0].date()), report['end'])
        bars = data['bars'][data['bars'].stock_id.isin(members)]
        frames, _, ids = _prepare(bars, data['calendar'], pd.Timestamp(report['end']))
        weekly = pd.read_parquet(weekly_path)
        weekly = weekly[weekly.stock_id.isin(members) & weekly.date.ge(days[0] - pd.Timedelta(days=60))
                        & weekly.date.le(pd.Timestamp(report['end']))].sort_values(['stock_id', 'date'])
        stocks = {}
        for sid in sorted(members):
            candles = []
            for day in days:
                valid = bool(frames['valid'].at[day, sid] and frames['eligible'].at[day, sid] == True)
                values = dict(date=str(day.date()), valid=valid, open=None, high=None, low=None,
                              close=None, volume=None)
                if valid:
                    factor = frames['c'].at[day, sid] / frames['close'].at[day, sid]
                    values.update(open=float(frames['open'].at[day, sid] * factor),
                                  high=float(frames['h'].at[day, sid]), low=float(frames['l'].at[day, sid]),
                                  close=float(frames['c'].at[day, sid]), volume=float(frames['volume'].at[day, sid]))
                candles.append(values)
            observations = []
            for row in weekly[weekly.stock_id.eq(sid)].to_dict('records'):
                observations.append(dict(observed_date=str(pd.Timestamp(row['date']).date()),
                    valid=bool(row['valid']), reason=row['reason'],
                    ratio=float(row['large_pct']) if row['valid'] else None,
                    delta4=float(row['large_pct_delta4']) if row['change_known'] else None))
            stocks[sid] = dict(stock_id=sid, name=members[sid]['name'], candles=candles, observations=observations)
        verify()
        result = dict(schema='holder_stability_charts_v1', study_sha256=EXPECTED_SHA256,
            start=str(days[0].date()), end=report['end'], price_basis='adjusted', volume_unit='shares',
            source_sha256=refs, stocks=stocks, network_requests=0, database_writes=0,
            live_qualified=False, adopted=False)
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as target:
            target.write(gzip.compress(payload, mtime=0))
        output.with_suffix('.sha256').write_text(digest(output) + '\n')
    print(json.dumps(dict(stocks=len(stocks), sessions=len(days), sha256=digest(output), bytes=output.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    build(parser.parse_args().output)
