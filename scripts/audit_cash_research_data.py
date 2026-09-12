#!/usr/bin/env python3
"""Quantify historical cohort and price-version gaps; never certify missing evidence."""
import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read, write, sha

CACHE = ROOT / '.cache/cash-risk-data-audit-20260913'


def run(output, prepare=False):
    if output.exists():
        raise ValueError('Use a new report path; previous evidence is immutable')
    CACHE.mkdir(exist_ok=True)
    calls = []

    def get(name, dataset, start, end):
        path = CACHE / (name+'.parquet')
        if not path.exists():
            if not prepare:
                raise ValueError('Missing source; use explicit --prepare: '+name)
            frame = fetch_dataset(dataset, start, end, token=load_config().finmind_token,
                                  max_retries=0, timeout=30)
            if frame.empty:
                raise ValueError('Empty source cannot prove complete coverage: '+name)
            frame.to_parquet(path, index=False)
            calls.append(dict(dataset=dataset, start=str(start), end=str(end),
                              cache_hit=bool(frame.attrs.get('cache_hit', False))))
        return pd.read_parquet(path)

    refs = read(ROOT / '.cache/cash-allocation-inputs/manifest.json')['references']
    for ref in refs.values():
        if sha(ROOT / ref['path']) != ref['sha256']:
            raise ValueError('Historical research input changed: '+ref['path'])
    companies = pd.read_parquet(ROOT / refs['companies']['path'])
    cohort = set(companies.stock_id)
    calendar = pd.read_parquet(ROOT / refs['calendar']['path'])
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, 'date']))
    signals = read(ROOT / refs['signals']['path'])
    delisted = get('delistings', 'TaiwanStockDelisting', date(2001, 1, 1), date(2026, 9, 9))
    delisted = delisted[delisted.stock_id.str.fullmatch(r'[1-9]\d{3}') &
                        delisted.date.between('2022-01-03', '2026-09-09')]
    absent = delisted[~delisted.stock_id.isin(cohort)]
    info = get('stockinfo', 'TaiwanStockInfo', date(2001, 1, 1), date(2026, 9, 9))
    types = info.groupby('stock_id')['type'].agg(lambda x: sorted(set(x))).to_dict()
    probes = []
    # Four-digit, non-00 identifiers align with the strategy's stock convention;
    # a listing transfer/reused identifier is not automatically a lost company.
    for year in range(2022, 2027):
        day = days[days.year == year][0].date()
        frame = get('market-'+str(day), 'TaiwanStockPrice', day, day)
        if set(frame.date.astype(str)) != {str(day)}:
            raise ValueError('Market probe returned other dates')
        ids = set(frame.loc[frame.stock_id.str.fullmatch(r'[1-9]\d{3}'), 'stock_id'])
        missing = sorted(ids-cohort)
        classification = dict(twse_or_tpex_reference=[], emerging_only_reference=[],
                              mixed_market_reference=[], unknown_reference=[])
        for sid in missing:
            markets = set(types.get(sid, []))
            listed = bool(markets & {'twse', 'tpex'})
            key = ('mixed_market_reference' if listed and 'emerging' in markets else
                   'twse_or_tpex_reference' if listed else
                   'emerging_only_reference' if markets == {'emerging'} else 'unknown_reference')
            classification[key].append(sid)
        probes.append(dict(date=str(day), observed_ids=len(ids), absent_from_frozen_cohort=missing,
            absent_count=len(missing), classification=classification,
            absent_and_in_period_delisting=sorted(set(missing) & set(absent.stock_id)),
            proof='Observed trading symbols absent from cohort; current provider market labels are not historical market membership'))
    base = ROOT / '.cache/million-replay-signals'
    variants = [pd.read_parquet(base / name).set_index('date') for name in
                ('close-official.parquet', 'close-quality.parquet')]
    for frame in variants:
        frame.index = pd.to_datetime(frame.index)
    if not variants[0].index.equals(variants[1].index) or not variants[0].columns.equals(variants[1].columns):
        raise ValueError('Adjusted comparison matrices do not align')
    returns = [f/f.shift(20)-1 for f in variants]
    delta = (returns[0]-returns[1]).abs()
    selected_differences = []
    for entry in signals['entries']:
        sid, day = entry['members'][0], pd.Timestamp(entry['signal_date'])
        a, b = (f.at[day, sid] for f in returns)
        selected_differences.append(dict(stock_id=sid, signal_date=str(day.date()),
            official_return20=float(a) if np.isfinite(a) else None,
            quality_return20=float(b) if np.isfinite(b) else None,
            difference_pp=float(abs(a-b)*100) if np.isfinite(a) and np.isfinite(b) else None))
    differing = [r for r in selected_differences if r['difference_pp'] is None or r['difference_pp'] > .1]
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), live_qualified=False,
        strategy_inputs=dict(prices=True, volume=True, current_company_cohort=True,
                             revenue=False, financial_statements=False, news=False, rate_expectations=False),
        publication_status='Fundamental publication lag not applicable to this price-only strategy; historical price vintages unverified',
        historical_universe=dict(frozen_companies=len(cohort), probes=probes,
            delisting_records_in_period=len(delisted), absent_delisting_records=absent.to_dict('records'),
            complete=False, caveat='Delisting/transfer/name reuse needs classification; probes quantify missing symbols, not full historical membership'),
        adjusted_variants=dict(independent_sources_verified=False, compared_candidates=len(selected_differences),
            candidates_with_return20_difference_above_0_1pp_or_missing=len(differing),
            flagged_candidates=differing,
            matrix_cells_difference_above_0_1pp=int((delta > .001).sum().sum()),
            both_returns_available=int((returns[0].notna() & returns[1].notna()).sum().sum()),
            caveat='Stored variants share lineage; agreement is not independent corporate-action certification'),
        requests_this_run=calls, raw_sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in CACHE.glob('*.parquet')},
        input_sha256={ref['path']:ref['sha256'] for ref in refs.values()},
        code_sha256=sha(__file__),
        unresolved=['Reconstruct dated market membership including delistings and transfers',
                    'Backfill and reconcile missing members before rebuilding monthly groups',
                    'Validate adjustment events and historical publication/revision timestamps'])
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output, args.prepare)
    print([(x['date'], x['absent_count']) for x in result['historical_universe']['probes']])
