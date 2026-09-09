#!/usr/bin/env python
"""Fixed revenue contrasts with frozen inputs and bounded source reconciliation."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.file_lock import file_lock
from scripts.research_flow import load_inputs, verify_inputs as verify_prices, atomic_json, save_frame
from scripts.research_rules import digest
from skills.flow_research import flow_scores, portfolio_diagnostics, rolling_comparison
from skills.revenue_research import NAMES, revenue_features, revenue_scores
from skills.rule_research import executable, simulate, metrics

CACHE = ROOT / '.cache/revenue-research'
PREREG = ROOT / 'docs/prereg_revenue_20260909.md'
OFFICIAL = ROOT/'docs/revenue_official_samples_20260909.json'
CODE = ('scripts/research_revenue.py', 'scripts/research_flow.py', 'skills/revenue_research.py',
        'skills/flow_research.py', 'skills/rule_research.py')


def verify_inputs():
    manifest = json.loads((CACHE/'inputs.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(CACHE/name) != expected:
            raise ValueError('Revenue input changed: '+name)
    if manifest['audit']['passed'] is not True:
        raise ValueError('Revenue source audit did not pass')
    return manifest


def prepare_inputs():
    from sqlalchemy import select
    from app.config import load_config
    from app.db import get_session
    from app.finmind import fetch_dataset
    from app.models import RawFundamental

    verify_prices()
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE/'inputs.json').exists():
        return verify_inputs()
    path = CACHE/'revenue.parquet'
    if not path.exists():
        with get_session() as session:
            rows = pd.read_sql(select(RawFundamental.stock_id, RawFundamental.trading_date,
                                     RawFundamental.revenue_current_month), session.get_bind())
        if rows.empty or rows.duplicated(['stock_id', 'trading_date']).any():
            raise ValueError('Empty/duplicate local revenue')
        save_frame(path, rows)
    rows = pd.read_parquet(path)
    rows.trading_date = pd.to_datetime(rows.trading_date)
    config = load_config()
    checks, files = [], [path]
    for sid in ('2330', '2327', '3105'):
        path = CACHE/f'finmind-{sid}.parquet'
        meta = path.with_suffix('.json')
        if not path.exists() or not meta.exists():
            data = fetch_dataset('TaiwanStockMonthRevenue', date(2016, 3, 1), date(2026, 6, 1),
                                 token=config.finmind_token, data_id=sid, max_retries=0, timeout=30,
                                 requests_per_hour=config.finmind_requests_per_hour)
            if data.empty:
                raise ValueError('Empty FinMind sample '+sid)
            save_frame(path, data)
            atomic_json(meta, {**data.attrs, 'rows': len(data), 'sha256': digest(path)})
        if json.loads(meta.read_text())['sha256'] != digest(path):
            raise ValueError('FinMind sample changed '+sid)
        data = pd.read_parquet(path)
        data['trading_date'] = pd.to_datetime(data['date'])
        period = pd.to_datetime(dict(year=data.revenue_year, month=data.revenue_month, day=1))
        mapping_ok = (period + pd.offsets.MonthBegin(1)).eq(data.trading_date).all()
        local = rows[rows.stock_id.eq(sid) & rows.trading_date.between('2016-03-01', '2026-06-01')]
        joined = local.merge(data[['stock_id', 'trading_date', 'revenue']],
                             on=['stock_id', 'trading_date'], how='outer', validate='one_to_one', indicator=True)
        good = joined['_merge'].eq('both') & joined.revenue_current_month.eq(joined.revenue)
        checks.append({'stock_id': sid, 'rows': len(joined), 'matched': int(good.sum()),
                       'period_mapping_ok': bool(mapping_ok)})
        files.extend([path, meta])
    # Explicit, reviewed facts from company SEC filings, not a silent scraper fallback.
    official_path = CACHE/'official-samples.json'
    if not official_path.exists():
        atomic_json(official_path, json.loads(OFFICIAL.read_text()))
    official = json.loads(official_path.read_text())
    samples = pd.DataFrame(official['samples'])
    tsmc = rows[rows.stock_id.eq('2330')].set_index('trading_date').revenue_current_month
    dates = pd.to_datetime(samples.revenue_month) + pd.offsets.MonthBegin(1)
    values = tsmc.reindex(dates).to_numpy(float)
    differences = np.abs(values-samples.revenue_ntd.to_numpy(float))
    official_check = {'sources': official['sources'], 'months': len(samples), 'unit': 'NTD',
                      'maximum_difference': float(differences.max()),
                      'passed': bool(np.isfinite(differences).all() and (differences == 0).all())}
    passed = all(c['rows'] == c['matched'] and c['period_mapping_ok'] for c in checks) and official_check['passed']
    audit = {'passed': passed, 'finmind_db_checks': checks, 'official_check': official_check,
             'negative_revenue_rows_excluded': int(rows.revenue_current_month.lt(0).sum())}
    atomic_json(CACHE/'audit.json', audit)
    if not passed:
        raise ValueError('Source reconciliation failed; inspect .cache/revenue-research/audit.json')
    files.extend([official_path, CACHE/'audit.json'])
    manifest = {'schema': 1, 'prepared_at': datetime.now(timezone.utc).isoformat(),
                'rows': len(rows), 'provider_date_min': str(rows.trading_date.min().date()),
                'provider_date_max': str(rows.trading_date.max().date()), 'audit': audit,
                'sample_requests': 3, 'files_sha256': {p.name: digest(p) for p in files}}
    atomic_json(CACHE/'inputs.json', manifest)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return manifest


def run(output):
    started = time.perf_counter()
    manifest = verify_inputs()
    fields, net, companies, source = load_inputs()
    close = fields['adj_close']
    rows = pd.read_parquet(CACHE/'revenue.parquet')
    flags = executable(*(fields[n] for n in ('raw_close', 'raw_volume', 'raw_high', 'raw_low')))
    prices = flow_scores(close, *(fields[n] for n in ('raw_close', 'raw_volume', 'raw_high', 'raw_low')), net)
    loaded = time.perf_counter()
    results, coverage = [], []
    segments = {'2018_2022': ('2018-01-01', '2022-12-31'),
                '2023_2025': ('2023-01-01', '2025-12-31'),
                '2026_partial': ('2026-01-01', source['last_date'])}
    benchmarks = {s: simulate(close, flags, None, benchmark='0050', slippage=slip)
                  for s, slip in [('base', .003), ('stress', .0045)]}
    for lag in (45, 60):
        features = revenue_features(rows, close.index, close.columns, lag)
        scores = revenue_scores(fields, prices['price'], prices['trust'], features)
        available = features['growth'].notna() & features['acceleration'].notna()
        for market, group in companies.groupby('market'):
            ids = close.columns.intersection(group.stock_id)
            for year in range(2018, 2027):
                valid = close.loc[close.index.year == year, ids].notna()
                denominator = int(valid.sum().sum())
                coverage.append({'lag_days': lag, 'market': market, 'year': year,
                                 'stock_days': denominator,
                                 'complete_fraction': float((available.loc[valid.index, ids] & valid).sum().sum()/denominator)})
        for scenario, slip in [('base', .003), ('stress', .0045)]:
            benchmark = benchmarks[scenario]
            markets = ['ALL', 'TWSE', 'TPEX'] if lag == 45 and scenario == 'stress' else ['ALL']
            for market in markets:
                for key, values in scores.items():
                    if market != 'ALL':
                        values = values.where(values.columns.isin(companies.loc[companies.market.eq(market), 'stock_id'])[None, :].repeat(len(values), axis=0))
                    simulation = simulate(close, flags, values, slippage=slip)
                    comparisons = {name: {'strategy': metrics(simulation.curve, *dates),
                                          'benchmark': metrics(benchmark.curve, *dates)} for name, dates in segments.items()}
                    results.append({'rule': key, 'name': NAMES[key], 'lag_days': lag,
                                    'scenario': scenario, 'market': market,
                                    'summary': simulation.summary, 'benchmark_summary': benchmark.summary,
                                    'segments': comparisons, 'rolling': rolling_comparison(simulation.curve, benchmark.curve),
                                    'diagnostics': portfolio_diagnostics(simulation, close, companies),
                                    'equity_curve': simulation.curve.to_dict('records'),
                                    'trades': simulation.trades, 'decisions': simulation.decisions})
            print(f'完成 lag={lag} / {scenario}', flush=True)
    verify_inputs(); verify_prices()
    report = {'schema': 1, 'experiment': 'revenue_20260909', 'research_only': True, 'live_qualified': False,
              'preregistration': str(PREREG.relative_to(ROOT)), 'preregistration_sha256': digest(PREREG),
              'code_sha256': {name: digest(ROOT/name) for name in CODE},
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip(),
              'source': source, 'revenue_inputs': manifest, 'coverage': coverage,
              'finmind_requests': 0, 'load_seconds': round(loaded-started, 3),
              'elapsed_seconds': round(time.perf_counter()-started, 3),
              'limitations': ['歷史營收缺公告時點及修訂版本；45/60 天只是可用時間假設，並非真正 point-in-time',
                              '目前公司名冊仍有存活者偏差，轉板前資料排除；還原價仍有官方對帳差異',
                              '歷史已被研究使用，分年與市場比較不是新的未見測試；滾動窗互相重疊',
                              '合成還原單位，未計最低手續費、整零股及資金容量；缺月和負營收保持無效',
                              '更多篩選可能增加現金；報酬差異不全是選股能力；不把營收成長當作已獲利或確定出貨'],
              'benchmark_curves': {k: v.curve.to_dict('records') for k, v in benchmarks.items()},
              'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    compact = {k: v for k, v in report.items() if k not in ('results', 'benchmark_curves')}
    compact['results'] = [{k: v for k, v in row.items() if k not in ('equity_curve', 'trades', 'decisions')} for row in results]
    atomic_json(output.with_name(output.stem+'.summary.json'), compact)
    print(json.dumps({'report': str(output), 'comparisons': len(results),
                      'elapsed_seconds': report['elapsed_seconds'], 'finmind_requests': 0}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-inputs', action='store_true', help='本機快照及三次 FinMind 抽查；已有快照重用')
    parser.add_argument('--output', type=Path, default=CACHE/'report.json')
    args = parser.parse_args()
    with file_lock(ROOT/'.cache/research-or-update.lock', timeout=0):
        if args.prepare_inputs:
            prepare_inputs()
        else:
            run(args.output)
