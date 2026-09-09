#!/usr/bin/env python
"""Prepare frozen inputs or run the preregistered trust/volume comparison offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import select

from app.file_lock import file_lock
from skills.flow_research import (FLOW_NAMES, company_universe, listing_mask, flow_scores,
                                 portfolio_diagnostics, rolling_comparison)
from skills.rule_research import executable, simulate, metrics
from scripts.research_rules import digest

INPUT_DIR = ROOT / '.cache/growth-flow-research'
PREREG = ROOT / 'docs/prereg_flow_20260909.md'
OFFICIAL = {
    'twse-company.json': 'https://openapi.twse.com.tw/v1/opendata/t187ap03_L',
    'tpex-company.json': 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O',
}


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')
    tmp.replace(path)


def save_frame(path, frame):
    tmp = path.with_suffix('.tmp')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def prepare_inputs():
    """At most two official requests; zero FinMind requests and no DB writes."""
    import requests
    from app.db import get_session
    from app.models import RawInstitutional

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = INPUT_DIR / 'inputs.json'
    if manifest_path.exists():
        print('研究輸入已固定；重用現有快照，不重新讀取 DB 或 API。', flush=True)
        verify_inputs()
        return
    for name, url in OFFICIAL.items():
        path = INPUT_DIR / name
        if not path.exists():
            response = requests.get(url, timeout=(5, 25), headers={'Accept': 'application/json'})
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list) or len(rows) < 500:
                raise ValueError(f'官方公司名冊不足：{name}')
            atomic_json(path, rows)
    companies = company_universe(*(json.loads((INPUT_DIR / name).read_text()) for name in OFFICIAL))
    save_frame(INPUT_DIR / 'companies.parquet', companies)
    allowed = pd.DataFrame({'stock_id': list(companies.stock_id) + ['0050']})
    source_paths = [ROOT / 'artifacts/adj_prices/adj_prices_10y.parquet',
                    ROOT / 'artifacts/cache/prices.parquet',
                    ROOT / '.cache/rule-research/0050-raw.parquet',
                    ROOT / '.cache/rule-research/raw-20260521.parquet',
                    ROOT / '.cache/rule-research/raw-20260522.parquet']
    for path in source_paths:
        if not path.exists():
            raise ValueError(f'缺少行情來源：{path}；先依前輪 research-rules 準備明確缺口')
    source_hash = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    quote_path = INPUT_DIR / 'quotes.parquet'
    # A partial run's quotes are rebuilt only before the input manifest is sealed.
    print('建立獨立行情快照…', flush=True)
    with duckdb.connect() as con:
        con.register('allowed', allowed)
        quotes = con.execute('''SELECT a.stock_id, CAST(a.trading_date AS DATE) AS trading_date,
            a.close AS adj_close, r.close AS raw_close, r.volume AS raw_volume,
            r.high AS raw_high, r.low AS raw_low
            FROM read_parquet(?) a JOIN allowed USING(stock_id)
            LEFT JOIN (
                SELECT * FROM read_parquet(?) WHERE stock_id!='0050'
                    AND trading_date NOT IN (DATE '2026-05-21', DATE '2026-05-22')
                UNION ALL SELECT * FROM read_parquet(?)
                UNION ALL SELECT * FROM read_parquet(?) WHERE stock_id!='0050'
            ) r ON a.stock_id=r.stock_id AND CAST(a.trading_date AS DATE)=r.trading_date
            WHERE a.trading_date >= '2016-02-15'
        ''', [str(source_paths[0]), str(source_paths[1]), str(source_paths[2]),
              [str(p) for p in source_paths[3:]]]).df()
    if quotes.empty or quotes.duplicated(['stock_id', 'trading_date']).any():
        raise ValueError('行情快照為空或股票日期重複')
    if source_hash != {str(p.relative_to(ROOT)): digest(p) for p in source_paths}:
        raise ValueError('建立快照時行情來源改變，停止')
    save_frame(quote_path, quotes)
    institution_path = INPUT_DIR / 'institutional.parquet'
    if not institution_path.exists():
        print('一次匯出本機投信資料；之後回測直接重用 Parquet…', flush=True)
        with get_session() as session:
            query = select(RawInstitutional.stock_id, RawInstitutional.trading_date,
                           RawInstitutional.trust_net).where(
                               RawInstitutional.trading_date >= quotes.trading_date.min().date(),
                               RawInstitutional.trading_date <= quotes.trading_date.max().date())
            inst = pd.read_sql(query, session.get_bind())
        inst = inst[inst.stock_id.isin(allowed.stock_id)].copy()
        if inst.empty or inst.duplicated(['stock_id', 'trading_date']).any():
            raise ValueError('投信資料為空或股票日期重複')
        save_frame(institution_path, inst)
    paths = [INPUT_DIR / name for name in OFFICIAL] + [INPUT_DIR / 'companies.parquet', quote_path, institution_path]
    atomic_json(manifest_path, {
        'schema': 1, 'prepared_at': datetime.now(timezone.utc).isoformat(),
        'finmind_requests': 0, 'official_sources': OFFICIAL, 'original_prices_sha256': source_hash,
        'files_sha256': {p.name: digest(p) for p in paths},
        'universe': 'current_official_cohort_with_listing_date_mask_not_historical_universe',
    })
    print('研究輸入已固定。', flush=True)


def verify_inputs():
    path = INPUT_DIR / 'inputs.json'
    if not path.exists():
        raise ValueError('尚無固定輸入；先執行 python scripts/research_flow.py --prepare-inputs')
    manifest = json.loads(path.read_text())
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(INPUT_DIR / name) != expected:
            raise ValueError(f'研究輸入變更：{name}；不能靜默沿用舊結果')
    return manifest


def load_inputs():
    source = verify_inputs()
    quotes = pd.read_parquet(INPUT_DIR / 'quotes.parquet')
    companies = pd.read_parquet(INPUT_DIR / 'companies.parquet')
    inst = pd.read_parquet(INPUT_DIR / 'institutional.parquet')
    quotes.trading_date = pd.to_datetime(quotes.trading_date)
    inst.trading_date = pd.to_datetime(inst.trading_date)
    fields = {name: quotes.pivot(index='trading_date', columns='stock_id', values=name).sort_index()
              for name in ('adj_close', 'raw_close', 'raw_volume', 'raw_high', 'raw_low')}
    close = fields['adj_close']
    if '0050' not in close or (close <= 0).any().any():
        raise ValueError('還原行情非正或缺少 0050')
    allowed = listing_mask(close.index, close.columns, companies)
    source.update(last_date=str(close.index[-1].date()), quote_rows=len(quotes),
                  official_companies=len(companies), stocks_with_quotes=len(close.columns)-1,
                  official_companies_without_quotes=sorted(set(companies.stock_id)-set(close.columns)),
                  prelisting_quotes_masked=int((~allowed & close.notna()).sum().sum()))
    fields = {name: frame.where(allowed) for name, frame in fields.items()}
    coverage = fields['raw_close'].notna().sum(axis=1) / fields['adj_close'].notna().sum(axis=1)
    if coverage.min() < .9:
        raise ValueError('原始行情存在整日缺漏，不可當成空手')
    net = inst.pivot(index='trading_date', columns='stock_id', values='trust_net').reindex(
        index=close.index, columns=close.columns).where(allowed)
    stock_days = fields['adj_close'].drop(columns='0050').notna()
    complete = net.drop(columns='0050').rolling(20, min_periods=20).count().eq(20)
    source.update(minimum_daily_raw_coverage=float(coverage.min()),
                  institutional_snapshot_rows=len(inst),
                  institutional_observed_fraction=float((net.drop(columns='0050').notna() & stock_days).sum().sum()/stock_days.sum().sum()),
                  complete_20day_institution_fraction=float((complete & stock_days).sum().sum()/stock_days.sum().sum()))
    source['institutional_coverage_by_year_market'] = []
    for market, companies_in_market in companies.groupby('market'):
        ids = companies_in_market.stock_id[companies_in_market.stock_id.isin(stock_days.columns)]
        for year in sorted(set(close.index.year)):
            valid = stock_days.loc[close.index.year == year, ids]
            observed = net.loc[valid.index, ids].notna() & valid
            full = complete.loc[valid.index, ids] & valid
            denominator = int(valid.sum().sum())
            source['institutional_coverage_by_year_market'].append({
                'year': int(year), 'market': market, 'quoted_stock_days': denominator,
                'observed_fraction': float(observed.sum().sum()/denominator),
                'complete_20day_fraction': float(full.sum().sum()/denominator)})
    unusual = net.abs().gt(fields['raw_volume']) & fields['raw_volume'].gt(0) & fields['adj_close'].notna()
    source['trust_net_exceeds_daily_volume_rows'] = int(unusual.sum().sum())
    return fields, net, companies, source


def main(output):
    started = time.perf_counter()
    fields, net, companies, source = load_inputs()
    loaded = time.perf_counter()
    close = fields['adj_close']
    flags = executable(*(fields[x] for x in ('raw_close', 'raw_volume', 'raw_high', 'raw_low')))
    scores = flow_scores(close, *(fields[x] for x in ('raw_close', 'raw_volume', 'raw_high', 'raw_low')), net)
    first = int(close.index.searchsorted(pd.Timestamp('2018-01-01')))
    signal_positions = [i-1 for i in range(first, len(close))
                        if i == first or close.index[i].to_period('M') != close.index[i-1].to_period('M')]
    eligible = scores['price'].iloc[signal_positions].notna()
    inst_complete = net.rolling(20, min_periods=20).count().eq(20).iloc[signal_positions]
    source['price_candidates_on_signal_dates'] = int(eligible.sum().sum())
    source['price_candidates_missing_20day_institution'] = int((eligible & ~inst_complete).sum().sum())
    segments = {'2018_2022': ('2018-01-01', '2022-12-31'),
                '2023_2025': ('2023-01-01', '2025-12-31'),
                '2026_partial': ('2026-01-01', source['last_date'])}
    results = []
    for scenario, slip in [('base', .003), ('stress', .0045)]:
        benchmark = simulate(close, flags, None, benchmark='0050', slippage=slip)
        for rule, values in scores.items():
            print(f'比較 {FLOW_NAMES[rule]} / {scenario}', flush=True)
            run = simulate(close, flags, values, slippage=slip)
            comparison = {name: {'strategy': metrics(run.curve, *dates),
                                 'benchmark': metrics(benchmark.curve, *dates)} for name, dates in segments.items()}
            for stats in comparison.values():
                stats['excess_return'] = stats['strategy']['total_return'] - stats['benchmark']['total_return']
            results.append({'rule': rule, 'name': FLOW_NAMES[rule], 'scenario': scenario,
                            'summary': run.summary, 'benchmark_summary': benchmark.summary,
                            'segments': comparison, 'rolling': rolling_comparison(run.curve, benchmark.curve),
                            'diagnostics': portfolio_diagnostics(run, close, companies),
                            'equity_curve': run.curve.to_dict('records'),
                            'benchmark_curve': benchmark.curve.to_dict('records'),
                            'trades': run.trades, 'decisions': run.decisions})
    verify_inputs()
    report = {'schema': 1, 'experiment': 'flow_20260909', 'research_only': True, 'live_qualified': False,
              'offline': True, 'source': source, 'preregistration': str(PREREG.relative_to(ROOT)),
              'preregistration_sha256': digest(PREREG),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                              (ROOT/'scripts/research_flow.py', ROOT/'skills/flow_research.py', ROOT/'skills/rule_research.py')},
              'load_seconds': round(loaded-started, 3), 'elapsed_seconds': round(time.perf_counter()-started, 3),
              'limitations': ['當前公司名冊仍有存活者偏差；轉板前期間排除，未重建完整歷史股票池',
                              '還原價尚未完成官方對帳，不可當作可實現績效',
                              '歷史已被研究使用，不是全新樣本外；重疊滾動窗不是獨立樣本',
                              '本輪只有價量與投信，不含營收、獲利或產品出貨事件',
                              '合成還原單位，不含最低手續費、交易單位及個人資金容量',
                              '法人歷史修正版本與可交易性僅近似，缺資料不視為零買超'],
              'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    compact = {k: v for k, v in report.items() if k != 'results'}
    compact['results'] = [{k: v for k, v in row.items() if k not in
                          ('equity_curve', 'benchmark_curve', 'trades', 'decisions')} for row in results]
    atomic_json(output.with_name(output.stem + '.summary.json'), compact)
    print(json.dumps({'report': str(output), 'elapsed_seconds': report['elapsed_seconds'],
                      'load_seconds': report['load_seconds'], 'finmind_requests': 0}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-inputs', action='store_true', help='固定官方名冊及本機 DB 快照，零 FinMind 請求')
    parser.add_argument('--output', type=Path, default=INPUT_DIR / 'report.json')
    args = parser.parse_args()
    with file_lock(ROOT / '.cache/research-or-update.lock', timeout=0):
        if args.prepare_inputs:
            prepare_inputs()
        else:
            main(args.output)
