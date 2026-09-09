#!/usr/bin/env python
"""Run the fixed source-linked theme cases entirely from frozen local inputs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
import pandas as pd

from app.file_lock import file_lock
from scripts.research_flow import INPUT_DIR, atomic_json, verify_inputs
from scripts.research_rules import digest
from skills.flow_research import listing_mask
from skills.rule_research import executable
from skills.theme_research import POLICIES, replay_event

CASES = ROOT / 'docs/theme_cases_20260909.json'
PREREG = ROOT / 'docs/prereg_themes_20260909.md'
AUDIT = ROOT / 'docs/theme_price_audit_20260909.json'
OUTPUT = ROOT / '.cache/theme-research'


def load_case_prices(ledger):
    source = verify_inputs()
    wanted = pd.DataFrame({'stock_id': sorted({m['stock_id'] for c in ledger['cases']
                                             for m in c['members']} | {'0050'})})
    with duckdb.connect() as con:
        con.register('wanted', wanted)
        quotes = con.execute('SELECT q.* FROM read_parquet(?) q JOIN wanted USING(stock_id)',
                             [str(INPUT_DIR / 'quotes.parquet')]).df()
    companies = pd.read_parquet(INPUT_DIR / 'companies.parquet')
    quotes.trading_date = pd.to_datetime(quotes.trading_date)
    fields = {name: quotes.pivot(index='trading_date', columns='stock_id', values=name).sort_index()
              for name in ('adj_close', 'raw_close', 'raw_volume', 'raw_high', 'raw_low')}
    close = fields['adj_close']
    if set(close.columns) != set(wanted.stock_id):
        raise ValueError('Missing case member prices; never silently drop a member')
    mask = listing_mask(close.index, close.columns, companies)
    source.update(last_date=str(close.index[-1].date()), selected_quote_rows=len(quotes),
                  selected_stocks=len(wanted)-1, price_source='frozen_flow_snapshot',
                  finmind_requests=0)
    return {key: frame.where(mask) for key, frame in fields.items()}, source


def main():
    started = time.perf_counter()
    ledger = json.loads(CASES.read_text())
    hashes = {'cases': digest(CASES), 'preregistration': digest(PREREG)}
    if (ledger.get('experiment') != 'themes_20260909'
            or ledger.get('mode') != 'retrospective_manual_seed_cases'
            or {c['id'] for c in ledger['cases']} != {'memory', 'passive', 'leo'}):
        raise ValueError('Unsupported event ledger')
    fields, source = load_case_prices(ledger)
    audit = json.loads(AUDIT.read_text())
    for check in audit['raw_checks']:
        if fields['raw_close'].at[pd.Timestamp(check['date']), check['stock_id']] != check['official_close']:
            raise ValueError('Official price spot-check no longer matches snapshot')
    flags = executable(*(fields[x] for x in ('raw_close', 'raw_volume', 'raw_high', 'raw_low')))
    results = []
    for case in ledger['cases']:
        for scenario, slip in [('base', .003), ('stress', .0045)]:
            benchmark = replay_event(fields['adj_close'], flags, ['0050'], case['source_date'], slippage=slip)
            for policy, label in POLICIES.items():
                run = replay_event(fields['adj_close'], flags, [m['stock_id'] for m in case['members']],
                                   case['source_date'], policy=policy, slippage=slip)
                results.append({'case_id': case['id'], 'policy': policy, 'policy_name': label,
                                'scenario': scenario, **run, 'benchmark': benchmark,
                                'excess_return': run['summary']['total_return'] - benchmark['summary']['total_return']})
    verify_inputs()
    if hashes != {'cases': digest(CASES), 'preregistration': digest(PREREG)}:
        raise ValueError('Preregistered cases changed during replay')
    report = {'schema': 1, 'experiment': ledger['experiment'], 'research_only': True,
              'live_qualified': False, 'mode': ledger['mode'], 'observed_at': ledger['observed_at'],
              'source': source, 'cases': ledger['cases'], 'definition_sha256': hashes,
              'price_audit': audit, 'price_audit_sha256': digest(AUDIT),
              'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                              (Path(__file__).resolve(), ROOT/'skills/theme_research.py')},
              'preregistration': str(PREREG.relative_to(ROOT)),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'elapsed_seconds': round(time.perf_counter()-started, 3), 'results': results,
              'limitations': ['三組是事後挑選的題材案例，不是歷史全市場選股或未見樣本。',
                  '來源日期與系統整理日不同；網頁未取得首次發布封存版本，低軌衛星尚缺逐家公司訂單證據。',
                  '行情來自當前公司名冊，存在存活者偏誤；還原價格仍待官方對帳。',
                  '成本已模擬，但未納入整股、最低手續費、實際撮合與容量。',
                  '案例期間重疊，不能平均、相加或視為一個可投資組合；停損不保證在觸發價成交。']}
    atomic_json(OUTPUT / 'report.json', report)
    summary = {**report, 'results': [{k: v for k, v in r.items() if k not in ('curve', 'benchmark')}
                                   | {'benchmark_summary': r['benchmark']['summary'],
                                      'equity_curve': [{'date': a['date'], 'strategy': a['equity'],
                                                        'benchmark': b['equity']}
                                                       for a, b in zip(r['curve'], r['benchmark']['curve'], strict=True)]}
                                   for r in results]}
    atomic_json(OUTPUT / 'report.summary.json', summary)
    atomic_json(ROOT / 'docs/research_themes_20260909.json', summary)
    print(json.dumps({'elapsed_seconds': report['elapsed_seconds'], 'finmind_requests': 0,
                      'stress': [{'case': r['case_id'], 'policy': r['policy'], **r['summary'],
                                  'benchmark_return': r['benchmark']['summary']['total_return'],
                                  'excess_return': r['excess_return']} for r in results if r['scenario'] == 'stress']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with file_lock(OUTPUT / 'run.lock'):
        main()
