#!/usr/bin/env python
"""Prepare causal signal snapshots, then run the fixed event/group comparisons offline."""
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
import numpy as np
import pandas as pd

from app.file_lock import file_lock
from scripts.prepare_event_groups import CACHE, verify as verify_news
from scripts.research_flow import load_inputs, verify_inputs as verify_prices, atomic_json, save_frame
from scripts.research_rules import digest
from skills.event_group_research import RULES, extract_news, build_signals, simulate_signals
from skills.official_adj_factors import compute_stock_factor_series
from skills.flow_research import portfolio_diagnostics, rolling_comparison
from skills.rule_research import executable, simulate

PREREG = ROOT/'docs/prereg_event_groups_20260909.md'
SOURCE_AUDIT = ROOT/'docs/event_news_source_audit_20260909.json'
CODE = ('scripts/prepare_event_groups.py', 'scripts/research_event_groups.py',
        'scripts/research_flow.py', 'skills/event_group_research.py', 'skills/news_radar.py',
        'skills/official_adj_factors.py', 'skills/rule_research.py', 'skills/flow_research.py')
# These six equity action feeds omit ETF unit splits. Explicit official evidence,
# not a price-inferred jump correction: TWSE notice dated 2025-05-14, 4-for-1.
BENCHMARK_SPLIT = {'stock_id':'0050','effective_date':'2025-06-18','units_multiplier':4,
    'source':'https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d696b406fc0196ce27c2e90063.pdf',
    'issuer_confirmation':'https://api.yuantafunds.com/ECImage/ECBackstage/file/2025-06-17/ca6610a73d444bc09bc2035df79bfd25.pdf'}


def code_hashes():
    return {name: digest(ROOT/name) for name in CODE}


def save_matrix(name, values):
    save_frame(CACHE/name, values.rename_axis('date').reset_index())


def read_matrix(name):
    frame = pd.read_parquet(CACHE/name).set_index('date')
    frame.index = pd.to_datetime(frame.index)
    return frame


def official_close(fields, events):
    raw = fields['raw_close'].where(fields['raw_close'].gt(0) & np.isfinite(fields['raw_close']))
    adjusted = raw.copy()
    groups = {sid: group for sid, group in events.groupby('stock_id')}
    for sid in raw:
        group = groups.get(sid)
        # All six official source date windows have been checked by preparation.
        # An identity with no action has an explicit constant factor of one.
        if group is not None:
            factor = compute_stock_factor_series(group.event_date, group.ratio, raw.index.date)
            adjusted[sid] = raw[sid]*factor
    if '0050' in adjusted:
        on_split = events.stock_id.eq('0050') & pd.to_datetime(events.event_date).eq(BENCHMARK_SPLIT['effective_date'])
        if on_split.any():
            raise ValueError('0050 split date now appears in action feed; reconcile before applying explicit split')
        adjusted.loc[adjusted.index < pd.Timestamp(BENCHMARK_SPLIT['effective_date']), '0050'] /= 4
    if ((adjusted <= 0) | np.isinf(adjusted)).any().any():
        raise ValueError('Invalid official adjusted prices')
    return adjusted


def prepare_signals():
    started = time.perf_counter()
    manifest = verify_news()
    fields, _, companies, source = load_inputs()
    fields = {k:v.loc['2021-01-01':].copy() for k,v in fields.items()}
    pieces = [pd.read_parquet(CACHE/name) for name in manifest['files_sha256'] if name.startswith('news-20') and name.endswith('.parquet')]
    for day in manifest['gap_dates']:
        part = pd.read_parquet(CACHE/f'gap-{day}.parquet')
        if part.empty:
            continue
        part = part.rename(columns={'date': 'news_datetime'})
        part['created_at'] = json.loads((CACHE/f'gap-{day}.meta.json').read_text())['saved_at']
        pieces.append(part[['stock_id', 'news_datetime', 'created_at', 'title', 'source', 'link']])
    news = pd.concat(pieces, ignore_index=True)
    mentions, events, rejected, counts = extract_news(news, companies.set_index('stock_id').name.to_dict())
    del news, pieces
    print(f'文字判讀：{counts}，營運線索 {len(events)}，題材點名 {len(mentions)}', flush=True)
    atomic_json(CACHE/'events.json', events)
    atomic_json(CACHE/'rejected.json', rejected)
    save_frame(CACHE/'mentions.parquet', mentions)
    action_rows = pd.read_parquet(CACHE/'official-events.parquet')
    official = official_close(fields, action_rows)
    flags = executable(*(fields[n] for n in ('raw_close', 'raw_volume', 'raw_high', 'raw_low')))
    save_matrix('trade-flags.parquet', flags)
    provenance = {'news': manifest, 'prices': source}
    files = ['events.json', 'rejected.json', 'mentions.parquet', 'trade-flags.parquet']
    build_stats = {}
    for basis, close in [('official', official), ('snapshot', fields['adj_close'])]:
        before = time.perf_counter()
        scores, annotated, group = build_signals({**fields, 'adj_close': close}, mentions, events)
        for key, values in scores.items():
            name = f'scores-{basis}-{key}.parquet'
            save_matrix(name, values); files.append(name)
        save_matrix(f'close-{basis}.parquet', close); files.append(f'close-{basis}.parquet')
        atomic_json(CACHE/f'events-{basis}.json', annotated); files.append(f'events-{basis}.json')
        atomic_json(CACHE/f'groups-{basis}.json', group); files.append(f'groups-{basis}.json')
        build_stats[basis] = {'seconds': round(time.perf_counter()-before, 3),
            'signal_stock_days': {key: int(value.loc['2022-01-01':].notna().sum().sum()) for key,value in scores.items()},
            'events': len(annotated), 'price_eligible_events': sum(x['price_eligible'] for x in annotated),
            'with_peer_confirmation': sum(x['price_eligible'] and x['peer_evidence']['confirmed'] for x in annotated)}
        print(basis, build_stats[basis], flush=True)
    relative = ((official/official.shift(1))/(fields['adj_close']/fields['adj_close'].shift(1))-1).abs()
    paired = relative.stack()
    benchmark_prices = pd.DataFrame({'official':official['0050'], 'snapshot':fields['adj_close']['0050']}).dropna()
    benchmark_returns = benchmark_prices.pct_change(fill_method=None)
    benchmark_difference = ((1+benchmark_returns.official)/(1+benchmark_returns.snapshot)-1).abs().dropna()
    if benchmark_difference.gt(.005).any():
        raise ValueError('0050 adjustment bases differ by over 50bp between observed quotes; reconcile benchmark before diagnostics')
    action_check = {'official_events': len(action_rows),
                    'nonpositive_raw_prices_masked': int(fields['raw_close'].le(0).sum().sum()),
                    'cash_increase_suspected': int(action_rows.cash_increase_suspected.sum()),
                    'multiple_actions_same_date': int(action_rows.duplicated(['stock_id','event_date'], keep=False).sum()),
                    'comparable_daily_returns': len(paired),
                    'daily_return_differences_over_50bp': int(paired.gt(.005).sum()),
                    'max_daily_return_difference': float(paired.max())}
    action_check['benchmark_split'] = BENCHMARK_SPLIT
    action_check['benchmark_max_observed_return_difference'] = float(benchmark_difference.max())
    verify_news(); verify_prices()
    info = {'schema': 1, 'experiment': 'event_groups_20260909',
            'prepared_at': datetime.now(timezone.utc).isoformat(), 'provenance': provenance,
            'counts': counts, 'build_stats': build_stats, 'action_check': action_check,
            'elapsed_seconds': round(time.perf_counter()-started, 3),
            'code_sha256': code_hashes(), 'preregistration_sha256': digest(PREREG),
            'source_audit_sha256': digest(SOURCE_AUDIT),
            'files_sha256': {name: digest(CACHE/name) for name in files}}
    atomic_json(CACHE/'signal-inputs.json', info)
    print(json.dumps({'signal_preparation_seconds': info['elapsed_seconds'], 'finmind_requests': 0}), flush=True)
    return info


def verify_signals():
    info = json.loads((CACHE/'signal-inputs.json').read_text())
    if info['code_sha256'] != code_hashes() or info['preregistration_sha256'] != digest(PREREG):
        raise ValueError('Research code/spec changed; explicitly run --prepare-signals again')
    if info['source_audit_sha256'] != digest(SOURCE_AUDIT):
        raise ValueError('Source audit changed; explicitly run --prepare-signals again')
    if info['provenance']['news'] != verify_news():
        raise ValueError('News manifest changed')
    verify_prices()
    for name, expected in info['files_sha256'].items():
        if Path(name).name != name or digest(CACHE/name) != expected:
            raise ValueError('Signal snapshot changed: '+name)
    return info


def independent_event_outcomes(close, flags, events):
    """All events retained, including losses and unavailable endpoints; not a portfolio."""
    result = []
    days = close.index
    prices = close.to_numpy(float)
    tradable = flags.to_numpy(bool) & np.isfinite(prices) & (prices > 0)
    columns = {sid:i for i,sid in enumerate(close.columns)}
    bm_column = columns['0050']
    for event in events:
        if event['signal_date'] < '2022-01-01':
            continue
        first = int(days.searchsorted(event['signal_date']))+1
        last = first+63
        row = {**event, 'horizon': 63, 'stock_net': None, 'benchmark_net': None, 'excess': None}
        if last >= len(days):
            row['outcome_status'] = 'incomplete_horizon'
        else:
            sid = event['stock_id']
            column = columns[sid]
            row.update(entry_date=str(days[first].date()), exit_date=str(days[last].date()))
            if not tradable[np.ix_([first,last],[column,bm_column])].all():
                row['outcome_status'] = 'endpoint_untradable'
            else:
                a = prices[first]; b = prices[last]
                stock_return = float(b[column]/a[column]*(1-.001425-.0045-.003)/(1+.001425+.0045)-1)
                bm = float(b[bm_column]/a[bm_column]*(1-.001425-.0045-.001)/(1+.001425+.0045)-1)
                row.update(outcome_status='complete', stock_net=stock_return, benchmark_net=bm, excess=stock_return-bm)
        result.append(row)
    return result


def require_diagnostic_mode(diagnostic_only):
    # This experiment has no eligible historical publication-time archive.
    # A clean sample alone must never promote the whole corpus to valid evidence.
    if not diagnostic_only:
        raise ValueError('News publication audit failed/unverified; only --diagnostic-only is allowed. Returns cannot qualify a strategy.')


def run(*, diagnostic_only=False):
    require_diagnostic_mode(diagnostic_only)
    started = time.perf_counter()
    inputs = verify_signals()
    companies = pd.read_parquet(ROOT/'.cache/growth-flow-research/companies.parquet')
    flags = read_matrix('trade-flags.parquet')
    results, benchmark_curves = [], {}
    for basis in ('official', 'snapshot'):
        close = read_matrix(f'close-{basis}.parquet')
        scores = {key:read_matrix(f'scores-{basis}-{key}.parquet') for key in RULES}
        for scenario, slip in [('base', .003), ('stress', .0045)]:
            benchmark = simulate(close, flags, None, benchmark='0050', start='2022-01-03', slippage=slip)
            benchmark_curves[f'{basis}-{scenario}'] = benchmark.curve.to_dict('records')
            delays = (0,1) if basis == 'official' and scenario == 'stress' else (0,)
            for delay in delays:
                for horizon in (63,126):
                    for key, values in scores.items():
                        sim, completed = simulate_signals(close, flags, values.shift(delay), horizon=horizon, slippage=slip)
                        diagnostics = portfolio_diagnostics(sim, close, companies)
                        worst = sorted(completed, key=lambda r:(r['net_return'],r['entry_date'],r['stock_id']))[:5]
                        best = sorted(completed, key=lambda r:(-r['net_return'],r['entry_date'],r['stock_id']))[:5]
                        results.append({'rule':key,'name':RULES[key],'basis':basis,'scenario':scenario,
                            'delay':delay,'horizon':horizon,'summary':sim.summary,'benchmark_summary':benchmark.summary,
                            'rolling':rolling_comparison(sim.curve,benchmark.curve),'diagnostics':diagnostics,
                            'best_trades':best,'worst_trades':worst,'completed':completed,
                            'curve':sim.curve.to_dict('records'),'trades':sim.trades,'decisions':sim.decisions})
            print(f'完成 {basis} / {scenario}', flush=True)
    event_outcomes = independent_event_outcomes(read_matrix('close-official.parquet'), flags,
                                               json.loads((CACHE/'events-official.json').read_text()))
    atomic_json(CACHE/'event-outcomes.json', event_outcomes)
    comparable = [e for e in event_outcomes if e['outcome_status']=='complete' and e['price_eligible']]
    event_stats = {'all_retained_events':len(event_outcomes),'comparable_price_eligible_events':len(comparable),
                  'negative_stock_returns':sum(e['stock_net']<0 for e in comparable),
                  'underperformed_0050':sum(e['excess']<0 for e in comparable),
                  'unknown_endpoints':sum(e['outcome_status']!='complete' for e in event_outcomes),
                  'median_excess':float(np.median([e['excess'] for e in comparable])) if comparable else None}
    verify_signals()
    report = {'schema':1,'experiment':'event_groups_20260909','research_only':True,'live_qualified':False,
              'diagnostic_only':True,'valid_strategy_evidence':False,
              'source_audit':json.loads(SOURCE_AUDIT.read_text()),'source_audit_sha256':digest(SOURCE_AUDIT),
              'generated_at':datetime.now(timezone.utc).isoformat(),
              'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'code_sha256':code_hashes(),'preregistration_sha256':digest(PREREG),
              'signal_manifest_sha256':digest(CACHE/'signal-inputs.json'),
              'inputs':inputs,'event_stats':event_stats,'finmind_requests':0,
              'elapsed_seconds':round(time.perf_counter()-started,3),
              'qualification':{'publication_versions':'unverified_backfilled_news',
                               'historical_universe':'current_company_cohort_only',
                               'corporate_actions':'official_reference_price_adjustment_not_cash_ledger',
                               'unseen_holdout':False},
              'limitations':['原文抽查已發現新聞日期錯置；以下保留供應商時序僅作程式與資料診斷，不可用來判定策略優劣',
                             '營運線索由標題規則判讀，未逐篇證實；原始首次發佈版本及時區不足，新聞為事後回補',
                             '題材成員只取過去新聞明確點名，不代表供應鏈受惠；字典與公司名稱現在才固定',
                             '當前公司名冊仍有存活者偏差，轉板前期間排除，缺完整歷史上市下市清單',
                             '官方參考價還原不是逐股股息現金帳，現金增資等需另核對；兩種還原口徑皆不保證全對',
                             '缺新聞日期與資料不齊皆保留揭露；事件重疊，單筆事件損益不可相加成投資績效',
                             '合成還原單位，未模擬交易單位、最低費用與個人容量；少曝險可能解釋報酬差異',
                             '歷史已研究過，不是未見樣本外；收盤停損次日執行，不保證最大損失'],
              'benchmark_curves':benchmark_curves,'results':results}
    atomic_json(CACHE/'report.json',report)
    compact = {k:v for k,v in report.items() if k not in ('results','benchmark_curves')}
    compact['results']=[{k:v for k,v in row.items() if k not in ('completed','curve','trades','decisions')} for row in results]
    atomic_json(CACHE/'report.summary.json',compact)
    print(json.dumps({'comparisons':len(results),'elapsed_seconds':report['elapsed_seconds'],
                      'event_stats':event_stats,'finmind_requests':0}),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-signals',action='store_true')
    parser.add_argument('--diagnostic-only',action='store_true',help='Retain the failed news clock only for diagnostics, never valid strategy evidence')
    args=parser.parse_args()
    with file_lock(ROOT/'.cache/research-or-update.lock',timeout=0):
        prepare_signals() if args.prepare_signals else run(diagnostic_only=args.diagnostic_only)
