#!/usr/bin/env python3
"""Publish every fixed allocation arm only after two identical complete offline accounts."""
from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from scripts.export_midpoint_2025_report import verify_cash
from skills.stock_universe_2019 import ARMS

NAMES = dict(original='原版', ungrouped300='前300取消群組限制', liquid_universe='全部合格流動個股', benchmark='0050 股息再投入')


def publication_path(path):
    path = Path(path).resolve()
    path.relative_to(ROOT)
    return path


def verify_runs(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    if left == right:
        raise ValueError('Two independent offline runs are required')
    reports = [read(p/'report.json') for p in (left, right)]
    for report in reports:
        if (set(report['cases']) != set(ARMS) or not report['all_completed'] or not report['validated']
            or report['preparation'] or report['start'] != '2019-01-02' or report['end'] != '2026-09-09'
            or report['initial_cash'] != 1_000_000 or report['live_qualified'] is not False
            or report['unseen_validation'] is not False):
            raise ValueError('Incomplete or mismatched fixed experiment')
    if reports[0]['source_sha256'] != reports[1]['source_sha256']:
        raise ValueError('Offline source sets differ')
    for relative, digest in reports[0]['source_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Changed source: '+relative)
    cases = {}
    for arm in ARMS:
        values = [read(p/(arm+'.json')) for p in (left, right)]
        if values[0] != values[1] or not values[0]['completed']:
            raise ValueError('Offline accounts differ: '+arm)
        for p, report, case in zip((left, right), reports, values):
            if sha(p/(arm+'.json')) != report['cases'][arm]['sha256']:
                raise ValueError('Case hash differs: '+arm)
            verify_cash(case)
        cases[arm] = values[0]
    return cases


def period_stats(daily, start, end):
    rows = [r for r in daily if start <= r['date'] <= end]
    if not rows:
        raise ValueError('Empty comparison period')
    initial, last = rows[0]['opening_nav'], rows[-1]['nav']
    peak, drawdown = initial, 0.
    for row in rows:
        peak = max(peak, row['nav'])
        drawdown = min(drawdown, row['nav']/peak-1)
    return dict(start=rows[0]['date'], end=rows[-1]['date'], start_nav=initial,
                end_nav=last, total_return=last/initial-1, max_drawdown=drawdown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = publication_path(args.output)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Preserve published results')
    cases = verify_runs(args.left, args.right)
    args.output.mkdir(parents=True)
    summary, annual, periods = [], [], []
    benchmark = cases['benchmark']['summary']
    benchmark_years = {r['year']: r['total_return'] for r in benchmark['annual']}
    for arm, case in cases.items():
        s, account = case['summary'], case['account']
        summary.append(dict(arm=arm, label=NAMES[arm], total_return=s['total_return'],
            excess_return=s['total_return']-benchmark['total_return'], final_nav=s['final_nav'],
            max_drawdown=s['max_drawdown'], trade_count=s['trade_count'], stock_cohorts=s['stock_cohorts'],
            total_cost=s['costs']['total_cost'],
            winning_years=sum(r['total_return']>benchmark_years[r['year']] for r in s['annual']),
            years=len(s['annual'])))
        for r in s['annual']:
            annual.append(dict(arm=arm, label=NAMES[arm], **r,
                               benchmark_return=benchmark_years[r['year']],
                               excess_return=r['total_return']-benchmark_years[r['year']]))
        for start, end in (('2019-01-02', '2021-12-31'), ('2022-01-01', '2024-12-31'),
                           ('2025-01-01', '2026-09-09')):
            p = period_stats(account['daily'], start, end)
            b = period_stats(cases['benchmark']['account']['daily'], start, end)
            periods.append(dict(arm=arm, label=NAMES[arm], **p,
                                benchmark_return=b['total_return'], excess_return=p['total_return']-b['total_return']))
        for name in ('daily', 'trades', 'orders', 'holdings', 'cash_ledger'):
            pd.DataFrame(account[name]).to_csv(args.output/(arm+'-'+name+'.csv'), index=False, encoding='utf-8-sig')
        write(args.output/(arm+'-decisions.json'), account.get('candidate_decisions', []))
    for name, rows in (('comparison', summary), ('annual', annual), ('periods', periods)):
        pd.DataFrame(rows).to_csv(args.output/(name+'.csv'), index=False, encoding='utf-8-sig')
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in args.output.iterdir()}
    result = dict(schema='stock_universe_2019_20260929', comparison=summary, annual=annual, periods=periods,
        offline_identical=True, start='2019-01-02', end='2026-09-09', initial_cash=1_000_000,
        source_reports=[dict(path=str((p/'report.json').resolve().relative_to(ROOT)), sha256=sha(p/'report.json'))
                        for p in (args.left, args.right)], exports_sha256=hashes,
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        live_qualified=False, actual_fill_verified=False,corporate_fractional_cash_date_verified=False, unseen_validation=False,
        complete_historical_universe=False, price_basis='channel_daily_high_low_midpoint_proxy')
    write(args.output.with_suffix('.json'), result)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
