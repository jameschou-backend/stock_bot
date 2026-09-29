#!/usr/bin/env python3
"""Publish the preregistered five-slot account against frozen three-slot/0050 accounts."""
from pathlib import Path
import argparse
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.export_midpoint_2025_report import verify_cash
from scripts.export_stock_universe_2019 import period_stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--left',type=Path,required=True)
    parser.add_argument('--right',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    a.output=a.output.resolve();a.output.relative_to(ROOT)
    if a.output.exists() or a.output.with_suffix('.json').exists():raise ValueError('Preserve existing publication')
    if a.left.resolve()==a.right.resolve():raise ValueError('Two independent runs required')
    reports=[read(p/'report.json') for p in (a.left,a.right)]
    for report in reports:
        if not (report['validated'] and report['all_completed'] and not report['preparation']
                and report['position_count']==5 and set(report['cases'])=={'liquid_universe'}
                and report['start']=='2019-01-02' and report['end']=='2026-09-09'
                and report['initial_cash']==1_000_000 and report['live_qualified'] is False):
            raise ValueError('Incomplete or mismatched five-slot account')
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:raise ValueError('Source sets differ')
    for p,h in reports[0]['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed source: '+p)
    values=[read(p/'liquid_universe.json') for p in (a.left,a.right)]
    for p,r in zip((a.left,a.right),reports):
        if sha(p/'liquid_universe.json')!=r['cases']['liquid_universe']['sha256']:raise ValueError('Case hash mismatch')
    if values[0]!=values[1]:raise ValueError('Full offline accounts differ')
    base=ROOT/'.cache/stock-universe-2019-20260929/final-a'
    old_report=read(base/'report.json')
    cases={'five':values[0]}
    refs={}
    for key,arm in [('three','liquid_universe'),('benchmark','benchmark')]:
        p=base/(arm+'.json')
        if sha(p)!=old_report['cases'][arm]['sha256']:raise ValueError('Frozen control differs')
        refs[str(p.relative_to(ROOT))]=sha(p);cases[key]=read(p)
    refs.update({str((p/'report.json').resolve().relative_to(ROOT)):sha(p/'report.json') for p in (a.left,a.right)})
    refs[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    five_settings=dict(cases['five']['account']['settings'])
    three_settings=dict(cases['three']['account']['settings'])
    if five_settings.pop('slots')!=5 or three_settings.pop('slots')!=3 or five_settings!=three_settings:
        raise ValueError('Control differs beyond the fixed slot count')
    signal_ref='.cache/stock-universe-2019-20260929/signals-v2.json'
    if reports[0]['source_sha256'][signal_ref]!=old_report['source_sha256'][signal_ref]:
        raise ValueError('Frozen candidates differ from three-slot control')
    if any(not (r['stock_id'].isdigit() and len(r['stock_id'])==4 and not r['stock_id'].startswith('00'))
           for r in cases['five']['account']['trades']):raise ValueError('Non-stock trade')
    comparison=[];annual=[];focus={}
    for arm,case in cases.items():
        if not case['completed']:raise ValueError('Incomplete case')
        verify_cash(case)
        account=case['account'];s=case['summary']
        if len(account['daily'])!=1867:raise ValueError('Incomplete market calendar')
        comparison.append(dict(arm=arm,**{k:s[k] for k in ('total_return','max_drawdown','final_nav','trade_count','stock_cohorts','costs')},
                               year_2026=period_stats(account['daily'],'2026-01-01','2026-09-09'),
                               average_invested_fraction=sum(r['market_value']/r['nav'] for r in account['daily'])/len(account['daily']),
                               slots_full_signals=sum(r.get('failure')=='slots_full' for r in account['orders'])))
        annual.extend(dict(arm=arm,**r) for r in s['annual'])
        focus[arm]={name:[r for r in account[name] if r.get('stock_id')=='2221' and r['date'].startswith('2026')]
                    for name in ('orders','trades')}
    a.output.mkdir(parents=True)
    for name in ('daily','trades','orders','holdings','cash_ledger'):
        pd.DataFrame(cases['five']['account'][name]).to_csv(a.output/('five-'+name+'.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame(annual).to_csv(a.output/'annual.csv',index=False,encoding='utf-8-sig')
    write(a.output/'2221.json',focus)
    result=dict(schema='stock_universe_five_20260929',start='2019-01-02',end='2026-09-09',initial_cash=1_000_000,
                comparison=comparison,annual=annual,focus_2221=focus,offline_identical=True,source_sha256=refs,
                exports_sha256={str(p.relative_to(ROOT)):sha(p) for p in a.output.iterdir()},
                live_qualified=False,unseen_validation=False,actual_fill_verified=False,
                price_basis='channel_daily_high_low_midpoint_proxy',complete_historical_universe=False,
                corporate_fractional_cash_date_verified=False)
    write(a.output.with_suffix('.json'),result)
    for row in comparison:print(row)
    print('2221',focus['five'])

if __name__=='__main__':main()
