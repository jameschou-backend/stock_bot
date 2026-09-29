#!/usr/bin/env python3
"""Publish all fixed release arms together after identical independent runs."""
from pathlib import Path
import argparse
import sys
from collections import Counter
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.export_midpoint_2025_report import verify_cash
from scripts.export_stock_universe_2019 import period_stats

ARMS=('control3','control5','stagnant3','stagnant5','stronger3','stronger5')


def main():
    p=argparse.ArgumentParser()
    for key in ('left','right','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.relative_to(ROOT)
    if a.left.resolve()==a.right.resolve() or a.output.exists() or a.output.with_suffix('.json').exists():
        raise ValueError('Independent runs and a new publication path required')
    reports=[read(d/'report.json') for d in (a.left,a.right)]
    for report in reports:
        if (not report['validated'] or not report['all_completed'] or report['preparation']
                or set(report['cases'])!=set(ARMS) or report['start']!='2019-01-02'
                or report['end']!='2026-09-09' or report['initial_cash']!=1_000_000
                or report['live_qualified'] is not False):
            raise ValueError('Incomplete fixed suite')
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:raise ValueError('Sources differ')
    for path,h in reports[0]['source_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Source changed: '+path)
    cases={}
    for arm in ARMS:
        values=[]
        for d,r in zip((a.left,a.right),reports):
            f=d/(arm+'.json')
            if sha(f)!=r['cases'][arm]['sha256']:raise ValueError('Case hash differs')
            values.append(read(f))
        if values[0]!=values[1] or not values[0]['completed']:raise ValueError('Accounts differ: '+arm)
        verify_cash(values[0]);cases[arm]=values[0]
    for arm,folder in [('control3','stock-universe-2019-20260929'),('control5','stock-universe-five-20260929')]:
        if cases[arm]['account']!=read(ROOT/'.cache'/folder/'final-a/liquid_universe.json')['account']:
            raise ValueError('Original control changed')
    benchmark_dir=ROOT/'.cache/stock-universe-2019-20260929/final-a'
    benchmark_report=read(benchmark_dir/'report.json')
    benchmark_path=benchmark_dir/'benchmark.json'
    if sha(benchmark_path)!=benchmark_report['cases']['benchmark']['sha256']:
        raise ValueError('Frozen benchmark changed')
    cases['benchmark']=read(benchmark_path)
    comparison=[];annual=[];periods=[];focus={}
    for arm,case in cases.items():
        s=case['summary'];account=case['account'];verify_cash(case)
        if len(account['daily'])!=1867:raise ValueError('Incomplete calendar')
        if arm!='benchmark':
            settings=dict(account['settings']);mode=settings.pop('release_mode','control')
            if settings!=cases['control'+arm[-1]]['account']['settings'] or mode!=arm[:-1]:
                raise ValueError('Changes outside preregistered release rule')
        positions={r['date']:i for i,r in enumerate(account['daily'])}
        ages=[positions[c['exit_date']]-positions[c['entry_date']] for c in account['cohorts'] if c['exit_date']]
        comparison.append(dict(arm=arm,**{k:s[k] for k in ('total_return','max_drawdown','final_nav','trade_count','stock_cohorts','costs')},
            median_closed_age=float(pd.Series(ages).median()) if ages else None,
            max_closed_age=max(ages) if ages else None,release_decisions=len(account.get('release_decisions',[])),
            exit_reasons=dict(Counter(r['reason'] for r in account.get('release_decisions',[]))),
            year_2026=period_stats(account['daily'],'2026-01-01','2026-09-09')))
        annual.extend(dict(arm=arm,**r) for r in s['annual'])
        for start,end in [('2019-01-02','2021-12-31'),('2022-01-01','2024-12-31'),('2025-01-01','2026-09-09')]:
            periods.append(dict(arm=arm,**period_stats(account['daily'],start,end)))
        focus[arm]={sid:[r for r in account['trades'] if r['stock_id']==sid and r['date'].startswith('2026')]
                    for sid in ('2221','5386')}
    a.output.mkdir(parents=True)
    for arm in ARMS:
        if arm.startswith('control'):continue
        for name in ('daily','trades'):
            pd.DataFrame(cases[arm]['account'][name]).to_csv(a.output/(arm+'-'+name+'.csv'),index=False,encoding='utf-8-sig')
        write(a.output/(arm+'-decisions.json'),cases[arm]['account']['release_decisions'])
    pd.DataFrame(annual).to_csv(a.output/'annual.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(periods).to_csv(a.output/'periods.csv',index=False,encoding='utf-8-sig')
    result=dict(schema='holding_release_20260929',start='2019-01-02',end='2026-09-09',initial_cash=1_000_000,
        comparison=comparison,annual=annual,periods=periods,focus=focus,offline_identical=True,
        source_reports=[dict(path=str((d/'report.json').resolve().relative_to(ROOT)),sha256=sha(d/'report.json')) for d in (a.left,a.right)],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            str(benchmark_path.relative_to(ROOT)):sha(benchmark_path),
            str((benchmark_dir/'report.json').relative_to(ROOT)):sha(benchmark_dir/'report.json')},
        exports_sha256={str(f.relative_to(ROOT)):sha(f) for f in a.output.iterdir()},
        live_qualified=False,unseen_validation=False,actual_fill_verified=False,
        complete_historical_universe=False,corporate_fractional_cash_date_verified=False,
        price_basis='channel_daily_high_low_midpoint_proxy',no_candidate_renewal=True)
    write(a.output.with_suffix('.json'),result)
    for r in comparison:print(r)


if __name__=='__main__':main()
