#!/usr/bin/env python3
"""Publish only identical, complete, offline intraday-variant account runs."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.export_midpoint_2025_report import read,sha,verify_cash,close
from scripts.research_portfolio_intraday import validate_intraday_account


def cohorts(result):
    account=result['account'];summary=result['summary']
    paid=defaultdict(float);held=defaultdict(float);rights=defaultdict(float)
    action_events={a['action_id']:a['event_id'] for a in account['corporate_actions']}
    for r in account['cash_ledger']:
        if r['kind'] not in ('initial_deposit','buy','sell'):
            # Sealed split accounting records an explicit zero cash movement
            # without an action ID when there is no fractional share to settle.
            if r['kind']=='fractional_share_payment' and r['cash_change']==0 and 'action_id' not in r:
                continue
            if r.get('action_id') not in action_events:
                raise ValueError('Nonzero corporate cash lacks cohort attribution')
            paid[action_events[r['action_id']]]+=r['cash_change']
    marks={}
    for r in summary['final_holdings']:
        held[r['event_id']]+=r['market_value'];marks[r['stock_id']]=r['price']
    for r in summary['final_receivables']:
        value=r['amount'] if r['kind']=='cash' else r['qty']*marks[r['stock_id']]+r['fraction']*(r['fractional_cash_per_share'] or 0)
        rights[r['event_id']]+=value
    rows=[]
    for c in account['cohorts']:
        eid=c['event_id'];trades=[t for t in account['trades'] if t['event_id']==eid]
        buys=[t for t in trades if t['side']=='buy'];sells=[t for t in trades if t['side']=='sell']
        pnl=sum(t['cash_change'] for t in trades)+paid[eid]+held[eid]+rights[eid]
        rows.append(dict(event_id=eid,stock_id=c['stock_id'],name=c['name'],entry_date=c['entry_date'],
            buy_qty=sum(t['qty'] for t in buys),sell_qty=sum(t['qty'] for t in sells),
            first_sell=sells[0]['date'] if sells else None,exit_date=c['exit_date'],
            exit_reasons=sorted({t['reason'] for t in sells}),pnl=pnl,
            remaining_value=held[eid]+rights[eid]))
    close(sum(r['pnl'] for r in rows),summary['profit'],'Cohort attribution')
    return rows


def publish(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left.resolve()==right.resolve() or output.exists():raise ValueError('Use two distinct runs and a new output')
    reports=[read(p/'report.json') for p in (left,right)]
    for report in reports:
        if report['preparation'] or not report['all_completed'] or set(report['cases'])!={'control','intraday15'}:
            raise ValueError('Incomplete or preparation result cannot publish')
        for path,digest in report['source_sha256'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Source changed: '+path)
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:
        raise ValueError('Source identities differ between runs')
    results={}
    for arm in ('control','intraday15'):
        a,b=[read(p/(arm+'.json')) for p in (left,right)]
        if a!=b or not a['completed'] or a['network_calls']:
            raise ValueError('Offline accounts differ: '+arm)
        for p,r in zip((left,right),reports):
            if sha(p/(arm+'.json'))!=r['cases'][arm]['sha256']:
                raise ValueError('Case file digest changed')
        verify_cash(a);results[arm]=a
    benchmark_path=ROOT/'.cache/midpoint-since-2025-20260928/final-a/benchmark.json'
    benchmark=read(benchmark_path)
    if not benchmark['completed']:raise ValueError('Incomplete benchmark')
    dates=[r['date'] for r in results['control']['account']['daily']]
    if dates!=[r['date'] for r in benchmark['account']['daily']]:raise ValueError('Benchmark period differs')
    verify_cash(benchmark);results['benchmark']=benchmark
    validate_intraday_account(results['intraday15']['account'],dates,dates[0],dates[-1])
    rows={k:cohorts(v) for k,v in results.items() if k!='benchmark'}
    maps={k:{r['event_id']:r for r in v} for k,v in rows.items()}
    differences=[]
    for eid in sorted(set(maps['control'])|set(maps['intraday15'])):
        a,b=maps['control'].get(eid),maps['intraday15'].get(eid);ref=b or a
        differences.append(dict(event_id=eid,stock_id=ref['stock_id'],name=ref['name'],
            control=a,candidate=b,pnl_difference=(b['pnl'] if b else 0)-(a['pnl'] if a else 0)))
    close(sum(d['pnl_difference'] for d in differences),
        results['intraday15']['summary']['profit']-results['control']['summary']['profit'],'Account profit difference')
    report=dict(start=dates[0],end=dates[-1],initial_cash=1000000,
        summaries={k:r['summary'] for k,r in results.items()},cohort_differences=differences,
        audits={k:r['audit'] for k,r in results.items() if k!='benchmark'},
        offline_identical=True,account_and_cohort_reconciled=True,
        policy='entry_next_session_activation_board_intraday_odd_next_session_HL2',
        actual_fill_verified=False,live_qualified=False,unseen_validation=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            *(f/n for f in (left,right) for n in ('report.json','control.json','intraday15.json')),
            benchmark_path,Path(__file__)]})
    output.mkdir(parents=True)
    import json
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
    for arm in ('control','intraday15'):
        pd.DataFrame(results[arm]['account']['trades']).to_csv(output/(arm+'-trades.csv'),index=False)
    daily=pd.DataFrame({'date':dates})
    for arm,r in results.items():
        for col in ('nav','cash','drawdown','total_return'):
            daily[arm+'_'+col]=[d[col] for d in r['account']['daily']]
    daily.to_csv(output/'daily-comparison.csv',index=False)
    print(json.dumps({k:{f:r['summary'][f] for f in ('total_return','max_drawdown','final_nav')} for k,r in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs=2,required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();publish(*a.runs,a.output)
