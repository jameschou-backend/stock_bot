#!/usr/bin/env python3
"""Publish matched risk-account runs and explicitly retrospective diagnostics."""
from collections import Counter, defaultdict
from pathlib import Path
import sys, argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import research_midpoint as study
from skills.midpoint_risk_research import ARMS, protective_exit
from skills.scenario_exit_replay import ExitSignals


def event_profits(result):
    a,s=result['account'],result['summary']
    profits=defaultdict(float)
    events={x['action_id']:x['event_id'] for x in a['corporate_actions']}
    for row in a['cash_ledger']:
        if row['kind']=='initial_deposit':continue
        if not row.get('event_id') and not row.get('action_id'):
            if row['kind']=='fractional_share_payment' and row['cash_change']==0:
                continue
            raise ValueError('Nonzero cash flow lacks cohort attribution')
        eid=row.get('event_id') or events[row['action_id']]
        profits[eid]+=row['cash_change']
    marks={}
    for h in s['final_holdings']:
        profits[h['event_id']]+=h['market_value'];marks[h['stock_id']]=h['price']
    for r in s['final_receivables']:
        value=r['amount'] if r['kind']=='cash' else r['qty']*marks[r['stock_id']]+r['fraction']*(r['fractional_cash_per_share'] or 0)
        profits[r['event_id']]+=value
    if abs(sum(profits.values())-s['profit'])>.03:raise ValueError('Cohort profits do not reconcile')
    return dict(profits)


def diagnostics(control,benchmark):
    a=control['account'];daily=pd.DataFrame(a['daily']);base=pd.DataFrame(benchmark['account']['daily'])
    trough=daily.drawdown.idxmin();peak=daily.loc[:trough].nav.idxmax()
    start,end=daily.at[peak,'date'],daily.at[trough,'date']
    holdings=pd.DataFrame(a['holdings'])
    names=dict(zip(holdings.stock_id,holdings.name))
    ids=set(holdings.loc[holdings.date.between(start,end),'stock_id'])
    def rights(sid,day):
        relevant=[x for x in a['corporate_actions'] if x['stock_id']==sid and x['date']<=day and
                  (not x.get('pay_date') or x['pay_date']>day)]
        if any(x['kind']!='cash_dividend' for x in relevant):
            raise ValueError('Drawdown attribution needs additional noncash-right reconstruction')
        return sum(x['entitlement_value'] for x in relevant)
    contributions=[]
    for sid in sorted(ids):
        initial=holdings.loc[(holdings.stock_id==sid)&(holdings.date==start),'market_value'].sum()
        final=holdings.loc[(holdings.stock_id==sid)&(holdings.date==end),'market_value'].sum()
        cash=sum(x['cash_change'] for x in a['cash_ledger'] if x.get('stock_id')==sid and start<x['date']<=end)
        pnl=final-initial+cash+rights(sid,end)-rights(sid,start)
        contributions.append(dict(stock_id=sid,name=names[sid],pnl=pnl,peak_weight=initial/daily.at[peak,'nav']))
    loss=daily.at[trough,'nav']-daily.at[peak,'nav']
    if abs(sum(x['pnl'] for x in contributions)-loss)>.03:raise ValueError('Drawdown attribution failed')
    directory=ROOT/'.cache/historical-selector-replay-20260925/final-v7/combined'
    paths=[directory/'close-official.parquet',directory/'eligibility.parquet']
    close=pd.read_parquet(paths[0]).set_index('date');close.index=pd.to_datetime(close.index)
    eligibility=pd.read_parquet(paths[1]).set_index('date');eligibility.index=pd.to_datetime(eligibility.index)
    ids=[s for s in close if s!='0050'];ma=close[ids].rolling(20,min_periods=20).mean()
    valid=close[ids].notna()&ma.notna()&eligibility[ids].astype(bool)
    breadth=(close[ids].gt(ma)&valid).sum(axis=1)/valid.sum(axis=1)
    signals=ExitSignals(close,close.index);fixed=[]
    for cohort in a['cohorts']:
        sid=cohort['stock_id'];entry=close.index.get_loc(pd.Timestamp(cohort['entry_date']))
        state=dict(entry_index=entry,entry_price=signals.price(entry,sid),peak_price=signals.price(entry,sid))
        sells=[t for t in a['trades'] if t['event_id']==cohort['event_id'] and t['side']=='sell']
        last=close.index.get_loc(pd.Timestamp(sells[0]['date'])) if sells else len(close)-1
        for i in range(entry+1,last+1):
            context=signals.context(i,sid,state);reason=protective_exit(context)
            if reason:
                fixed.append(dict(stock_id=sid,event_id=cohort['event_id'],entry_date=cohort['entry_date'],
                    signal_date=str(close.index[i-1].date()),target_date=str(close.index[i].date()),reason=reason,
                    original_first_sale=sells[0]['date'] if sells else None,**context))
                break
    return dict(peak_date=start,trough_date=end,peak_nav=float(daily.at[peak,'nav']),
        trough_nav=float(daily.at[trough,'nav']),loss=float(loss),contributions=contributions,
        benchmark_same_window=float(base.at[trough,'nav']/base.at[peak,'nav']-1),
        breadth=[dict(date=day,above_ma20=float(breadth.loc[day]),valid_stocks=int(valid.loc[day].sum())) for day in (start,end)],
        breadth_scope='known reconstructed historical cohort; valid eligible stocks with complete MA20; not proof of complete historical market',
        fixed_entry_exit_diagnostic=fixed,fixed_entry_is_portfolio_backtest=False),paths


def publish(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():raise ValueError('Use distinct runs and a new publication')
    reports=[study.old.read(p/'report.json') for p in (left,right)]
    if any(r['preparation'] or not r['all_completed'] or set(r['cases'])!=set(ARMS) for r in reports):
        raise ValueError('All eight complete offline accounts required')
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:raise ValueError('Sources differ')
    if study.old.file_identities([ROOT/p for p in reports[0]['source_sha256']],ROOT)!=reports[0]['source_sha256']:
        raise ValueError('Sources changed')
    accounts={};rows={};refs={}
    for folder in (left,right):
        refs[str((folder/'report.json').relative_to(ROOT))]=study.old.sha(folder/'report.json')
    for arm in ARMS:
        paths=[p/(arm+'.json') for p in (left,right)];a,b=map(study.old.read,paths)
        if a!=b:raise ValueError('Offline result mismatch: '+arm)
        accounts[arm]=a
        for path in paths:refs[str(path.relative_to(ROOT))]=study.old.sha(path)
        data=pd.DataFrame(a['account']['daily']);profits=event_profits(a)
        reentries=[c for c in a['account']['cohorts'] if c.get('reentry_root')]
        rows[arm]=dict(summary=a['summary'],reentry_attempts=len(a['reentry_log']),
            reentry_cohorts=len(reentries),reentry_cohort_pnl=sum(profits[c['event_id']] for c in reentries),
            average_stock_exposure=float((data.market_value/data.nav).mean()),
            flat_sessions=int(data.holdings.eq(0).sum()),
            exit_reasons=dict(Counter(t['reason'] for t in a['account']['trades'] if t['side']=='sell')),
            result_path=str(paths[0].relative_to(ROOT)),result_sha256=study.old.sha(paths[0]),event_pnl=profits)
    benchmark_path=ROOT/'.cache/midpoint-since-2025-20260928/final-a/benchmark.json'
    benchmark=study.old.read(benchmark_path)
    refs[str(benchmark_path.relative_to(ROOT))]=study.old.sha(benchmark_path)
    diagnostic,paths=diagnostics(accounts['control'],benchmark)
    refs.update(study.old.file_identities([Path(__file__),*paths],ROOT))
    result=dict(start=reports[0]['start'],end=reports[0]['end'],arms=rows,
        benchmark=benchmark['summary'],diagnostics=diagnostic,source_sha256=refs,
        offline_identical=True,all_completed=True,live_qualified=False,actual_fill_verified=False,
        unseen_validation=False,first_batch=list(ARMS[:5]),exploratory_ablation=list(ARMS[5:]))
    study.old.write(output,result);output.with_suffix('.sha256').write_text(study.old.sha(output)+'\n')
    print({arm:{k:v['summary'][k] for k in ('total_return','max_drawdown')} for arm,v in rows.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('left');p.add_argument('right');p.add_argument('output')
    args=p.parse_args();publish(args.left,args.right,args.output)
