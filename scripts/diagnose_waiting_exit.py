#!/usr/bin/env python3
"""Describe launch times, including slow winners and unknown future windows."""
from pathlib import Path
import argparse,json,sys
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from skills.waiting_exit import waiting_context,waiting_choice,MODES


def describe(prices):
    v=np.asarray(prices,dtype=float)
    if not len(v) or not (np.isfinite(v)&(v>0)).all():return {'known':False}
    gains=v/v[0]-1
    result=dict(known=True,observed_closes=len(v),end_return=float(gains[-1]),peak_return=float(gains.max()))
    for pct in (5,10,20):
        hits=np.flatnonzero(gains>=pct/100-1e-12)
        result['first_'+str(pct)+'pct_day']=int(hits[0]+1) if len(hits) else None
    return result


def statistics(rows):
    out=[]
    for scope,group in [('all',rows),('surge60',[r for r in rows if r['status60']==5]),('negative60',[r for r in rows if r.get('forward60',{}).get('end_return',0)<0])]:
        for threshold in (5,10,20):
            values=[r['forward60'].get('first_'+str(threshold)+'pct_day') for r in group if r['forward60'].get('known')]
            reached=[v for v in values if v is not None]
            out.append(dict(group=scope,threshold=threshold,total=len(group),known=len(values),never=len(values)-len(reached),reached=len(reached),
                median_day=float(np.median(reached)) if reached else None,by_day5=sum(v<=5 for v in reached),by_day10=sum(v<=10 for v in reached)))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve prior diagnosis')
    a.output.mkdir(parents=True);refs={}
    def bind(p,expected=None):
        digest=sha(p)
        if expected and digest!=expected:raise ValueError('Source changed '+str(p))
        refs[str(p.relative_to(ROOT))]=digest
    base=ROOT/'.cache/partial-risk-2019-20260929/inputs-final';manifest=read(base/'manifest.json')
    for name in ['close-official.parquet','close-quality.parquet','eligibility.parquet']:
        bind(base/name,manifest['files_sha256'][name])
    close=pd.read_parquet(base/'close-official.parquet').set_index('date');ma=close.rolling(20).mean()
    other=pd.read_parquet(base/'close-quality.parquet').set_index('date');eligible=pd.read_parquet(base/'eligibility.parquet').set_index('date')
    pub=ROOT/'artifacts/forward_simulation/surge_capture_20260930';rp=read(pub/'report.json');bind(pub/'report.json')
    label_path=pub/'daily-labels-60.parquet';bind(label_path,rp['exports_sha256'][label_path.name]);status=pd.read_parquet(label_path)
    src=ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json';bind(src);entries=read(src)['entries']['liquid_universe']
    case_path=src.parent/'final-a/liquid_universe.json';bind(case_path,read(case_path.parent/'report.json')['cases']['liquid_universe']['sha256']);account=read(case_path)['account']
    days=close.index;lookup={d:i for i,d in enumerate(days)};all_rows=[]
    def analyze(e):
        sid=e.get('stock_id') or e['members'][0];start=lookup[pd.Timestamp(e['entry_date'])];st=int(status.at[days[start],sid]);fwd=describe(close[sid].iloc[start:start+61]) if st in (4,5) else {'known':False}
        row=dict(event_id=e['event_id'],stock_id=sid,entry_date=e['entry_date'],signal_date=e['signal_date'],status60=st,forward60=fwd)
        for mode in MODES[1:]:
            n=10 if mode=='stall10' else 5;end=start+n-1
            ctx=waiting_context(close,ma,sid,start,end) if end<len(days) else None
            row[mode]=dict(context=ctx,trigger=waiting_choice(ctx,mode) if ctx else None)
            row[mode]['after_decision']=(describe(close[sid].iloc[end:start+61])
                if fwd['known'] and ctx else {'known':False})
        return row
    for e in entries:all_rows.append(analyze(e))
    cohorts=[]
    for co in account['cohorts']:
        row=analyze(co);row['name']=co['name'];start=lookup[pd.Timestamp(co['entry_date'])]
        sells=[o for o in account['orders'] if o['event_id']==co['event_id'] and o['side']=='sell']
        end=lookup[pd.Timestamp(min(sells,key=lambda o:o['date'])['signal_date'])] if sells else len(days)-1
        p=close[co['stock_id']].iloc[start:end+1];q=other[co['stock_id']].iloc[start:end+1]
        valid=(p.notna().all() and q.notna().all() and p.gt(0).all() and q.gt(0).all() and eligible[co['stock_id']].iloc[start:end+1].all()
               and p.pct_change(fill_method=None).iloc[1:].abs().le(.2).all() and q.pct_change(fill_method=None).iloc[1:].abs().le(.2).all()
               and (p.pct_change(fill_method=None)-q.pct_change(fill_method=None)).iloc[1:].abs().le(.005).all()
               and abs(p.iloc[-1]/p.iloc[0]-q.iloc[-1]/q.iloc[0])<=.02)
        row['before_exit']=describe(p) if valid else {'known':False};row['original_exit_signal']=str(days[end].date()) if sells else None
        row['closed']=bool(sells)
        for mode in MODES[1:]:
            n=10 if mode=='stall10' else 5
            fired=None
            # Original exit has precedence when its signal occurs on the same close.
            stop=end if sells else len(days)-1
            for decision in range(start+n-1,stop):
                if waiting_choice(waiting_context(close,ma,co['stock_id'],start,decision),mode):
                    fired=decision;break
            row[mode]['first_original_eligible_day']=fired-start+1 if fired is not None else None
            row[mode]['first_original_eligible_date']=str(days[fired].date()) if fired is not None else None
        cohorts.append(row)
    for name,rows in [('signals',all_rows),('cohorts',cohorts)]:
        pd.json_normalize(rows,sep='_').to_csv(a.output/(name+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0},encoding='utf-8-sig')
    # Retention is a descriptive original-entry decision, not a synthetic portfolio.
    rejection=[]
    for name,rows in [('signals',all_rows),('cohorts',cohorts)]:
        for mode in MODES[1:]:
            fired=[r for r in rows if r[mode]['trigger'] is True]
            known=[r for r in fired if r['forward60']['known']]
            rejection.append(dict(population=name,mode=mode,total=len(rows),triggered=len(fired),known60=len(known),
                future_surge=sum(r['status60']==5 for r in known),future_negative=sum(r['forward60']['end_return']<0 for r in known),
                future_reached20=sum(r['forward60']['first_20pct_day'] is not None for r in known)))
        # Separate actual-held-window launches from the fixed future window statistics.
    before_stats=[]
    for n in (5,10,20):
        valid=[r['before_exit'] for r in cohorts if r['before_exit']['known'] and r['closed']]
        hits=[r['first_'+str(n)+'pct_day'] for r in valid if r['first_'+str(n)+'pct_day'] is not None]
        before_stats.append(dict(threshold=n,closed_known=len(valid),reached=len(hits),median_day=float(np.median(hits)) if hits else None,
            by_day5=sum(v<=5 for v in hits),by_day10=sum(v<=10 for v in hits)))
    held_rejection=[]
    for mode in MODES[1:]:
        fired=[r for r in cohorts if r[mode]['first_original_eligible_day'] is not None]
        known=[r for r in fired if r['forward60']['known']]
        held_rejection.append(dict(mode=mode,triggered=len(fired),known60=len(known),future_surge=sum(r['status60']==5 for r in known),
            future_negative=sum(r['forward60']['end_return']<0 for r in known),future_reached20=sum(r['forward60']['first_20pct_day'] is not None for r in known)))
    after_decision=[]
    # Compare only subsequent movement, excluding returns already observed when deciding.
    for year in ['all',*sorted({r['entry_date'][:4] for r in all_rows})]:
        group=[r for r in all_rows if year=='all' or r['entry_date'].startswith(year)]
        for mode in MODES[1:]:
            for trigger in (False,True):
                observed=[r[mode]['after_decision'] for r in group
                          if r[mode]['trigger'] is trigger and r[mode]['after_decision']['known']]
                after_decision.append(dict(year=year,mode=mode,trigger=trigger,known=len(observed),
                    negative=sum(r['end_return']<0 for r in observed),
                    reached20=sum(r['first_20pct_day'] is not None for r in observed),
                    median_return=float(np.median([r['end_return'] for r in observed])) if observed else None))
    bind(Path(__file__));bind(ROOT/'skills/waiting_exit.py');bind(ROOT/'docs/prereg_waiting_exit_20260930.md')
    report=dict(source_sha256=refs,before_exit_statistics=before_stats,original_held_rejection=held_rejection,after_decision_statistics=after_decision,signal_statistics=statistics(all_rows),cohort_statistics=statistics(cohorts),rejection=rejection,
        signal_status60=dict(Counter(r['status60'] for r in all_rows)),cohort_status60=dict(Counter(r['status60'] for r in cohorts)),
        before_exit_known=sum(r['before_exit']['known'] for r in cohorts),
        day_count='First fill day close is day 1; fixed horizon ends 60 sessions AFTER that close (day 61)',
        unseen_validation=False,live_qualified=False)
    write(a.output/'report.json',report)
    print(report['cohort_statistics']);print(rejection)

if __name__=='__main__':main()
