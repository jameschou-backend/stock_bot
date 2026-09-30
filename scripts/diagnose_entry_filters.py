#!/usr/bin/env python3
"""Describe entry-to-first-exit-decision prices and reject/retain tradeoffs."""
from pathlib import Path
from collections import Counter
import argparse,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from skills.entry_filters import ARMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous diagnosis')
    inputs=ROOT/'.cache/entry-filters-20260930/signals-v1.json'
    filtered=read(inputs)
    for name,h in filtered['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed filter source '+name)
    path=ROOT/'.cache/stock-universe-2019-20260929/final-a/liquid_universe.json'
    report=read(path.parent/'report.json')
    if sha(path)!=report['cases']['liquid_universe']['sha256']:raise ValueError('Changed baseline')
    account=read(path)['account'];decisions={r['event_id']:r for r in filtered['decisions']}
    c=pd.read_parquet(ROOT/'.cache/partial-risk-2019-20260929/inputs-final/close-official.parquet').set_index('date')
    rows=[]
    for co in account['cohorts']:
        orders=[o for o in account['orders'] if o['side']=='sell' and o['event_id']==co['event_id']]
        first=min(orders,key=lambda o:o['date']) if orders else None
        end=first['signal_date'] if first else account['daily'][-1]['date']
        if end<co['entry_date']:raise ValueError('Exit instruction before entry')
        prices=c.loc[co['entry_date']:end,co['stock_id']]
        valid=bool(len(prices) and prices.notna().all() and prices.gt(0).all())
        row={k:co[k] for k in ('event_id','stock_id','name','entry_date','exit_date')}
        row.update(decisions[co['event_id']])
        row.update(decision_end=end,exit_reason=first['reason'] if first else None,
            max_close_gain=float(prices.max()/prices.iloc[0]-1) if valid else None,
            min_close_gain=float(prices.min()/prices.iloc[0]-1) if valid else None,
            decision_close_gain=float(prices.iloc[-1]/prices.iloc[0]-1) if valid else None)
        row['path_category']='open' if first is None else ('unknown' if not valid else (
            'never_above_5pct' if row['max_close_gain']<=.05 else (
            'rose15_then_nonpositive' if row['max_close_gain']>=.15 and row['decision_close_gain']<=0 else (
            'small_rise_then_nonpositive' if row['decision_close_gain']<=0 else 'positive_at_exit_decision'))))
        rows.append(row)
    publication=ROOT/'artifacts/forward_simulation/surge_capture_20260930'
    published=read(publication/'report.json');signal_path=publication/'signals.csv.gz'
    if sha(signal_path)!=published['exports_sha256']['signals.csv.gz']:raise ValueError('Changed future labels')
    signals=pd.read_csv(signal_path);signals=signals[signals.horizon.eq(60)]
    stats=[]
    for arm in ARMS:
        keep={e['event_id'] for e in filtered['entries'][arm]};part=signals[signals.event_id.isin(keep)]
        known=part.status.isin([4,5]);surge=int(part.status.eq(5).sum());negative=int((known&part.forward_return.lt(0)).sum())
        stats.append(dict(arm=arm,signals=len(part),known=int(known.sum()),unknown=int((~known).sum()),
            surge=surge,negative=negative,surge_rate=surge/int(known.sum()) if known.any() else None,
            negative_rate=negative/int(known.sum()) if known.any() else None,
            baseline_cohorts_retained=sum(r['decisions'][arm] for r in rows),
            rejected_baseline_categories=dict(Counter(r['path_category'] for r in rows if not r['decisions'][arm])),
            unknown_filter_features=sum(r['reasons'][arm]=='unknown' for r in filtered['decisions'])))
    write(a.output,dict(path_categories=dict(Counter(r['path_category'] for r in rows)),cohorts=rows,filter_statistics=stats,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (inputs,path,signal_path,Path(__file__))},
        definition='Adjusted close from first fill date to first exit SIGNAL close; not actual net cohort profit or intraday excursion',
        live_qualified=False,unseen_validation=False))
    print('path categories',dict(Counter(r['path_category'] for r in rows)))
    print(stats)


if __name__=='__main__':main()
