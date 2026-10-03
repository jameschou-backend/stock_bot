#!/usr/bin/env python3
"""Fixed institutional flow/relative-price quadrants, with lag sensitivity."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.export_signal_explorer import verify_evidence
from scripts.research_early_signal_losses import digest, evaluate_filter, statistics, PERIODS
from skills.trial_registry import append_trial_registry

SPEC=ROOT/'docs/research_sequential_prereg_20261003.md'
SPEC_SHA='7a8247ef6304e684e085d6360d9e2bf2a4649c1b1e84da48f7afebac0576efa1'
QUADRANTS=('buy_strong','buy_weak','sell_strong','sell_weak','neutral')


def flow_response_features(signals, close, quality, eligible, volume, flows, lag):
    """signals contains identity/time only; no outcomes permitted or required."""
    if lag not in (1,3):raise ValueError('Fixed lag 1 or 3 required')
    if not set(signals.columns).issubset({'signal_id','stock_id','signal_date'}):raise ValueError('T-only signal identity required')
    days=close.index
    if not days.is_unique or not days.is_monotonic_increasing or not close.columns.is_unique:
        raise ValueError('Ordered unique axes required')
    if any(not f.index.equals(days) or not f.columns.equals(close.columns) for f in (quality,eligible,volume)):
        raise ValueError('Price/identity/volume axes differ')
    if flows.duplicated(['date','stock_id']).any():raise ValueError('Duplicate flow coordinate')
    valid_price=(close.gt(0)&quality.gt(0)&np.isfinite(close)&np.isfinite(quality)&eligible.eq(True))
    rets=close/close.shift(1)-1;checks=quality/quality.shift(1)-1
    daily_ok=valid_price&valid_price.shift(1,fill_value=False)&rets.abs().le(.2)&checks.abs().le(.2)&(rets-checks).abs().le(.005)
    rel=(close/close.shift(5)-1).sub(close['0050']/close['0050'].shift(5)-1,axis=0)
    path_ok=daily_ok.rolling(5,min_periods=5).sum().eq(5)
    for col in ('foreign','trust'):
        if not col in flows:raise ValueError('Need separately complete foreign and trust')
    net=flows.set_index(['date','stock_id'])[['foreign','trust']].sum(axis=1,min_count=2).unstack('stock_id').reindex(index=days,columns=close.columns)
    net=net.where(eligible.eq(True))
    net5=net.rolling(5,min_periods=5).sum()
    vol5=volume.where(np.isfinite(volume)&volume.gt(0)&eligible.eq(True)).rolling(5,min_periods=5).sum()
    ratio=net5/vol5
    results=[]
    for r in signals.to_dict('records'):
        i=days.get_loc(pd.Timestamp(r['signal_date']))-lag;sid=r['stock_id']
        row=dict(**r,lag=lag,flow_end=None,price_start=None,flow_ratio5=None,relative_return5=None,
                 flow_quadrant=None,flow_buy_strong=None,flow_issue=None)
        if i<5 or sid not in close:row['flow_issue']='insufficient_calendar_or_stock'
        else:
            stamp=days[i];a=ratio.at[stamp,sid];b=rel.at[stamp,sid]
            row.update(flow_end=str(stamp.date()),price_start=str(days[i-5].date()))
            if not path_ok.at[stamp,sid] or not path_ok.at[stamp,'0050']:
                row['flow_issue']='price_adjustment_or_identity_unknown'
            elif not np.isfinite(a) or not np.isfinite(b):row['flow_issue']='flow_category_or_volume_missing_or_conflicting'
            else:
                quadrant=('neutral' if a==0 or b==0 else ('buy_' if a>0 else 'sell_')+('strong' if b>0 else 'weak'))
                row.update(flow_ratio5=float(a),relative_return5=float(b),flow_quadrant=quadrant,
                           flow_buy_strong=quadrant=='buy_strong')
        results.append(row)
    return results


def opportunity(rows,field):
    known=[r for r in rows if r['status']=='closed' and r[field] is not None]
    kept=[r for r in known if r[field] is True]
    return dict(known_closed=len(known),selected_closed=len(kept),
        baseline_mean=float(np.mean([r['net_return'] for r in known])) if known else None,
        selected_plus_cash_mean=sum(r['net_return'] for r in kept)/len(known) if known else None)


def summarize(rows,field):
    out=evaluate_filter(rows,field);out['opportunities']=opportunity(rows,field)
    return out


def scopes(rows):
    return {'all':rows,**{str(y):[r for r in rows if r['signal_date'].startswith(str(y))] for y in range(2019,2027)},
            **{name:[r for r in rows if lo<=r['signal_date']<=hi] for name,(lo,hi) in PERIODS.items()}}


def run(bundle,rank_dir,flow_dir,sector_dir,output,record_trials):
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('New repository output required')
    if digest(SPEC)!=SPEC_SHA:raise ValueError('Prereg changed')
    refs,_=verify_evidence(bundle,rank_dir)
    meta=json.loads((flow_dir/'manifest.json').read_text())
    if digest(flow_dir/'manifest.json')!=(flow_dir/'manifest.sha256').read_text().strip():raise ValueError('Flow manifest changed')
    for name,h in meta['files_sha256'].items():
        p=ROOT/name
        if digest(p)!=h:raise ValueError('Flow source changed: '+name)
        refs[name]=h
    refs[str((flow_dir/'manifest.json').relative_to(ROOT))]=digest(flow_dir/'manifest.json')
    sector=json.loads((sector_dir/'features.json').read_text())
    if digest(sector_dir/'features.json')!='8e5cbcb33c5747f9240b383547ca14227c8be1b46af7f90e79c84c7bb283a00d':
        raise ValueError('Frozen sector features changed')
    refs[str((sector_dir/'features.json').relative_to(ROOT))]=digest(sector_dir/'features.json')
    sector={r['signal_id']:r['group_participation'] for r in sector}
    signals=pd.read_parquet(rank_dir/'signal-ranks.parquet',columns=['signal_id','stock_id','signal_date'])
    frames={name:pd.read_parquet(bundle/(name+'.parquet')).set_index('date') for name in ('close-official','close-quality','eligibility','raw-volume')}
    for f in frames.values():f.index=pd.to_datetime(f.index)
    flows=pd.read_parquet(flow_dir/'flows.parquet');flows.date=pd.to_datetime(flows.date)
    output.mkdir(parents=True);all_features=[]
    for lag in (1,3):
        features=flow_response_features(signals,*[frames[n] for n in ('close-official','close-quality','eligibility','raw-volume')],flows,lag)
        for r in features:
            group=sector[r['signal_id']];flow=r['flow_buy_strong']
            r['flow_and_group']=bool(group and flow) if group is not None and flow is not None else None
        all_features.extend(features)
    (output/'features.json').write_text(json.dumps(all_features,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Outcomes are accessed only after both lag feature sets have been materialized.
    outcomes=pd.read_parquet(rank_dir/'signal-ranks.parquet').to_dict('records');original={r['signal_id']:r for r in outcomes}
    rows=[dict(original[f['signal_id']],**{k:v for k,v in f.items() if k not in ('stock_id','signal_date')}) for f in all_features]
    pd.DataFrame(rows).to_parquet(output/'signals.parquet',index=False)
    results={};trials=[]
    for lag in (1,3):
        lagrows=[r for r in rows if r['lag']==lag];by_scope=scopes(lagrows);result={}
        for name,subset in by_scope.items():
            result[name]=dict(filters={k:summarize(subset,k) for k in ('flow_buy_strong','flow_and_group')},
                quadrants={q:statistics([r for r in subset if r['flow_quadrant']==q]) for q in (*QUADRANTS,None)})
        for field in ('flow_buy_strong','flow_and_group'):
            coverage=sum(r[field] is not None for r in lagrows)/len(lagrows)
            improvements={n:(r['filters'][field]['opportunities']['selected_plus_cash_mean']-r['filters'][field]['opportunities']['baseline_mean'])
                          if r['filters'][field]['opportunities']['known_closed'] else None for n,r in result.items() if n in PERIODS}
            retention=result['all']['filters'][field]['return30_retention']
            gate=coverage>=.8 and retention is not None and retention>=.8 and all(v is not None and v>0 for v in improvements.values())
            record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='sequential_flow_response',
                command=' '.join(sys.argv),params=dict(lag=lag,arm=field,cash_account=False,unseen_validation=False),
                prereg_sha256=SPEC_SHA,known_coverage=coverage,period_opportunity_improvement=improvements,
                original_return30_retention=retention,research_promotion=gate,result=result['all']['filters'][field],sharpe=None)
            if record_trials:record['registry_row']=append_trial_registry(record)
            trials.append(record)
        results[str(lag)]=result
    for name,value in [('comparisons.json',results),('trials.json',trials)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for p in (SPEC,Path(__file__),ROOT/'skills/launch_flows.py',ROOT/'scripts/prepare_sequential_flows_20261003.py',ROOT/'scripts/research_early_signal_losses.py'):
        refs[str(p.relative_to(ROOT))]=digest(p)
    report=dict(schema='sequential_flow_response_v1',signals=len(original),source_sha256=refs,
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        qualification=dict(cash_account=False,live_qualified=False,first_publication_verified=False,unseen_validation=False),
        promotion=[{k:r[k] for k in ('params','known_coverage','period_opportunity_improvement','original_return30_retention','research_promotion')} for r in trials])
    p=output/'report.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'report.sha256').write_text(digest(p)+'\n');print(json.dumps(report['promotion'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--record-trials',action='store_true')
    p.add_argument('--flow-inputs',type=Path,default=ROOT/'.cache/sequential-research-20261003/flow-inputs')
    a=p.parse_args();run(ROOT/'.cache/all-signals-2019-20261002/inputs',ROOT/'.cache/signal-rank-20261003/rank-v2',a.flow_inputs.resolve(),
        ROOT/'.cache/sequential-research-20261003/sector-v1',a.output.resolve(),a.record_trials)
