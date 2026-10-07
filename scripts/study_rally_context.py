#!/usr/bin/env python3
"""Fixed offline context study; labels never feed signal construction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.scan_market_strategies import load_poc
from scripts.study_rally_precursors import forward_labels
from skills.rally_context_features import COHORTS, FILTERS, build_signal_features, attach_context
from skills.rally_context_stats import summarize_filters
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.outcomes import COSTS, _net
from skills.trial_registry import append_trial_registry


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verified_json(ref, sources):
    path=ROOT/ref['path']
    actual=digest(path)
    if actual!=ref['sha256']:raise ValueError('Source hash mismatch: '+str(path))
    sources[ref['path']]=actual
    return json.loads(path.read_text())


def fixed_outcomes(events, f, days, ids):
    """Apply future paths only after all signal/context features are materialized."""
    parts=[]
    rr,cc=events.signal_index.to_numpy(int),events.column_index.to_numpy(int)
    n=len(days);bj=ids.index('0050')
    open_adj=(f['open']*f['c']/f['close']).to_numpy(float)
    for horizon,threshold in ((20,.30),(60,.50)):
        labels=forward_labels(f,horizon,threshold)
        part=events.copy()
        part['horizon']=horizon;part['threshold']=threshold
        part['entry_date']=[str(days[i+1].date()) if i+1<n else None for i in rr]
        part['exit_date']=[str(days[i+horizon].date()) if i+horizon<n else None for i in rr]
        for dest,source in [('mature','mature'),('complete','complete'),('gross_return','gross'),('net_return','net')]:
            part[dest]=labels[source][rr,cc]
        bm=np.full(n,np.nan)
        upto=np.arange(n-horizon)
        values=_net(open_adj[upto+1,bj],f['c'].to_numpy()[upto+horizon,bj],COSTS['benchmark_sell_tax'])
        bm[upto]=np.where(labels['complete'][upto,bj],values,np.nan)
        part['benchmark_net_return']=bm[rr]
        # Sliding future extrema are diagnostics, not executable exit prices.
        future_high=f['h'].iloc[::-1].rolling(horizon,min_periods=horizon).max().iloc[::-1].shift(-1).to_numpy()
        future_low=f['l'].iloc[::-1].rolling(horizon,min_periods=horizon).min().iloc[::-1].shift(-1).to_numpy()
        entry=np.full(len(events),np.nan);known=rr+1<n
        entry[known]=open_adj[rr[known]+1,cc[known]]
        part['mfe']=np.where(part.complete,future_high[rr,cc]/entry-1,np.nan)
        part['mae']=np.where(part.complete,future_low[rr,cc]/entry-1,np.nan)
        parts.append(part)
    return pd.concat(parts,ignore_index=True)


def run(args):
    started=time.perf_counter()
    output=args.output.resolve()
    output.relative_to(ROOT)
    if output.exists():raise ValueError('Use a new output directory to preserve prior trials')
    output.mkdir(parents=True)
    sources={}
    paths=[Path(__file__),ROOT/'skills/rally_context_features.py',ROOT/'skills/rally_context_stats.py',
           ROOT/'scripts/study_rally_precursors.py',ROOT/'scripts/scan_market_strategies.py',
           *sorted((ROOT/'skills/strategy_scanner').glob('*.py')),
           ROOT/'docs/prereg_rally_context_20261007.md',ROOT/'docs/rally_context_sources_20261007.json']
    for p in paths:sources[str(p.relative_to(ROOT))]=digest(p)
    manifest=json.loads((ROOT/'docs/rally_context_sources_20261007.json').read_text())
    external={}
    for kind in ('flow','sector'):
        source=manifest['sources'][kind]
        external[kind]=verified_json(source['features'],sources)
        old=verified_json(source['report'],sources)
        # Verify the preserved direct closure, without importing its old labels.
        for key in ('source_sha256','output_sha256'):
            for name,expected in old.get(key,{}).items():
                p=(ROOT/source['report']['path']).parent/name if key=='output_sha256' and '/' not in name else ROOT/name
                if digest(p)!=expected:raise ValueError('Context ancestor changed: '+name)
                sources[str(p.relative_to(ROOT))]=expected
    data=load_bundle(args.bundle,args.start,args.end)
    poc,poc_meta=load_poc(args.poc_report,bundle=args.bundle,
        manifest_hash=data['provenance']['source_hashes']['manifest.json'])
    events,f,days,ids=build_signal_features(data['bars'],data['calendar'],start=args.start,end=args.end,
        original_signals=data['original_signals'],poc=poc,provenance=data['provenance'])
    events=attach_context(events,external['flow'],external['sector'],days)
    feature_path=output/'signal-features.parquet'
    events.to_parquet(feature_path,index=False)
    features_sha=digest(feature_path)
    # Outcome-free features have been saved before evaluating any future path.
    labelled=fixed_outcomes(events,f,days,ids)
    labelled_path=output/'labelled-events.parquet'
    labelled.to_parquet(labelled_path,index=False)
    result=summarize_filters(labelled,FILTERS)
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:raise ValueError('Source changed during study: '+name)
    if digest(feature_path)!=features_sha:raise ValueError('Outcome-free features changed')
    report=dict(schema='rally_context_study_v1',start=args.start,end=args.end,
        generated_at=datetime.now(timezone.utc).isoformat(),cohorts=list(COHORTS),filters=list(FILTERS),
        cost_model=COSTS,source_provenance=data['provenance'],poc_provenance=poc_meta,
        source_sha256=sources,features=dict(path=str(feature_path.relative_to(ROOT)),sha256=features_sha),
        labelled_events=dict(path=str(labelled_path.relative_to(ROOT)),sha256=digest(labelled_path)),
        feature_event_counts=events.groupby('cohort').size().to_dict(),result=result,
        hypothesis_count=len(COHORTS)*len(FILTERS)*2,elapsed_seconds=time.perf_counter()-started,
        account_backtest=False,live_qualified=False,unseen_validation=False,
        historical_period_already_researched=True,multiple_testing_adjusted=False,
        network_requests=0,finmind_requests=0,database_mutations=False,
        limitations=['Descriptive predeclared context comparisons, not causal effects or unseen confirmation.',
          'Overlapping stock paths and common market/peer shocks; no independent-trial significance claim.',
          'Known subsets differ per filter; unknown, immature and missing-path events remain explicit.',
          'Price peers are not historical themes; turnover shares are not net capital inflows.',
          'Flow lag1/lag3 availability is assumed, not verified historical first publication.',
          'No complete news/holder-publication archive; these families remain untested.',
          'Signal returns use next adjusted open/fixed close proxies, not capacity or account fills.',
          'Scanner direct inputs checked; complete historical market universe and entire ancestor closure uncertified.'])
    p=output/'report.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    p.with_suffix('.sha256').write_text(digest(p)+'\n')
    # One record per hypothesis, with all slices in the report, no winner-only logging.
    for cohort in COHORTS:
        for horizon in (20,60):
            for condition in FILTERS:
                append_trial_registry(dict(timestamp=report['generated_at'],source='rally_context_20261007',
                    study_type='descriptive_signal_context',start=args.start,end=args.end,
                    params=dict(cohort=cohort,horizon=horizon,condition=condition),
                    result_path=str(p.relative_to(ROOT)),report_sha256=digest(p),
                    account_backtest=False,live_qualified=False,unseen_validation=False))
    print(json.dumps(dict(output=str(output),events=len(events),hypotheses=report['hypothesis_count'],
        elapsed_seconds=report['elapsed_seconds'],finmind_requests=0),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'.cache/scanner-20261006/inputs-v1')
    parser.add_argument('--poc-report',type=Path,default=ROOT/'.cache/scanner-20261006/poc-complete-v2/report.json')
    parser.add_argument('--start',default='2024-01-02')
    parser.add_argument('--end',default='2026-10-05')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
