#!/usr/bin/env python3
"""Describe sealed trade outcomes; no new orders, parameter search or strategy."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha


def metrics(frame):
    closed = frame[frame.status.eq('closed')]
    wins = closed[closed.net_return > 0]
    losses = closed[closed.net_return < 0]
    return dict(total=len(frame), stocks=int(frame.stock_id.nunique()), closed=len(closed),
                winners=len(wins), losers=len(losses), unknown=int(frame.status.eq('unknown').sum()),
                unclosed=int(frame.status.isin(['open','pending_exit']).sum()),
                win_rate=float(len(wins)/len(closed)) if len(closed) else None,
                mean=float(closed.net_return.mean()) if len(closed) else None,
                median=float(closed.net_return.median()) if len(closed) else None,
                mean_win=float(wins.net_return.mean()) if len(wins) else None,
                mean_loss=float(losses.net_return.mean()) if len(losses) else None,
                net_winners50=int((closed.net_return >= .5).sum()),
                net_winners100=int((closed.net_return >= 1).sum()))


def grouped(frame, column, bins):
    groups = pd.cut(frame[column], bins, right=False)
    return {str(key): metrics(group) for key,group in frame.groupby(groups,observed=True)}


def diagnose(output):
    if output.exists():
        raise ValueError('Choose a new output path; preserve prior diagnostics')
    source = ROOT/'artifacts/forward_simulation/independent_signals_20261001'
    report = read(source/'report.json')
    refs = {}
    def bind(path, expected=None):
        h = sha(path)
        if expected is not None and h != expected:
            raise ValueError('Sealed source changed: '+str(path))
        refs[str(path.relative_to(ROOT))] = h
    bind(source/'report.json')
    bind(source/'signals.parquet',report['exports_sha256']['signals.parquet'])
    inputs = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    prices = inputs/'close-official.parquet'
    bind(prices,report['source_sha256'][str(prices.relative_to(ROOT))])
    signals = ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json'
    bind(signals,report['source_sha256'][str(signals.relative_to(ROOT))])
    bind(Path(__file__))
    p = pd.read_parquet(source/'signals.parquet')
    close = pd.read_parquet(prices).set_index('date')
    days = close.index
    index = {str(d.date()):i for i,d in enumerate(days)}
    if len(p)!=report['all_signals']['signals'] or not p.event_id.is_unique:
        raise ValueError('Signal population changed')
    p['mature'] = p.entry_date.map(index) + 63 < len(days)
    p['year'] = p.signal_date.str[:4]
    p['month'] = p.signal_date.str[:7]
    m = p[p.mature & p.status.eq('closed')].copy()
    ma = close.rolling(20,min_periods=20).mean()
    m['extension'] = [close.at[pd.Timestamp(d),s]/ma.at[pd.Timestamp(d),s]-1
                      for d,s in zip(m.signal_date,m.stock_id)]
    entries = read(signals)['entries']['liquid_universe']
    metadata = {e['event_id']:e for e in entries}
    m['return20'] = [metadata[e]['leader_evidence']['leader_return20'] for e in m.event_id]
    recent, prior = {}, {}
    for e in sorted(entries,key=lambda e:(e['signal_date'],e['event_id'])):
        sid, i = e['members'][0], index[e['signal_date']]
        q = [j for j in recent.get(sid,[]) if j >= i-20]
        prior[e['event_id']] = len(q)
        recent[sid] = q+[i]
    m['prior_signals20'] = m.event_id.map(prior)
    if not np.isfinite(m[['extension','return20','volume_ratio_at_signal','prior_signals20']]).all().all():
        raise ValueError('Diagnostic feature missing; do not silently drop a group')
    # A chronological same-stock subset, not a cash allocation rerun. Unknown
    # holdings block later same-stock signals to avoid guessing their exit.
    kept, blocked = [], {}
    for row in p.sort_values(['entry_date','event_id']).to_dict('records'):
        sid, i = row['stock_id'], index[row['entry_date']]
        if i <= blocked.get(sid,-1):
            continue
        kept.append(row)
        blocked[sid] = index[row['exit_date']] if row['status']=='closed' else len(days)-1
    distinct = pd.DataFrame(kept)
    losses, wins = m[m.net_return<0], m[m.net_return>0]
    peak = pd.cut(losses.peak_close_return,[-np.inf,0,.05,.1,.2,np.inf],right=False)
    bins = {str(k):dict(count=len(g),fraction=len(g)/len(losses),mean_return=float(g.net_return.mean()))
            for k,g in losses.groupby(peak,observed=True)}
    assert sum(g['count'] for g in bins.values())==len(losses)
    concentration=[]
    ordered=m.sort_values('net_return',ascending=False)
    for fraction in (.01,.05):
        n=int(np.ceil(len(m)*fraction))
        concentration.append(dict(fraction=fraction,count=n,
            share_of_positive_unit_returns=float(ordered.iloc[:n].net_return.sum()/wins.net_return.sum()),
            remaining_mean=float(ordered.iloc[n:].net_return.mean()),
            interpretation='Retrospective concentration only; not a tradable exclusion rule'))
    metrics_m = metrics(m)
    result=dict(source_sha256=refs,data_end=str(days[-1].date()),
        full_window_last_entry=str(days[-64].date()),all_signals=metrics(p),
        mature_population=metrics(p[p.mature]),unmature_population=metrics(p[~p.mature]),
        first_per_stock_mature=metrics(p[p.mature & p.first_signal_for_stock]),
        nonoverlap_mature=metrics(distinct[distinct.mature]),
        loss_peak_groups=bins,
        mature_exit_reasons={k:metrics(g) for k,g in m.groupby('reason')},
        mature_years={k:metrics(g) for k,g in p[p.mature].groupby('year')},
        mature_months={k:metrics(g) for k,g in p[p.mature].groupby('month')},
        extension=grouped(m,'extension',[-np.inf,.1,.2,np.inf]),
        extension_years={y:grouped(g,'extension',[-np.inf,.1,.2,np.inf]) for y,g in m.groupby('year')},
        prior20_return=grouped(m,'return20',[0,.1,.2,.4,np.inf]),
        volume=grouped(m,'volume_ratio_at_signal',[1.5,2,3,5,np.inf]),
        volume_years={y:grouped(g,'volume_ratio_at_signal',[1.5,2,3,5,np.inf]) for y,g in m.groupby('year')},
        prior20_signals=grouped(m,'prior_signals20',[-.1,.1,3.1,10.1,np.inf]),
        gross_negative=int((m.gross_return<0).sum()),
        nonnegative_gross_to_net_loss=int((m.gross_return.ge(0)&m.net_return.lt(0)).sum()),
        equal_unit_profit_factor=float(wins.net_return.sum()/-losses.net_return.sum()),
        retrospective_break_even_win_rate=-metrics_m['mean_loss']/(metrics_m['mean_win']-metrics_m['mean_loss']),
        concentration=concentration,
        loss_exit_clusters=[dict(date=d,signals=len(g),stocks=int(g.stock_id.nunique()))
                            for d,g in losses.groupby('exit_date')],
        descriptive_only=True,parameters_changed=False,new_backtest=False,
        independence_assumed=False,live_qualified=False,unseen_validation=False)
    assert result['mature_population']['unclosed']==0
    assert result['mature_population']['closed']+result['mature_population']['unknown']==result['mature_population']['total']
    assert metrics_m['winners']+metrics_m['losers']==len(m)
    assert abs(metrics_m['mean'] - (metrics_m['win_rate']*metrics_m['mean_win'] +
               (1-metrics_m['win_rate'])*metrics_m['mean_loss']))<1e-12
    for path,h in refs.items():
        if sha(ROOT/path)!=h:
            raise ValueError('Source changed during diagnosis')
    write(output,result)
    print(json.dumps({k:result[k] for k in ['full_window_last_entry','mature_population',
          'nonoverlap_mature','loss_peak_groups','extension','concentration']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.relative_to(ROOT)
    diagnose(output)
