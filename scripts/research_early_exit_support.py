#!/usr/bin/env python3
"""One preregistered exit comparison and retrospective post-exit paths."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from skills.independent_three_black import ThreeBlackPath, black_at_close, path_issue
from skills.independent_signals import net_unit_return
from skills.ordinary_volume_bundle import bind, digest
from skills.trial_registry import append_trial_registry

PERIODS = {'2019-2022': (2019, 2022), '2023-2024': (2023, 2024), '2025-2026': (2025, 2026)}
STATUSES = ('closed', 'open', 'pending_exit', 'unresolved', 'not_entered')


def simulate(path, entry, support=None):
    """Scalar research-only exit loop; support is fixed at the signal close."""
    path.validate()
    if type(entry) is not int or not 1 <= entry < len(path.days):
        raise ValueError('Entry must follow a signal session')
    if support is not None and (not math.isfinite(support) or support <= 0):
        raise ValueError('Fixed positive signal-time support required')
    last, anchor = len(path.days)-1, path.close[entry]
    reason = trigger = None
    end = last
    for j in range(entry, min(entry+62, last)+1):
        current = path.close[j]
        known = math.isfinite(current) and current > 0 and math.isfinite(anchor) and anchor > 0
        if known and current/anchor-1 <= -.12+1e-12:
            reason = 'loss12'
        elif j+1-entry >= 63:
            reason = 'time63'
        elif black_at_close(path, entry, j) and (support is None or current < support):
            reason = 'three_black' if support is None else 'three_black_below_signal_support'
        if reason:
            trigger, end = j, min(j+1, last)
            break
    closed = trigger is not None and trigger < last
    issue = path_issue(path, entry, end)
    if issue is None and closed and not path.volume[end] > 0:
        issue = 'no_volume_on_assumed_exit'
    if issue is None and not (math.isfinite(path.close[entry-1]) and path.close[entry-1] > 0):
        issue = 'missing_pre_entry_close'
    date = lambda i: str(path.days[i].date())
    result = dict(status='unresolved', outcome='unknown', exit_reason=None,
        exit_trigger_date=None, exit_date=None, holding_days=None,
        entry_price=None, exit_price=None, adjusted_entry_price=None,
        adjusted_end_price=None, gross_return=None, net_return=None,
        unrealized_net_return=None, data_issue_code=issue, observed_end_date=date(end))
    if issue:
        return result
    raw_entry = float((path.high[entry]+path.low[entry])/2)
    adjusted_entry = raw_entry*path.close[entry]/path.raw_close[entry]
    raw_end = float((path.high[end]+path.low[end])/2) if closed else float(path.raw_close[end])
    adjusted_end = raw_end*path.close[end]/path.raw_close[end]
    gross = adjusted_end/adjusted_entry-1
    net = net_unit_return(adjusted_end/adjusted_entry)
    result.update(status='closed' if closed else 'pending_exit' if trigger is not None else 'open',
        outcome=('profit' if net > 1e-12 else 'loss' if net < -1e-12 else 'flat') if closed else 'unrealized',
        exit_reason=reason, exit_trigger_date=date(trigger) if trigger is not None else None,
        exit_date=date(end) if closed else None, holding_days=end-entry,
        entry_price=raw_entry, exit_price=raw_end if closed else None,
        adjusted_entry_price=float(adjusted_entry), adjusted_end_price=float(adjusted_end),
        gross_return=float(gross), net_return=float(net) if closed else None,
        unrealized_net_return=float(net) if not closed else None)
    return result


def metrics(rows):
    closed = [r for r in rows if r['status'] == 'closed']
    returns = np.array([r['net_return'] for r in closed], dtype=float)
    early = [r for r in closed if r['holding_days'] <= 5 and r['net_return'] < 0]
    n = len(closed)
    worst_n = math.ceil(n*.05) if n else 0
    return dict(total=len(rows), status_counts={s:sum(r['status'] == s for r in rows) for s in STATUSES},
        closed=n, wins=sum(r['net_return'] > 1e-12 for r in closed),
        win_rate=float(np.mean(returns > 1e-12)) if n else None,
        mean_net_return=float(returns.mean()) if n else None,
        median_net_return=float(np.median(returns)) if n else None,
        mean_holding_days=float(np.mean([r['holding_days'] for r in closed])) if n else None,
        median_holding_days=float(np.median([r['holding_days'] for r in closed])) if n else None,
        early_loss_count=len(early), early_loss_fraction_of_closed=len(early)/n if n else None,
        worst5_count=worst_n, worst5_mean=float(np.sort(returns)[:worst_n].mean()) if n else None,
        exits=dict(Counter(r['exit_reason'] for r in closed)),
        data_issues=dict(Counter(r.get('data_issue_code') for r in rows if r['status'] == 'unresolved')))


def compare(baseline, variant):
    if [r['signal_id'] for r in baseline] != [r['signal_id'] for r in variant]:
        raise ValueError('Paired study must retain every original signal in order')
    pairs = [(a,b) for a,b in zip(baseline,variant) if a['status'] == b['status'] == 'closed']
    delta = [b['net_return']-a['net_return'] for a,b in pairs]
    return dict(baseline_all=metrics(baseline), variant_all=metrics(variant),
        paired_closed=len(pairs), paired_baseline=metrics([a for a,b in pairs]),
        paired_variant=metrics([b for a,b in pairs]),
        improved=sum(d > 1e-12 for d in delta), worsened=sum(d < -1e-12 for d in delta),
        unchanged=sum(abs(d) <= 1e-12 for d in delta),
        mean_return_change=float(np.mean(delta)) if delta else None,
        median_return_change=float(np.median(delta)) if delta else None,
        mean_holding_change=float(np.mean([b['holding_days']-a['holding_days'] for a,b in pairs])) if pairs else None,
        status_transitions=dict(Counter(a['status']+'->'+b['status'] for a,b in zip(baseline,variant))),
        baseline_closed_to_other=[dict(signal_id=a['signal_id'],stock_id=a['stock_id'],name=a['name'],
            baseline_exit_date=a['exit_date'],baseline_net_return=a['net_return'],
            new_status=b['status'],new_issue=b.get('data_issue_code'),new_unrealized_net_return=b.get('unrealized_net_return'))
            for a,b in zip(baseline,variant) if a['status']=='closed' and b['status']!='closed'])


def followup(path, row, horizon):
    """After-the-fact diagnostic; never a signal input or a hypothetical fill."""
    if row['status'] != 'closed' or horizon not in (5,20):
        raise ValueError('A closed baseline and preregistered horizon are required')
    start = int(path.days.get_loc(pd.Timestamp(row['exit_date'])))
    end = start+horizon
    result = dict(signal_id=row['signal_id'],stock_id=row['stock_id'],name=row['name'],
        signal_date=row['signal_date'],entry_date=row['entry_date'],exit_date=row['exit_date'],
        baseline_net_return=row['net_return'],horizon=horizon,status='incomplete_window',
        end_date=None,exit_to_close_return=None,entry_to_close_return=None,
        recovered_entry_within_window=None,recovered_entry_at_end=None,
        first_recovery_date=None,data_issue_code=None)
    if end >= len(path.days):
        return result
    issue = path_issue(path,start,end)
    result.update(end_date=str(path.days[end].date()),data_issue_code=issue)
    if issue:
        result['status']='unresolved'
        return result
    after=path.close[start+1:end+1]
    entry,exit_price=row['adjusted_entry_price'],row['adjusted_end_price']
    recover=np.flatnonzero(after >= entry)
    result.update(status='complete',exit_to_close_return=float(path.close[end]/exit_price-1),
        entry_to_close_return=float(path.close[end]/entry-1),
        recovered_entry_within_window=bool(len(recover)),
        recovered_entry_at_end=bool(path.close[end]>=entry),
        first_recovery_date=str(path.days[start+1+int(recover[0])].date()) if len(recover) else None)
    return result


def followup_summary(rows):
    complete=[r for r in rows if r['status']=='complete'];n=len(complete)
    return dict(total=len(rows),complete=n,incomplete_window=sum(r['status']=='incomplete_window' for r in rows),
        unresolved=sum(r['status']=='unresolved' for r in rows),
        mean_from_exit=float(np.mean([r['exit_to_close_return'] for r in complete])) if n else None,
        median_from_exit=float(np.median([r['exit_to_close_return'] for r in complete])) if n else None,
        mean_from_entry=float(np.mean([r['entry_to_close_return'] for r in complete])) if n else None,
        median_from_entry=float(np.median([r['entry_to_close_return'] for r in complete])) if n else None,
        recovered_within=sum(r['recovered_entry_within_window'] for r in complete),
        recovered_within_rate=sum(r['recovered_entry_within_window'] for r in complete)/n if n else None,
        recovered_at_end=sum(r['recovered_entry_at_end'] for r in complete),
        recovered_at_end_rate=sum(r['recovered_entry_at_end'] for r in complete)/n if n else None,
        data_issues=dict(Counter(r['data_issue_code'] for r in rows if r['status']=='unresolved')))


def run(inputs, baseline, output):
    inputs,baseline,output=[Path(p).resolve() for p in (inputs,baseline,output)]
    if any(not p.is_relative_to(ROOT) for p in (inputs,baseline,output)) or output.exists():
        raise ValueError('Use repository sources and a new output directory')
    refs={}
    def seal(p,expected=None):
        return bind(ROOT,refs,str(p.relative_to(ROOT)),expected or digest(p))
    manifest=json.loads(seal(inputs/'manifest.json',(inputs/'manifest.sha256').read_text().strip()).read_text())
    for name,h in manifest['files_sha256'].items():seal(inputs/name,h)
    for name,h in manifest['source_sha256'].items():bind(ROOT,refs,name,h)
    report=json.loads(seal(baseline/'report.json',(baseline/'report.sha256').read_text().strip()).read_text())
    for name,h in report['source_sha256'].items():bind(ROOT,refs,name,h)
    for name,h in report['output_sha256'].items():bind(ROOT,refs,name,h)
    payload=json.loads((baseline/'workbook-data.json').read_text());original=payload['rows']
    if len(original)!=30188 or payload['metadata']['data_through']!='2026-10-02':
        raise ValueError('Require the preregistered 30,188 signal study')
    ids=sorted({r['stock_id'] for r in original})
    frames={n:pd.read_parquet(inputs/(n+'.parquet'),columns=['date',*ids]).set_index('date') for n in ('close-official','close-quality','eligibility')}
    c,other,eligible=[frames[n] for n in frames];days=pd.DatetimeIndex(c.index)
    raw=pd.read_parquet(inputs/'quotes-unmasked.parquet');raw=raw[raw.stock_id.isin(ids)]
    if raw.duplicated(['date','stock_id']).any():raise ValueError('Duplicate prices')
    fields={n:raw.pivot(index='date',columns='stock_id',values=n).reindex(index=days,columns=ids) for n in ('close','high','low','volume','open')}
    paths={sid:ThreeBlackPath(days,c[sid].to_numpy(float),other[sid].to_numpy(float),eligible[sid].to_numpy(bool),
        *[fields[n][sid].to_numpy(float) for n in ('close','high','low','volume','open')]) for sid in ids}
    features=pd.read_parquet(inputs/'signal-features.parquet').set_index('event_id')
    if not features.index.is_unique:raise ValueError('Duplicate signal features')
    rebuilt=[];variant=[];subsequent=[];checked=0
    for i,row in enumerate(original):
        sid=row['stock_id'];path=paths[sid];base={k:row[k] for k in ('signal_id','stock_id','name','signal_date','entry_date')}
        if row['status']=='not_entered':
            result=dict(status='not_entered',outcome='unrealized',exit_reason=None,data_issue_code=None)
            rebuilt.append(dict(base,**result));variant.append(dict(base,**result));continue
        entry=int(days.get_loc(pd.Timestamp(row['entry_date'])));signal=entry-1
        if str(days[signal].date())!=row['signal_date']:raise ValueError('Entry differs from T+1')
        support=float(features.at[row['signal_id'],'previous60_high_adjusted'])
        prior=path.close[signal-60:signal]
        if len(prior)!=60 or not np.isfinite(prior).all() or not math.isclose(support,float(max(prior)),rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Support is not the fixed signal-time previous60 close high')
        result=simulate(path,entry)
        for key in ('status','outcome','exit_reason','exit_trigger_date','exit_date','holding_days','data_issue_code'):
            if result[key]!=row.get(key):raise ValueError('Baseline reproduction differs '+row['signal_id']+' '+key)
        for key in ('net_return','unrealized_net_return','gross_return','adjusted_entry_price','adjusted_end_price'):
            a,b=result[key],row.get(key)
            if a is None and b is None:continue
            if a is None or b is None or not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Baseline price or return differs '+row['signal_id']+' '+key)
        checked+=1
        a=dict(base,**result);b=dict(base,**simulate(path,entry,support),fixed_signal_support=support)
        rebuilt.append(a);variant.append(b)
        if a['status']=='closed' and a['holding_days']<=5 and a['net_return']<0:
            subsequent.extend(followup(path,a,n) for n in (5,20))
        if (i+1)%5000==0:print('signals',i+1,flush=True)
    results=dict(all=compare(rebuilt,variant),
        annual={str(y):compare([r for r in rebuilt if r['signal_date'].startswith(str(y))],
                              [r for r in variant if r['signal_date'].startswith(str(y))]) for y in range(2019,2027)},
        periods={label:compare([r for r in rebuilt if lo<=int(r['signal_date'][:4])<=hi],
                              [r for r in variant if lo<=int(r['signal_date'][:4])<=hi]) for label,(lo,hi) in PERIODS.items()})
    followups={str(n):dict(all=followup_summary([r for r in subsequent if r['horizon']==n]),
        annual={str(y):followup_summary([r for r in subsequent if r['horizon']==n and r['signal_date'].startswith(str(y))]) for y in range(2019,2027)},
        periods={label:followup_summary([r for r in subsequent if r['horizon']==n and lo<=int(r['signal_date'][:4])<=hi]) for label,(lo,hi) in PERIODS.items()}) for n in (5,20)}
    # Fixed examples: first chronological early-loss event in each year, plus
    # deterministic largest positive/negative paired changes (explicit extremes).
    early=sorted([r for r in rebuilt if r['status']=='closed' and r['holding_days']<=5 and r['net_return']<0],key=lambda r:(r['signal_date'],r['signal_id']))
    byid={r['signal_id']:r for r in variant}
    examples=[dict(baseline=next(r for r in early if r['signal_date'].startswith(str(y))),
        variant=byid[next(r['signal_id'] for r in early if r['signal_date'].startswith(str(y)))]) for y in range(2019,2027)]
    deltas=sorted([dict(signal_id=a['signal_id'],stock_id=a['stock_id'],name=a['name'],signal_date=a['signal_date'],
        baseline_net_return=a['net_return'],variant_net_return=b['net_return'],delta=b['net_return']-a['net_return'],
        baseline_exit=a['exit_date'],variant_exit=b['exit_date']) for a,b in zip(rebuilt,variant) if a['status']==b['status']=='closed'],key=lambda r:(r['delta'],r['signal_id']))
    for p in (Path(__file__),ROOT/'tests/test_research_early_exit_support.py',ROOT/'docs/prereg_early_signal_losses_20261002.md',ROOT/'skills/independent_three_black.py',ROOT/'skills/independent_signals.py',ROOT/'skills/trial_registry.py'):
        seal(p)
    output.mkdir(parents=True)
    pd.DataFrame(rebuilt).to_parquet(output/'baseline-rebuilt.parquet',index=False)
    pd.DataFrame(variant).to_parquet(output/'support-exit.parquet',index=False)
    pd.DataFrame(subsequent).to_parquet(output/'post-exit-followups.parquet',index=False)
    result=dict(schema='early_exit_support_research_v1',created_at=datetime.now(timezone.utc).isoformat(),
        baseline_rows_reproduced=checked,not_entered_preserved=21,baseline_closed_reproduced=sum(r['status']=='closed' for r in rebuilt),
        results=results,post_exit=followups,examples_first_early_loss_per_year=examples,
        examples_largest_improvements=list(reversed(deltas[-5:])),examples_largest_deteriorations=deltas[:5],
        definitions=dict(early_loss='closed net_return<0 and elapsed market sessions<=5',
            support='signal-time maximum adjusted close of the preceding60 market sessions; strictly below',
            worst5='mean of the worst ceil(closed_count*0.05) returns',
            recovery='at least one adjusted close within next N sessions >= original adjusted HL2 entry; excludes exit day; before trading costs',
            followup='complete quality-valid exit-to-end path; closing price ratios vs adjusted exit HL2 and entry HL2; not executable returns'),
        caveats=payload['metadata']['limitations']+['Post-exit followups are hindsight descriptions, never used by entry/exit rules.',
            'Paired results exclude unresolved/open comparisons; all status changes and all-arm denominators separately reported.',
            'The three historical periods are already researched, not unseen validation.'],
        source_sha256=refs,output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        cash_account=False,live_qualified=False,actual_fill_verified=False,production_strategy_changed=False)
    p=output/'report.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');p.with_suffix('.sha256').write_text(digest(p)+'\n')
    print(json.dumps(dict(comparison=results['all'],post_exit={k:v['all'] for k,v in followups.items()}),ensure_ascii=False,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/inputs')
    parser.add_argument('--baseline',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/research-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/early-signal-losses-20261002/exits')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve completed or failed attempts; choose a new output')
    started=datetime.now(timezone.utc).isoformat();success=False;result=None;error=None
    try:
        result=run(args.inputs,args.baseline,args.output);success=True
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        # Two arms are counted even on an incomplete attempt; neither results
        # nor adverse outcomes can disappear from the trial denominator.
        records=[]
        for arm in ('baseline_three_black','three_black_below_signal_support'):
            summary=(result['results']['all']['baseline_all' if arm=='baseline_three_black' else 'variant_all']
                     if success else None)
            record=dict(source='early_exit_support',timestamp=started,command=' '.join(sys.argv),arm=arm,
                completed=success,error=error,summary=summary,sharpe=None,cash_account=False,
                unit_of_analysis='independent_signal_not_portfolio',output=str(args.output.relative_to(ROOT)),
                live_qualified=False,unseen_validation=False)
            record['registry_line']=append_trial_registry(record)
            records.append(record)
        args.output.mkdir(parents=True,exist_ok=True)
        p=args.output/'trials.json';p.write_text(json.dumps(dict(actual_run_attempts=1,arm_evaluations=2,
            source='early_exit_support',records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        p.with_suffix('.sha256').write_text(digest(p)+'\n')


if __name__=='__main__':
    main()
