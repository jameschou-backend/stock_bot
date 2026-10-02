#!/usr/bin/env python3
"""Frozen-signal entry-distance study; all fills remain daily-OHLC proxies."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from scripts.research_early_signal_losses import verified_sources, statistics, evaluate_filter, PERIODS, digest
from skills.independent_signals import net_unit_return
from skills.independent_three_black import ThreeBlackPath
from skills.intraday_limit_replay import limit_price
from skills.trial_registry import append_trial_registry

ARMS=('hl2_baseline','signal_distance5','open_control','preplaced_limit5')


def signal_distance(path, signal):
    """Only the completed signal close and its preceding60 closes are read."""
    prior=path.close[max(0,signal-60):signal]
    current=float(path.close[signal])
    result=dict(support=None,signal_distance=None,signal_distance5=None,signal_feature_issue=None,
                t0_limit_raw=None,t0_adjustment_factor=None)
    if len(prior)!=60 or not (np.isfinite(prior)&(prior>0)).all() or not math.isfinite(current) or current<=0:
        return dict(result,signal_feature_issue='missing_signal_or_prior60_adjusted_prices')
    support=float(max(prior));distance=current/support-1
    result.update(support=support,signal_distance=distance,signal_distance5=bool(current/support<=1.05))
    raw=float(path.raw_close[signal])
    if not math.isfinite(raw) or raw<=0:
        return dict(result,signal_feature_issue='missing_signal_raw_close')
    factor=current/raw
    # Tick rounding is an order instruction made at T0, never from T1 prices.
    result.update(t0_limit_raw=limit_price(support/factor*1.05,'1234','buy'),t0_adjustment_factor=factor)
    return result


def entry_proxy(path, entry, features, arm):
    """Assess the fixed T0 instruction after observing T1; no queue inference."""
    if arm not in ('open_control','preplaced_limit5'):
        raise ValueError('Unknown entry proxy')
    if type(entry) is not int or not 1<=entry<len(path.days):
        raise ValueError('Entry must follow a completed signal')
    result=dict(entry_proxy_status='unresolved',entry_proxy_price=None,entry_proxy_issue=None,
                order_limit=features['t0_limit_raw'],t1_adjustment_factor=None,
                instruction_known_at=str(path.days[entry-1].date())+' 收盤後、次日交易前',
                execution_evidence_known_at=str(path.days[entry].date())+' 全日行情完成後')
    if features['signal_feature_issue'] or features['t0_limit_raw'] is None:
        return dict(result,entry_proxy_issue=features['signal_feature_issue'] or 'unknown_fixed_limit')
    if not path.eligible[entry-1:entry+1].all():
        return dict(result,entry_proxy_issue='signal_or_entry_identity_unresolved')
    t0=[float(a[entry-1]) for a in (path.opened,path.high,path.low,path.raw_close,path.volume)]
    if not all(math.isfinite(x) and x>0 for x in t0):
        return dict(result,entry_proxy_issue='missing_or_nonpositive_signal_ohlcv')
    op,hi,lo,cl,_=t0
    if not lo<=min(op,cl)<=max(op,cl)<=hi:
        return dict(result,entry_proxy_issue='impossible_signal_ohlc')
    opened,high,low,closed,volume=[float(a[entry]) for a in (path.opened,path.high,path.low,path.raw_close,path.volume)]
    if not all(math.isfinite(x) and x>0 for x in (opened,high,low,closed,volume)):
        return dict(result,entry_proxy_issue='missing_or_nonpositive_entry_ohlcv')
    if not low<=min(opened,closed)<=max(opened,closed)<=high:
        return dict(result,entry_proxy_issue='impossible_entry_ohlc')
    if high==low:
        return dict(result,entry_proxy_issue='single_price_session_without_queue_evidence')
    adjusted=float(path.close[entry])
    if not math.isfinite(adjusted) or adjusted<=0:
        return dict(result,entry_proxy_issue='missing_entry_adjusted_close')
    c,other=path.close[entry-1:entry+1],path.other[entry-1:entry+1]
    if not all((np.isfinite(v)&(v>0)).all() for v in (c,other)):
        return dict(result,entry_proxy_issue='missing_signal_entry_adjustment_reference')
    first,second=float(c[1]/c[0]-1),float(other[1]/other[0]-1)
    if abs(first)>.20 or abs(second)>.20 or abs(first-second)>.005:
        return dict(result,entry_proxy_issue='signal_entry_adjustment_reference_conflict')
    factor=adjusted/closed;result['t1_adjustment_factor']=factor
    if not math.isclose(factor,features['t0_adjustment_factor'],rel_tol=1e-10,abs_tol=1e-12):
        return dict(result,entry_proxy_issue='t0_t1_adjustment_factor_changed')
    if arm=='open_control':
        return dict(result,entry_proxy_status='filled_proxy',entry_proxy_price=opened)
    limit=features['t0_limit_raw']
    if opened<=limit:
        return dict(result,entry_proxy_status='filled_proxy',entry_proxy_price=opened)
    if low<limit:
        return dict(result,entry_proxy_status='filled_proxy',entry_proxy_price=limit)
    return dict(result,entry_proxy_status='not_filled',entry_proxy_issue='never_strictly_traded_below_preplaced_limit')


def blank_execution(row,status,issue):
    result=dict(row,status=status,outcome='unknown' if status=='unresolved' else 'not_invested',
        data_issue_code=issue,entry_date=None,modeled_fill_date=None)
    for key in ('entry_price','exit_price','mark_price','adjusted_entry_price','adjusted_end_price',
                'gross_return','net_return','unrealized_net_return','holding_days','holding_days_inclusive',
                'calendar_days','exit_reason','exit_trigger_date','exit_date','mfe','mfe_date',
                'peak_close_return','peak_close_date','confirmed_high_return','confirmed_high_date'):
        result[key]=None
    return result


def execution_row(original,path,entry,features,arm):
    """Keep original exit dates/close anchor; alter only modeled entry price."""
    decision=entry_proxy(path,entry,features,arm)
    result=dict(original,original_status=original['status'],planned_entry_date=original['entry_date'],
        original_entry_date=original['entry_date'],modeled_fill_date=original['entry_date'] if decision['entry_proxy_status']=='filled_proxy' else None,
        **features,**decision)
    if decision['entry_proxy_status']!='filled_proxy':
        return blank_execution(result,decision['entry_proxy_status'],decision['entry_proxy_issue'])
    # A known no-fill above is preserved even if the unused later path is bad.
    # A filled proxy cannot erase that path's unresolved execution/exit evidence.
    if original['status']=='unresolved':
        return result
    if original['status'] not in ('closed','open','pending_exit'):
        raise ValueError('Unexpected original entered status')
    raw_entry=decision['entry_proxy_price'];adjusted_entry=raw_entry*decision['t1_adjustment_factor']
    old_entry=original['adjusted_entry_price'];ratio=original['adjusted_end_price']/adjusted_entry
    net=float(net_unit_return(ratio))
    result.update(entry_price=float(raw_entry),adjusted_entry_price=float(adjusted_entry),gross_return=float(ratio-1),
        net_return=net if original['status']=='closed' else None,
        unrealized_net_return=net if original['status']!='closed' else None,
        outcome=('profit' if net>1e-12 else 'loss' if net< -1e-12 else 'flat') if original['status']=='closed' else 'unrealized')
    for key in ('mfe','peak_close_return','confirmed_high_return'):
        result[key]=(1+original[key])*old_entry/adjusted_entry-1 if original.get(key) is not None else None
    return result


def paired_comparison(control,variant):
    if [r['signal_id'] for r in control]!=[r['signal_id'] for r in variant]:
        raise ValueError('Every signal must remain in order')
    pairs=[(a,b) for a,b in zip(control,variant) if a['status']==b['status']=='closed']
    deltas=[b['net_return']-a['net_return'] for a,b in pairs]
    return dict(paired_closed=len(pairs),control=statistics([a for a,b in pairs]),variant=statistics([b for a,b in pairs]),
        improved=sum(d>1e-12 for d in deltas),worsened=sum(d< -1e-12 for d in deltas),
        unchanged=sum(abs(d)<=1e-12 for d in deltas),mean_difference=float(np.mean(deltas)) if deltas else None,
        median_difference=float(np.median(deltas)) if deltas else None,
        status_transitions=dict(Counter(a['status']+'->'+b['status'] for a,b in zip(control,variant))))


def opportunity_comparison(baseline,control,variant):
    """Common original-closed, valid-entry events; known unfilled stays cash0."""
    if not len(baseline)==len(control)==len(variant):raise ValueError('Different event populations')
    eligible=[];unknown=[]
    for b,c,v in zip(baseline,control,variant):
        if b['signal_id']!=c['signal_id'] or b['signal_id']!=v['signal_id']:raise ValueError('Event mismatch')
        if b['status']!='closed':continue
        if c['status']=='closed' and v['status'] in ('closed','not_filled'):
            eligible.append((b,c,v))
        else:unknown.append(dict(signal_id=b['signal_id'],control_status=c['status'],variant_status=v['status'],
            control_issue=c.get('entry_proxy_issue'),variant_issue=v.get('entry_proxy_issue')))
    values=[v['net_return'] if v['status']=='closed' else 0. for b,c,v in eligible]
    control_values=[c['net_return'] for b,c,v in eligible]
    return dict(original_closed=sum(b['status']=='closed' for b in baseline),common_known_events=len(eligible),
        filled=sum(v['status']=='closed' for b,c,v in eligible),known_not_filled=sum(v['status']=='not_filled' for b,c,v in eligible),
        excluded_unknown=len(unknown),unknown_events=unknown,
        control_mean=float(np.mean(control_values)) if eligible else None,
        variant_mean_with_known_unfilled_cash_zero=float(np.mean(values)) if eligible else None,
        mean_difference=float(np.mean(np.array(values)-control_values)) if eligible else None,
        definition='Equal independent original-closed known-entry event budgets; known no-fill0, unknown excluded explicitly; not account return')


def retention(baseline,arm):
    targets={'winners':[r['signal_id'] for r in baseline if r['status']=='closed' and r['net_return']>0],
             'net30':[r['signal_id'] for r in baseline if r['status']=='closed' and r['net_return']>=.3]}
    byid={r['signal_id']:r for r in arm};result={}
    for name,ids in targets.items():
        rows=[byid[eid] for eid in ids];filled=sum(r.get('entry_proxy_status')=='filled_proxy' and r['status']=='closed' for r in rows)
        result[name]=dict(original=len(ids),filled_and_valid_closed=filled,fill_retention=filled/len(ids) if ids else None,
            known_not_filled=sum(r['status']=='not_filled' for r in rows),unknown=sum(r['status']=='unresolved' for r in rows),
            still_net_positive=sum(r['status']=='closed' and r['net_return']>0 for r in rows),
            still_net30=sum(r['status']=='closed' and r['net_return']>=.3 for r in rows))
    return result


def scoped_results(baseline,open_rows,limit_rows):
    screen=evaluate_filter(baseline,'signal_distance5')
    known=[r for r in baseline if r['status']=='closed' and r['signal_distance5'] is not None]
    screen['equal_event_opportunity']=dict(known_original_closed=len(known),
        unknown_original_closed=sum(r['status']=='closed' and r['signal_distance5'] is None for r in baseline),
        baseline_mean=float(np.mean([r['net_return'] for r in known])) if known else None,
        filtered_mean_with_rejected_cash_zero=float(np.mean([r['net_return'] if r['signal_distance5'] else 0. for r in known])) if known else None,
        definition='Fixed original-event denominator; rejected event budget held as0; not funded portfolio return')
    return dict(hl2_baseline=statistics(baseline),signal_distance5=screen,
        open_control=statistics(open_rows),preplaced_limit5=statistics(limit_rows),
        open_vs_hl2=paired_comparison(baseline,open_rows),limit_vs_open=paired_comparison(open_rows,limit_rows),
        limit_opportunity_vs_open=opportunity_comparison(baseline,open_rows,limit_rows),
        open_retention=retention(baseline,open_rows),limit_retention=retention(baseline,limit_rows),
        execution_issues={name:dict(Counter(r.get('entry_proxy_issue') for r in rows if r.get('entry_proxy_status')=='unresolved')) for name,rows in [('open_control',open_rows),('preplaced_limit5',limit_rows)]})


def run(inputs,research,output,prereg,prereg_sha):
    inputs,research,output,prereg=[Path(p).resolve() for p in (inputs,research,output,prereg)]
    if output.exists() or any(not p.is_relative_to(ROOT) for p in (inputs,research,output,prereg)):
        raise ValueError('Use sealed repository sources and a new output directory')
    if not prereg.is_file() or digest(prereg)!=prereg_sha:raise ValueError('Preregistration is absent or changed')
    refs,manifest,report=verified_sources(inputs,research)
    original=json.loads((research/'workbook-data.json').read_text())['rows']
    if len(original)!=30188 or report['metadata']['data_through']!='2026-10-02':raise ValueError('Wrong frozen study')
    ids=sorted({r['stock_id'] for r in original});frames={n:pd.read_parquet(inputs/(n+'.parquet'),columns=['date',*ids]).set_index('date') for n in ('close-official','close-quality','eligibility')}
    c,other,eligible=[frames[n] for n in frames];days=pd.DatetimeIndex(c.index)
    raw=pd.read_parquet(inputs/'quotes-unmasked.parquet');raw=raw[raw.stock_id.isin(ids)]
    if raw.duplicated(['date','stock_id']).any():raise ValueError('Duplicate raw prices')
    mats={n:raw.pivot(index='date',columns='stock_id',values=n).reindex(index=days,columns=ids) for n in ('close','high','low','volume','open')}
    paths={sid:ThreeBlackPath(days,c[sid].to_numpy(float),other[sid].to_numpy(float),eligible[sid].to_numpy(bool),*[mats[n][sid].to_numpy(float) for n in ('close','high','low','volume','open')]) for sid in ids}
    frozen_features=pd.read_parquet(inputs/'signal-features.parquet').set_index('event_id')
    baseline=[];opens=[];limits=[]
    for row in original:
        path=paths[row['stock_id']];signal=int(days.get_loc(pd.Timestamp(row['signal_date'])));features=signal_distance(path,signal)
        support=float(frozen_features.at[row['signal_id'],'previous60_high_adjusted'])
        if features['support'] is None or not math.isclose(support,features['support'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Causal support differs from sealed signal features')
        baseline.append(dict(row,**features))
        if row['entry_date'] is None:
            if signal!=len(days)-1 or row['status']!='not_entered':raise ValueError('Unexpected missing entry date')
            pending=dict(row,**features,planned_entry_date=None,original_entry_date=None,modeled_fill_date=None,
                entry_proxy_status='not_entered',entry_proxy_price=None,entry_proxy_issue=None)
            opens.append(pending);limits.append(dict(pending));continue
        entry=int(days.get_loc(pd.Timestamp(row['entry_date'])))
        if entry!=signal+1:raise ValueError('Original entry must be T+1')
        opens.append(execution_row(row,path,entry,features,'open_control'))
        limits.append(execution_row(row,path,entry,features,'preplaced_limit5'))
    scopes={'all':list(range(len(original)))}
    scopes.update({str(y):[i for i,r in enumerate(original) if r['signal_date'].startswith(str(y))] for y in range(2019,2027)})
    scopes.update({name:[i for i,r in enumerate(original) if lo<=r['signal_date']<=hi] for name,(lo,hi) in PERIODS.items()})
    results={name:scoped_results([baseline[i] for i in indexes],[opens[i] for i in indexes],[limits[i] for i in indexes]) for name,indexes in scopes.items()}
    for p in (Path(__file__),ROOT/'tests/test_entry_distance_research.py',prereg,ROOT/'scripts/research_early_signal_losses.py',ROOT/'skills/intraday_limit_replay.py',ROOT/'skills/trial_registry.py'):
        refs[str(p.relative_to(ROOT))]=digest(p)
    output.mkdir(parents=True)
    for name,rows in [('hl2-signals',baseline),('open-control',opens),('preplaced-limit',limits)]:pd.DataFrame(rows).to_parquet(output/(name+'.parquet'),index=False)
    stock_rows=[];large_misses=[]
    for arm,rows in [('hl2_baseline',baseline),('signal_distance5',[r for r in baseline if r['signal_distance5'] is True]),('open_control',opens),('preplaced_limit5',limits)]:
        for sid in ids:
            group=[r for r in rows if r['stock_id']==sid];closed=[r for r in group if r['status']=='closed']
            stock_rows.append(dict(arm=arm,stock_id=sid,signals=len(group),closed=len(closed),
                sum_independent_unit_net_returns=float(sum(r['net_return'] for r in closed)),
                mean_net_return=float(np.mean([r['net_return'] for r in closed])) if closed else None,
                note='Sum is an overlapping unit-signal diagnostic, not portfolio contribution'))
    for b,o,l in zip(baseline,opens,limits):
        if b['status']!='closed' or b['net_return']<.30:continue
        for arm,row,retained in [('signal_distance5',b,b['signal_distance5'] is True),('open_control',o,o['status']=='closed'),('preplaced_limit5',l,l['status']=='closed')]:
            if retained:continue
            large_misses.append(dict(signal_id=b['signal_id'],stock_id=b['stock_id'],name=b['name'],signal_date=b['signal_date'],
                arm=arm,baseline_net_return=b['net_return'],original_exit=b['exit_date'],
                decision=('excluded' if b['signal_distance5'] is False else 'unknown') if arm=='signal_distance5' else row['status'],
                issue=row.get('entry_proxy_issue'),signal_distance=b['signal_distance']))
    pd.DataFrame(stock_rows).to_parquet(output/'stock-diagnostics.parquet',index=False)
    pd.DataFrame(large_misses).to_parquet(output/'missed-large-signals.parquet',index=False)
    result=dict(schema='entry_distance_fixed_research_v1',created_at=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        prereg_sha256=prereg_sha,sample_count=len(original),results=results,source_sha256=refs,
        definitions=dict(signal_screen='T0 adjusted close / maximum adjusted close of prior60 sessions <=1.05; unchanged T+1HL2',
            order_limit='T0 fixed support / T0 adjustment factor *1.05, floor ordinary-stock tick; T0 placed, expires T1',
            open='All T0 signals request T1 opening quote proxy; no price filter',
            limit='After shared guards: open<=limit uses open proxy; otherwise low<limit uses limit proxy; low==limit no fill',
            common_entry_guards='T0/T1 eligible, finite positive possible OHLCV; T1high>low; paired adjusted-source transition quality; T0->T1 adjustment factor isclose rel1e-10 abs1e-12',
            exits='Original threeblack/loss12/time63 dates and priorities, entry-day adjusted-close anchor unchanged; never reset by entry price',
            opportunities='Same original closed known-entry event budgets. Filled unit return; known no-fill0. Unknown listed, not0.'),
        limitations=report['metadata']['limitations']+['Opening/limit prices are daily-OHLC hypothetical execution proxies, not queue-verified or volume-capacity-certified fills.',
            'Signal extension screen does not guarantee actual next-day entry distance; no next-day price is used to decide that screen.',
            'Same-day guards diagnose missing/conflicting execution evidence after an instruction; they are not hindsight stock-selection rules.',
            'Corporate adjustment change between signal and entry is unresolved for both open and limit; no guessed raw-price conversion.',
            'These already researched dates are not unseen validation. No new threshold search or combinations.',
            'Known no-fill is assessed before later path quality; a later unused data gap must not erase a genuine known no-fill.'],
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        live_qualified=False,actual_fill_verified=False,unseen_validation=False,cash_account=False,production_strategy_changed=False)
    p=output/'report.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');p.with_suffix('.sha256').write_text(digest(p)+'\n')
    print(json.dumps({k:v for k,v in results['all'].items() if k not in ('limit_opportunity_vs_open',)},ensure_ascii=False,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/inputs')
    parser.add_argument('--research',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/research-v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prereg',type=Path,required=True);parser.add_argument('--prereg-sha256',required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('Preserve prior attempts; choose a new repository output')
    success=False;result=None;error=None;started=datetime.now(timezone.utc).isoformat()
    try:
        result=run(args.inputs,args.research,output,args.prereg,args.prereg_sha256);success=True
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        trials=[]
        for arm in ARMS:
            row=dict(source='entry_distance',timestamp=started,command=' '.join(sys.argv),arm=arm,completed=success,error=error,
                sharpe=None,cash_account=False,live_qualified=False,unseen_validation=False,
                prereg_sha256=args.prereg_sha256,output=str(output.relative_to(ROOT)),
                summary=result['results']['all'][arm] if success else None)
            row['registry_line']=append_trial_registry(row);trials.append(row)
        output.mkdir(parents=True,exist_ok=True);p=output/'trials.json'
        p.write_text(json.dumps(dict(actual_run_attempts=1,arm_evaluations=4,records=trials),ensure_ascii=False,indent=2,allow_nan=False)+'\n');p.with_suffix('.sha256').write_text(digest(p)+'\n')


if __name__=='__main__':main()
