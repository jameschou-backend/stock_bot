#!/usr/bin/env python3
"""Fixed point-in-time breakout recurrence phases; independent signal outcomes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.research_early_signal_losses import (PERIODS, digest, evaluate_filter,
                                                 statistics, verified_sources)
from skills.trial_registry import append_trial_registry

PHASES={'first_after_20':'20日未突破後首次',
        'rebreak_after_5':'休息5日後再突破',
        'recent_repeat':'近期重複突破'}


def price_breakout_history(close, other, eligible):
    """All price-only breakouts, regardless of volume, market regime or signals.

    A flag is unknown unless the current and prior 60 observations have valid
    positive comparable adjusted prices and explicit listing eligibility.
    Per-day adjustment jumps use the original study's 20% / 0.5pp checks.
    No future reference prices or confirmed peaks are accessed.
    """
    for value in (close,other,eligible):
        if not isinstance(value,pd.Series) or not value.index.equals(close.index):
            raise ValueError('Breakout arrays must share an explicit calendar')
    if not isinstance(close.index,pd.DatetimeIndex) or not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError('Unique ordered market dates required')
    if not eligible.dropna().isin([True,False]).all():
        raise ValueError('Eligibility must be bool or missing')
    c,independent=close.astype(float),other.astype(float)
    good=(np.isfinite(c)&c.gt(0)&np.isfinite(independent)&independent.gt(0)&eligible.eq(True).fillna(False))
    own_return=c/c.shift(1)-1;other_return=independent/independent.shift(1)-1
    pair_good=(good & good.shift(1,fill_value=False)&own_return.abs().le(.20)&other_return.abs().le(.20)
               &(own_return-other_return).abs().le(.005))
    # 61 valid close observations contain exactly 60 known adjacent returns.
    known=(good.rolling(61,min_periods=61).sum().eq(61)
           &pair_good.rolling(60,min_periods=60).sum().eq(60))
    prior_high=c.where(good).shift(1).rolling(60,min_periods=60).max()
    flag=pd.Series(pd.NA,index=c.index,dtype='boolean')
    flag.loc[known]=c.loc[known].gt(prior_high.loc[known])
    return pd.DataFrame({'breakout':flag,'prior_high60':prior_high,
                         'known':known,'adjusted_close':c})


def classify_phase(history,signal_date):
    """Disjoint categories from prior indices i-20..i-1 only."""
    i=int(history.index.get_loc(pd.Timestamp(signal_date)))
    result=dict(phase='unknown',phase_issue=None,phase_available_at=str(pd.Timestamp(signal_date).date())+' 收盤資料完成後',
        phase_current_price_breakout=None,phase_previous20_unknown_count=None,
        phase_prior5_breakout_count=None,phase_prior20_breakout_count=None,
        phase_last_breakout_distance_in20=None,phase_breakout_threshold=None,
        phase_breakout_fraction=None)
    current=history.iloc[i]
    if not bool(current['known']):
        result['phase_issue']='current_price_breakout_quality_unknown';return result
    result['phase_current_price_breakout']=bool(current.breakout)
    result['phase_breakout_threshold']=float(current.prior_high60)
    result['phase_breakout_fraction']=float(current.adjusted_close/current.prior_high60-1)
    if not current.breakout:
        result['phase']='not_price_breakout';result['phase_issue']='original_signal_not_reproduced_by_price_breakout';return result
    if i<20:
        result['phase_issue']='fewer_than_20_prior_market_sessions';return result
    previous=history.breakout.iloc[i-20:i]
    unknown=int(previous.isna().sum());result['phase_previous20_unknown_count']=unknown
    if unknown:
        result['phase_issue']='unknown_price_breakout_in_prior20';return result
    bits=previous.to_numpy(dtype=bool)
    count20=int(bits.sum());count5=int(bits[-5:].sum())
    result.update(phase_prior5_breakout_count=count5,phase_prior20_breakout_count=count20,
        phase_last_breakout_distance_in20=int(20-np.flatnonzero(bits)[-1]) if count20 else None,
        phase='first_after_20' if count20==0 else 'recent_repeat' if count5 else 'rebreak_after_5')
    return result


def annotate_signals(rows, close, other, eligible):
    """Keep all original rows; recurrence is built from daily prices, not rows."""
    if any(not f.index.equals(close.index) or not f.columns.equals(close.columns) for f in (other,eligible)):
        raise ValueError('Input feature axes differ')
    histories={sid:price_breakout_history(close[sid],other[sid],eligible[sid])
               for sid in sorted({r['stock_id'] for r in rows})}
    result=[]
    for original in rows:
        row=dict(original);row.update(classify_phase(histories[row['stock_id']],row['signal_date']))
        row['phase_source_scope']=('FinMind延伸，未完成官方交叉核對' if row['signal_date']>'2026-09-09'
                                  else '修復後封存歷史訊號資料，仍非全市場與成交認證')
        for name in PHASES:
            row['phase_keep_'+name]=bool(row['phase']==name) if row['phase'] in PHASES else None
        result.append(row)
    return result


def opportunity_summary(rows, arm):
    """One unit per known original closed opportunity; rejected units stay cash.

    Unknown phase assignments are shown separately. They never silently receive
    a zero return. This is a sum of overlapping unit observations, not a funded
    account, reinvestment path, annual return, or capacity-certified outcome.
    """
    closed=[r for r in rows if r['status']=='closed']
    known=[r for r in closed if arm=='baseline' or r['phase'] in PHASES]
    selected=known if arm=='baseline' else [r for r in known if r['phase']==arm]
    total=float(sum(r['net_return'] for r in selected))
    unknown=len(closed)-len(known)
    return dict(original_closed_opportunities=len(closed),known_closed_opportunities=len(known),
        unknown_closed_opportunities=unknown,selected_closed_opportunities=len(selected),
        rejected_known_closed_opportunities=len(known)-len(selected),selected_unit_return_sum=total,
        equal_units_mean_per_known_original_opportunity=total/len(known) if known else None,
        unfiltered_mean_on_same_known_opportunities=float(np.mean([r['net_return'] for r in known])) if known else None,
        equal_units_mean_per_all_original_opportunity=total/len(closed) if closed and not unknown else None,
        unknown_prevents_full_original_denominator=bool(unknown))


def compare(rows):
    scopes={'all':rows}
    scopes.update({str(y):[r for r in rows if r['signal_date'].startswith(str(y))] for y in range(2019,2027)})
    scopes.update({name:[r for r in rows if start<=r['signal_date']<=end] for name,(start,end) in PERIODS.items()})
    results={}
    for name,group in scopes.items():
        result={}
        for arm in ['baseline',*PHASES]:
            value=evaluate_filter(group,None if arm=='baseline' else 'phase_keep_'+arm)
            value['arm']=arm;value['equal_units_opportunities']=opportunity_summary(group,arm)
            result[arm]=value
        result['unknown_phase']=statistics([r for r in group if r['phase']=='unknown'])
        result['not_price_breakout']=statistics([r for r in group if r['phase']=='not_price_breakout'])
        results[name]=result
    return results


def record_attempt(arm,prereg_sha,summary,*,status='completed',output=None):
    value=dict(timestamp=datetime.now(timezone.utc).isoformat(),command=' '.join(sys.argv),
        source='breakout_phase_20261002',sharpe=None,status=status,
        params=dict(arm=arm,price_breakout_lookback=60,prior_phase_window=20,recent_window=5,
            start='2019-01-01',end='2026-10-02',independent_signal_research=True,
            cash_account=False,unseen_validation=False),prereg_sha256=prereg_sha,result=summary)
    if output:value['output']=str(output.relative_to(ROOT))
    return value


def run(bundle,research,output,prereg,*,record_trials=False):
    bundle,research,output,prereg=[p.resolve() for p in (bundle,research,output,prereg)]
    if output.exists() or not output.is_relative_to(ROOT) or not prereg.is_file() or not prereg.is_relative_to(ROOT):
        raise ValueError('Require registered specification and a new repository output')
    prereg_sha=digest(prereg)
    refs,manifest,prior=verified_sources(bundle,research)
    payload=json.loads((research/'workbook-data.json').read_text());rows=payload['rows']
    events=json.loads((bundle/'signals.json').read_text())['entries']
    if len(rows)!=30188 or len({r['signal_id'] for r in rows})!=len(rows) or {r['signal_id'] for r in rows}!={e['event_id'] for e in events}:
        raise ValueError('Require all 30,188 original strategy signals')
    ids=sorted({r['stock_id'] for r in rows})
    frames={n:pd.read_parquet(bundle/(n+'.parquet'),columns=['date',*ids]).set_index('date')
            for n in ('close-official','close-quality','eligibility')}
    for f in frames.values():f.index=pd.to_datetime(f.index)
    enriched=annotate_signals(rows,*[frames[n] for n in ('close-official','close-quality','eligibility')])
    results=compare(enriched)
    output.mkdir(parents=True)
    def dump(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(enriched).to_parquet(output/'signal-features.parquet',index=False)
    dump('phase-comparisons.json',results)
    contributions,missed=[],[]
    closed=[r for r in enriched if r['status']=='closed']
    for arm in ['baseline',*PHASES]:
        for sid in sorted({r['stock_id'] for r in closed}):
            stock=[r for r in closed if r['stock_id']==sid]
            kept=stock if arm=='baseline' else [r for r in stock if r['phase']==arm]
            contributions.append(dict(arm=arm,stock_id=sid,name=stock[0]['name'],
                original_closed_count=len(stock),selected_closed_count=len(kept),
                selected_unit_return_sum=float(sum(r['net_return'] for r in kept)),
                selected_mean_return=float(np.mean([r['net_return'] for r in kept])) if kept else None,
                selected_positive_unit_return_sum=float(sum(max(0,r['net_return']) for r in kept)),
                selected_negative_unit_return_sum=float(sum(min(0,r['net_return']) for r in kept))))
        if arm!='baseline':
            for r in closed:
                if r['net_return']>=.30 and r['phase']!=arm:
                    missed.append(dict(arm=arm,signal_id=r['signal_id'],stock_id=r['stock_id'],name=r['name'],
                        signal_date=r['signal_date'],entry_date=r['entry_date'],exit_date=r['exit_date'],
                        net_return=r['net_return'],holding_days=r['holding_days'],phase=r['phase'],
                        exclusion_kind='known_rejected' if r['phase'] in PHASES else 'unknown_phase'))
    pd.DataFrame(contributions).to_parquet(output/'stock-contributions.parquet',index=False)
    pd.DataFrame(missed).to_parquet(output/'missed-return30-signals.parquet',index=False)
    source_files=[Path(__file__),ROOT/'tests/test_breakout_phase_research.py',prereg,
        ROOT/'scripts/research_early_signal_losses.py',ROOT/'skills/independent_three_black.py',ROOT/'skills/trial_registry.py']
    for path in source_files:refs[str(path.relative_to(ROOT))]=digest(path)
    records=[]
    for arm in ['baseline',*PHASES]:
        record=record_attempt(arm,prereg_sha,results['all'][arm],output=output)
        record['scope_results']={scope:value[arm] for scope,value in results.items()}
        if record_trials:record['registry_row']=append_trial_registry(record)
        records.append(record)
    dump('trials.json',records)
    report=dict(schema='fixed_breakout_phase_research_v1',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs,output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        sample_count=len(enriched),phase_counts=dict(Counter(r['phase'] for r in enriched)),
        issue_counts=dict(Counter(r['phase_issue'] for r in enriched if r['phase_issue'])),
        all_results=results['all'],registry_recorded=record_trials,
        rules=dict(phases=PHASES,price_breakout='close[t] > max(close[t-60:t]); adjusted prices; all sessions regardless of volume/regime/strategy events',
                   first_after_20='No breakout in t-20..t-1',rebreak_after_5='Some breakout t-20..t-6, none t-5..t-1',
                   recent_repeat='Some breakout t-5..t-1',unknown='At least one unknown flag in prior20, or current breakout quality unknown',
                   phase_is_proxy=True),
        limitations=payload['metadata']['limitations']+[
            'Recurrence phases are dated descriptive proxies, not confirmed bases, future peaks or last rally legs.',
            'Every eligible historical price breakout participates; no selection by volume or membership in the strategy signal table.',
            'Unknown classifications preserved outside each filter, with their outcomes reported separately.',
            'Equal-unit opportunity means include zero for known rejected signals only; unknowns remain separate and prevent full-denominator reporting.',
            'No threshold search, combinations, capital allocations or portfolio returns; all periods were previously studied.',
            'Direct frozen source hashes verified; no new independent official-origin or publication-time certification.'],
        live_qualified=False,cash_account=False,unseen_validation=False)
    dump('report.json',report);(output/'report.sha256').write_text(digest(output/'report.json')+'\n')
    print(json.dumps(dict(phases=report['phase_counts'],results=report['all_results']),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/inputs')
    p.add_argument('--research',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/research-v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prereg',type=Path,required=True)
    a=p.parse_args()
    try:run(a.inputs,a.research,a.output,a.prereg,record_trials=True)
    except Exception as exc:
        # Failed CLI attempts are visible; unit-test imports never append trials.
        if a.prereg.is_file():
            record=record_attempt('preparation_or_publication_failure',digest(a.prereg),
                dict(error_type=type(exc).__name__,error=str(exc),intended_arms=['baseline',*PHASES]),status='failed')
            append_trial_registry(record)
        raise
