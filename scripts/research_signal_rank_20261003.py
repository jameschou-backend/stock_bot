#!/usr/bin/env python3
"""Frozen T0 relative-strength ranks versus independent signal outcomes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.research_early_signal_losses import (PERIODS,digest,evaluate_filter,
                                                 statistics,verified_sources)
from skills.trial_registry import append_trial_registry

RANK_BANDS=('rank1','rank2','rank3','rank4_5','rank6_10','rank11_20','rank21_plus')
QUINTILES=('Q1_top20pct','Q2','Q3','Q4','Q5_bottom20pct')
SCORE_BINS=('score_0_10pp','score_10_20pp','score_20_40pp','score_above40pp')
ARMS=('baseline','top1','top3','top5',*RANK_BANDS,*QUINTILES,*SCORE_BINS)
EXPECTED_MANIFEST='6cd6ef3cbf9ebfced4741b2e60fea34212a2dc5605c6b6b5895c003a34bb65bf'
EXPECTED_WORKBOOK='ee7d2e6ffdcd8ba620419f0a9dabb2eb8a45c829d8e1cc13c2d6db52e66d476c'


def rank_band(rank):
    if type(rank) is not int or rank<1:raise ValueError('Positive integer rank required')
    return ('rank1' if rank==1 else 'rank2' if rank==2 else 'rank3' if rank==3 else
            'rank4_5' if rank<=5 else 'rank6_10' if rank<=10 else 'rank11_20' if rank<=20 else 'rank21_plus')


def score_bin(score):
    if not isinstance(score,(float,int)) or isinstance(score,bool) or not math.isfinite(score) or score<=0:
        raise ValueError('Finite positive original relative20 priority required')
    return ('score_0_10pp' if score<=.10 else 'score_10_20pp' if score<=.20 else
            'score_20_40pp' if score<=.40 else 'score_above40pp')


def rank_events(events,adjusted):
    """Rank every original same-day candidate before reading any outcome.

    Future signals/prices cannot change the date's N, score or tie-breaking.
    Scores are relative 20-market-session returns, not ML probabilities.
    """
    if (not isinstance(adjusted.index,pd.DatetimeIndex) or not adjusted.index.is_unique
            or not adjusted.index.is_monotonic_increasing or not adjusted.columns.is_unique or '0050' not in adjusted):
        raise ValueError('Ordered unique market calendar and benchmark required')
    r20=adjusted/adjusted.shift(20)-1
    relative=r20.sub(r20['0050'],axis=0)
    seen=set();stock_dates=set();groups=defaultdict(list)
    for event in events:
        key=event.get('event_id');day=event.get('signal_date')
        if not isinstance(key,str) or key in seen or len(event.get('members',[]))!=1:
            raise ValueError('Unique individual-stock event required')
        sid=event['members'][0]
        if (day,sid) in stock_dates or sid=='0050':raise ValueError('Unique stock per signal day required')
        score=event.get('priority');bucket=score_bin(score)
        stamp=pd.Timestamp(day)
        if stamp not in adjusted.index or sid not in adjusted:raise ValueError('Signal price coordinate missing')
        i=int(adjusted.index.get_loc(stamp))
        if i<20:raise ValueError('Signal lacks 20 prior market sessions')
        points=[float(adjusted.at[d,s]) for d in (stamp,adjusted.index[i-20]) for s in (sid,'0050')]
        if not all(math.isfinite(v) and v>0 for v in points):raise ValueError('Score observation missing')
        recomputed=float(relative.at[stamp,sid])
        if not math.isfinite(recomputed) or not math.isclose(recomputed,score,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Sealed priority differs from T0 relative20')
        groups[day].append(dict(signal_id=key,stock_id=sid,signal_date=day,
            rank_priority=float(score),rank_priority_recomputed=recomputed,
            rank_priority_error=abs(float(score)-recomputed),rank_score_bin=bucket,
            rank_stock_return20=float(r20.at[stamp,sid]),rank_benchmark_return20=float(r20.at[stamp,'0050']),
            rank_score_start=str(adjusted.index[i-20].date()),rank_available_at=day+' 收盤資料完成後'))
        seen.add(key);stock_dates.add((day,sid))
    results={}
    for day,group in groups.items():
        ordered=sorted(group,key=lambda r:(-r['rank_priority'],r['signal_id']));n=len(ordered)
        for k,row in enumerate(ordered,1):
            pct=(k-.5)/n
            quint=QUINTILES[min(4,int(pct*5))] if n>=10 else None
            row.update(daily_rank=k,daily_candidate_count=n,rank_band=rank_band(k),
                daily_midpoint_percentile=pct,rank_quintile=quint,
                rank_quintile_eligible=n>=10,
                rank_quintile_issue=None if n>=10 else 'fewer_than_10_original_candidates')
            for arm in ARMS:
                value=(True if arm=='baseline' else k<=int(arm[3:]) if arm in ('top1','top3','top5') else
                       row['rank_band']==arm if arm in RANK_BANDS else
                       (quint==arm if quint is not None else None) if arm in QUINTILES else row['rank_score_bin']==arm)
                row['rank_keep_'+arm]=value
            results[row['signal_id']]=row
    return results


def opportunities(rows,arm):
    closed=[r for r in rows if r['status']=='closed']
    known=[r for r in closed if r.get('rank_keep_'+arm) is not None]
    selected=[r for r in known if r['rank_keep_'+arm]]
    returns=float(sum(r['net_return'] for r in selected));unknown=len(closed)-len(known)
    return dict(original_closed_opportunities=len(closed),known_closed_opportunities=len(known),
        selected_closed_opportunities=len(selected),rejected_known_closed_opportunities=len(known)-len(selected),
        unknown_closed_opportunities=unknown,selected_unit_return_sum=returns,
        equal_units_mean_per_known_original_opportunity=returns/len(known) if known else None,
        unfiltered_mean_on_same_known_opportunities=float(np.mean([r['net_return'] for r in known])) if known else None,
        equal_units_mean_per_all_original_opportunity=returns/len(closed) if closed and not unknown else None,
        unknown_prevents_full_original_denominator=bool(unknown))


def arm_result(rows,arm):
    population=[r for r in rows if r['daily_candidate_count']>=10] if arm in QUINTILES else rows
    result=evaluate_filter(population,'rank_keep_'+arm)
    result['arm']=arm
    result['equal_units_opportunities']=opportunities(population,arm)
    result['scope_coverage']=dict(all_signals=len(rows),population_signals=len(population),
        original_closed=statistics(rows)['closed'],population_closed=statistics(population)['closed'],
        all_signal_days=len({r['signal_date'] for r in rows}),population_days=len({r['signal_date'] for r in population}),
        excluded_low_candidate_days=len({r['signal_date'] for r in rows if r['daily_candidate_count']<10}) if arm in QUINTILES else 0)
    result['p_win_given_selected']=result['kept']['win_rate']
    result['p_selected_given_win']=result['winner_retention']
    result['closed_opportunity_retention']=result['kept']['closed']/result['baseline']['closed'] if result['baseline']['closed'] else None
    result['selected_return30_count']=result['original_return30']-result['excluded_return30']-result['unknown_return30']
    result['selected_return30_rate']=result['selected_return30_count']/result['kept']['closed'] if result['kept']['closed'] else None
    return result


def matched_days(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['signal_date']].append(r)
    pairs=[];excluded=[]
    for day,group in sorted(groups.items()):
        n=group[0]['daily_candidate_count']
        if n<10:
            excluded.append(dict(signal_date=day,reason='fewer_than_10_original_candidates',n=n));continue
        selected=sorted([r for r in group if r['daily_rank']<=10],key=lambda r:r['daily_rank'])
        if len(selected)!=10 or [r['daily_rank'] for r in selected]!=list(range(1,11)):
            raise ValueError('Matched comparison lost original top10 ranking')
        if any(r['status']!='closed' for r in selected):
            excluded.append(dict(signal_date=day,reason='not_all_original_top10_closed',n=n,
                status_counts=dict(Counter(r['status'] for r in selected))));continue
        top,lower=selected[:3],selected[3:]
        a=float(np.mean([r['net_return'] for r in top]));b=float(np.mean([r['net_return'] for r in lower]))
        wa=float(np.mean([r['net_return']>0 for r in top]));wb=float(np.mean([r['net_return']>0 for r in lower]))
        pairs.append(dict(signal_date=day,n=n,top3_mean=a,ranks4_10_mean=b,paired_difference=a-b,
            top3_win_fraction=wa,ranks4_10_win_fraction=wb,paired_win_fraction_difference=wa-wb))
    def summary(subset):
        return dict(days=len(subset),top3_signals=3*len(subset),ranks4_10_signals=7*len(subset),
            mean_daily_top3_return=float(np.mean([r['top3_mean'] for r in subset])) if subset else None,
            mean_daily_ranks4_10_return=float(np.mean([r['ranks4_10_mean'] for r in subset])) if subset else None,
            mean_paired_difference=float(np.mean([r['paired_difference'] for r in subset])) if subset else None,
            median_paired_difference=float(np.median([r['paired_difference'] for r in subset])) if subset else None,
            top3_better_days=sum(r['paired_difference']>0 for r in subset),
            top3_worse_days=sum(r['paired_difference']<0 for r in subset),
            mean_daily_top3_win_fraction=float(np.mean([r['top3_win_fraction'] for r in subset])) if subset else None,
            mean_daily_ranks4_10_win_fraction=float(np.mean([r['ranks4_10_win_fraction'] for r in subset])) if subset else None)
    scopes={'all':pairs}
    scopes.update({str(y):[r for r in pairs if r['signal_date'].startswith(str(y))] for y in range(2019,2027)})
    scopes.update({name:[r for r in pairs if lo<=r['signal_date']<=hi] for name,(lo,hi) in PERIODS.items()})
    return dict(pairs=pairs,excluded=excluded,scopes={k:summary(v) for k,v in scopes.items()},
        coverage=dict(original_days=len(groups),n_ge10_days=sum(group[0]['daily_candidate_count']>=10 for group in groups.values()),
            matched_days=len(pairs),excluded_reasons=dict(Counter(r['reason'] for r in excluded))),
        independent_observations_claimed=False)


def score_outcome_distribution(rows):
    groups={name:[r for r in rows if r['status']=='closed' and ((r['net_return']>0) if name=='profit' else (r['net_return']<0))]
            for name in ('profit','loss')}
    out={}
    for name,group in groups.items():
        out[name]=dict(count=len(group),priority_mean=float(np.mean([r['rank_priority'] for r in group])) if group else None,
            priority_median=float(np.median([r['rank_priority'] for r in group])) if group else None,
            rank_band_counts=dict(Counter(r['rank_band'] for r in group)),
            top3_count=sum(r['daily_rank']<=3 for r in group))
    return out


def run(bundle,research,output,prereg,*,record_trials=False):
    bundle,research,output,prereg=[p.resolve() for p in (bundle,research,output,prereg)]
    if output.exists() or not output.is_relative_to(ROOT) or not prereg.is_file() or not prereg.is_relative_to(ROOT):
        raise ValueError('Require preregistration and a new repository output directory')
    refs,manifest,previous=verified_sources(bundle,research)
    if digest(bundle/'manifest.json')!=EXPECTED_MANIFEST or digest(research/'workbook-data.json')!=EXPECTED_WORKBOOK:
        raise ValueError('Inputs differ from the fixed preregistered evidence hashes')
    prereg_sha=digest(prereg)
    events=json.loads((bundle/'signals.json').read_text())['entries']
    if len(events)!=30188:raise ValueError('Require all frozen 30188 strategy signals')
    ids=sorted({e['members'][0] for e in events}|{'0050'})
    close=pd.read_parquet(bundle/'close-official.parquet',columns=['date',*ids]).set_index('date');close.index=pd.to_datetime(close.index)
    ranks=rank_events(events,close)
    frozen_features=pd.read_parquet(bundle/'signal-features.parquet').set_index('event_id')
    if not frozen_features.index.is_unique or set(frozen_features.index)!=set(ranks):
        raise ValueError('Frozen feature universe differs from signal ranks')
    for key,value in ranks.items():
        feature_score=frozen_features.at[key,'relative20']
        if not np.isfinite(feature_score) or not math.isclose(feature_score,value['rank_priority'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Frozen feature score differs from sealed priority')
    # Ranking is complete before this program loads individual future outcomes.
    payload=json.loads((research/'workbook-data.json').read_text());originals=payload['rows']
    if len(originals)!=30188 or len({r['signal_id'] for r in originals})!=len(originals) or set(ranks)!={r['signal_id'] for r in originals}:
        raise ValueError('Rank universe differs from original outcome universe')
    rows=[]
    for original in originals:
        ranked=ranks[original['signal_id']]
        if ranked['stock_id']!=original['stock_id'] or ranked['signal_date']!=original['signal_date']:
            raise ValueError('Rank coordinate differs from original signal')
        score=original.get('relative20')
        if not isinstance(score,(int,float)) or not math.isfinite(score) or not math.isclose(score,ranked['rank_priority'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Original workbook score differs from sealed priority')
        rows.append(dict(original,**{k:v for k,v in ranked.items() if k not in ('signal_id','stock_id','signal_date')}))
    scopes={'all':rows}
    scopes.update({str(y):[r for r in rows if r['signal_date'].startswith(str(y))] for y in range(2019,2027)})
    scopes.update({name:[r for r in rows if lo<=r['signal_date']<=hi] for name,(lo,hi) in PERIODS.items()})
    results={scope:{} for scope in scopes};trials=[]
    for arm in ARMS:
        all_scopes={scope:arm_result(group,arm) for scope,group in scopes.items()}
        for scope,value in all_scopes.items():results[scope][arm]=value
        record=dict(timestamp=datetime.now(timezone.utc).isoformat(),command=' '.join(sys.argv),
            source='signal_rank_20261003',sharpe=None,status='completed',
            params=dict(arm=arm,start='2019-01-01',end='2026-10-02',independent_signal_research=True,
                cash_account=False,unseen_validation=False),prereg_sha256=prereg_sha,
            result=all_scopes['all'],scope_results=all_scopes)
        if record_trials:record['registry_row']=append_trial_registry(record)
        trials.append(record)
    matched=matched_days(rows)
    quintile_baselines={}
    for scope,group in scopes.items():
        population=[r for r in group if r['daily_candidate_count']>=10]
        closed=[r for r in population if r['status']=='closed']
        big=sum(r['net_return']>=.30 for r in closed)
        quintile_baselines[scope]=dict(summary=statistics(population),original_scope_summary=statistics(group),
            original_days=len({r['signal_date'] for r in group}),included_days=len({r['signal_date'] for r in population}),
            excluded_smallN_signals=len(group)-len(population),return30_count=big,
            return30_rate=big/len(closed) if closed else None,equal_units_opportunities=opportunities(population,'baseline'))
    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),command=' '.join(sys.argv),source='signal_rank_20261003',
        sharpe=None,status='completed',params=dict(arm='matched_daily_top3_vs_ranks4_10',min_candidates=10,
            require_all_original_top10_closed=True,independent_signal_research=True,cash_account=False,unseen_validation=False),
        prereg_sha256=prereg_sha,result=dict(scopes=matched['scopes'],coverage=matched['coverage']))
    if record_trials:record['registry_row']=append_trial_registry(record)
    trials.append(record)
    output.mkdir(parents=True)
    def dump(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(rows).to_parquet(output/'signal-ranks.parquet',index=False)
    pd.DataFrame(matched['pairs']).to_parquet(output/'matched-days.parquet',index=False)
    dump('rank-comparisons.json',results);dump('matched-comparisons.json',matched);dump('trials.json',trials)
    dump('quintile-baselines.json',quintile_baselines)
    for p in (Path(__file__),prereg,ROOT/'tests/test_signal_rank_research.py',ROOT/'scripts/research_early_signal_losses.py',ROOT/'skills/trial_registry.py'):
        refs[str(p.relative_to(ROOT))]=digest(p)
    report=dict(schema='frozen_signal_rank_research_v1',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs,output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        sample_count=len(rows),daily_signal_days=len({r['signal_date'] for r in rows}),
        max_priority_recomputation_error=max(r['rank_priority_error'] for r in rows),all_results=results['all'],
        score_outcome_distribution=score_outcome_distribution(rows),matched_summary=matched['scopes']['all'],
        matched_coverage=matched['coverage'],quintile_baseline=quintile_baselines['all'],registry_recorded=record_trials,
        definitions=dict(score='Adjusted stock 20-market-session return minus0050 same-date return; original priority, not model probability',
            ordering='Each date ALL original signals sorted by descending priority then ascending event_id, before outcomes',
            percentile='(rank-0.5)/N; lower is stronger; quintile intervals [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1); only N>=10',
            raw_score_bins='(0,.10],(.10,.20],(.20,.40],(.40,infinity)',
            equal_units='Each original known closed opportunity receives original return if selected or explicit cash0 if rejected; not portfolio',
            matched='Same original date N>=10 and all original top10 closed: mean(rank1..3)-mean(rank4..10), day weighted equally'),
        limitations=payload['metadata']['limitations']+[
            'Ranking is daily relative strength, not an earnings, news, institutional-flow or ML probability score.',
            'Sparse candidate days remain in absolute rankings and are excluded explicitly from daily quintile populations.',
            'Matched-day results omit dates without all first10 complete outcomes; lost coverage is reported and not evidence of avoided losses.',
            'Repeated stocks and overlapping holding periods are dependent; no independent-sample t-test or confidence claim.',
            'No portfolio rerun or threshold search; every result is previously studied historical evidence.'],
        live_qualified=False,cash_account=False,unseen_validation=False)
    dump('report.json',report);(output/'report.sha256').write_text(digest(output/'report.json')+'\n')
    print(json.dumps(dict(total=len(rows),top_filters={k:results['all'][k] for k in ('baseline','top1','top3','top5')},matched=report['matched_summary']),ensure_ascii=False,indent=2))
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
        if a.prereg.is_file():
            receipt=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='signal_rank_20261003',
                status='failed',sharpe=None,command=' '.join(sys.argv),prereg_sha256=digest(a.prereg),
                error_type=type(exc).__name__,error=str(exc),
                note='Completed arm evaluations were recorded immediately before this failure; do not delete them.')
            append_trial_registry(receipt)
            failure=a.output.parent/(a.output.name+'-failure.json')
            if failure.resolve().is_relative_to(ROOT):
                failure.parent.mkdir(parents=True,exist_ok=True)
                if not failure.exists():failure.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        raise
