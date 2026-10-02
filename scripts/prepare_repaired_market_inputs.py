#!/usr/bin/env python3
"""Rebuild fixed-strategy inputs offline without changing sealed research."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_million_signals import official_adjusted
from skills.historical_selector_replay import eligibility_matrix
from skills.historical_identity_repair import load_identity_repair,apply_identity_repair,account_entry_decision
from skills.independent_factor_repair import derive_adjusted_close
from skills.liquidity_candidates import filter_candidates
from skills.market_input_validation import require,resolve_episode
from skills.official_market_classification import load_nonordinary_overlay
from skills.official_market_supplement import bound_path,sha
from skills.official_quote_repair import encode,verify_document
from skills.official_quote_repair_enrichment import align_independent_close,provider_rows
from skills.repaired_market_inputs import merge_quote_repairs,candidate_diff,apply_entry_policy,observed_roster_check
from skills.stock_universe_2019 import generate

BASE=ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
REPAIRS=ROOT/'.cache/market-input-validation-20261002/repairs-v3'
EVIDENCE=ROOT/'.cache/market-input-repair-20261002/quote-evidence'
CUTOFF='2026-09-08'


def read(path):return json.loads(Path(path).read_text())


def required_history_days(calendar, entries, *, start='2019-01-02',end='2026-09-09',lookback=126):
    stamps=[str(pd.Timestamp(d).date()) for d in calendar]
    require(stamps==sorted(set(stamps)),'Unique ordered history calendar required')
    positions={d:i for i,d in enumerate(stamps)}
    required={d for d in stamps if start<=d<=end}
    for entry in entries:
        i=positions[entry['signal_date']]
        require(i>=lookback,'Candidate lacks explicit 126-session history')
        required.update(stamps[i-lookback:i+1])
    return sorted(required)


def candidate_prefix_inputs(frames, groups, cutoff):
    """Keep next-session calendar identity while removing its observations."""
    days=frames['raw-close'].index
    valid=days[days<=pd.Timestamp(cutoff)]
    require(len(valid)>0,'Prefix cutoff precedes history')
    last=valid[-1];position=days.get_loc(last)
    require(position+1<len(days),'Prefix requires a following calendar session')
    entry=days[position+1]
    prefix={name:frame.loc[:entry].copy() for name,frame in frames.items()}
    for name,frame in prefix.items():
        frame.loc[entry]=False if name=='eligibility' else np.nan
    known=deepcopy(groups)
    known['entries']=[e for e in known['entries'] if e['signal_date']<=str(last.date())]
    known['diffusion']['groups']=[g for g in known['diffusion']['groups']
        if g['month']<=str(last.to_period('M'))]
    return prefix,known,str(last.date()),str(entry.date())


def audit_candidate_prefixes(frames,companies,groups,identity,full_raw,full_accepted):
    """Exact all-candidate comparison, using only observations through each close."""
    checks=[]
    for cutoff in ('2023-12-29','2025-12-31'):
        prefix,known,last,entry=candidate_prefix_inputs(frames,groups,cutoff)
        raw=generate(prefix,companies,known,last)['liquid_universe']
        expected_raw=[e for e in full_raw if e['signal_date']<=last]
        require(raw==expected_raw,'Truncated raw candidate prefix differs: '+cutoff)
        filtered,_=filter_candidates(prefix['raw-close'],prefix['raw-volume'],raw,last)
        arms={}
        for name,entries in filtered.items():
            actual,_=apply_entry_policy(entries,identity,account_entry_decision,
                research_risk_notice_assumed=False)
            expected=[e for e in full_accepted[name] if e['signal_date']<=last]
            require(actual==expected,'Truncated accepted candidate prefix differs: '+name+'/'+cutoff)
            arms[name]=dict(count=len(actual),exact_events_equal=True,
                events_sha256=hashlib.sha256(encode(actual).encode()).hexdigest())
        checks.append(dict(requested_cutoff=cutoff,signal_end=last,next_session_calendar_only=entry,
            latest_observation_date=last,raw_candidates_checked=len(raw),
            raw_exact_events_equal=True,raw_events_sha256=hashlib.sha256(encode(raw).encode()).hexdigest(),
            arms=arms))
    return dict(schema='repaired_candidate_prefix_audit_v1',checks=checks,
        all_exact=True,future_price_volume_and_eligibility_observations_removed=True,
        future_monthly_groups_removed=True,parameters_changed=False,
        source_snapshot_is_revised=True,point_in_time_publication_archive_proven=False,
        unseen_validation=False,live_qualified=False)


def prepare(output):
    output=Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(),'Choose a new repository input bundle')
    ready,refs=verify_document(ROOT,EVIDENCE/'ready-v1/report.json')
    for name,digest in ready['output_sha256'].items():
        require(sha(bound_path(ROOT,name))==digest,'Ready quote output changed')
        refs[name]=digest
    def source(path,expected=None):
        path=Path(path).resolve();require(path.is_relative_to(ROOT),'Source escapes repository')
        name=str(path.relative_to(ROOT));digest=sha(path)
        if expected is None:expected=refs.get(name)
        require(expected is not None and digest==expected,'Unbound or changed input: '+name)
        require(name not in refs or refs[name]==digest,'Conflicting input hash: '+name)
        refs[name]=digest
        return path
    repair=read(source(REPAIRS/'report.json'))
    for name,digest in repair['output_sha256'].items():source(ROOT/name,digest)
    frames={name:pd.read_parquet(source(REPAIRS/(name+'.parquet'))).set_index('date')
            for name in ('raw-close','raw-volume','close-official','close-quality','eligibility')}
    for frame in frames.values():frame.index=pd.to_datetime(frame.index)
    old_frames={k:v.copy() for k,v in frames.items()}
    companies=pd.read_parquet(source(REPAIRS/'companies.parquet'))
    old_companies=companies.copy()
    identity=read(source(REPAIRS/'identity.json'))
    quotes=pd.read_parquet(source(REPAIRS/'quotes-unmasked.parquet'))
    quotes.date=pd.to_datetime(quotes.date)
    events=pd.read_parquet(source(REPAIRS/'events.parquet'))
    groups=read(source(BASE/'signals.json'))
    prior=read(source(ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json'))
    liquid=read(source(ROOT/'.cache/liquidity-universe-20261001/signals-v1.json'))
    print('Reproduce frozen candidate generation and liquidity filters',flush=True)
    before=generate(frames,companies,groups,CUTOFF)['liquid_universe']
    require(before==prior['entries']['liquid_universe'],'Frozen candidate reproduction differs')
    baseline,_=filter_candidates(frames['raw-close'],frames['raw-volume'],before,CUTOFF)
    require(set(liquid['entries'])=={'liquid_universe','median50m','prior50m','persistent50m'},
            'Unexpected frozen liquidity arms')
    require(baseline['original']==baseline['cap40']==liquid['entries']['liquid_universe']
            and all(baseline[name]==liquid['entries'][name] for name in
                    ('median50m','prior50m','persistent50m')),'Frozen liquidity candidate reproduction differs')

    additions=pd.read_parquet(source(EVIDENCE/'ready-v1/quote-supplement-ready.parquet'))
    initial_unresolved=read(source(EVIDENCE/'ready-v1/unresolved.json'))
    print('Derive the two missing 1583 adjusted values from independent bracketed factors',flush=True)
    ledger=read(source(EVIDENCE/'ready-v1/provider-ledger-snapshot.json'))
    matches=[item for item in ledger['attempts'].values() if item['query']['dataset']=='TaiwanStockPriceAdj'
             and item['query']['data_id']=='1583']
    require(len(matches)==1,'Missing unique independent 1583 source')
    item=matches[0];receipt=read(source(ROOT/item['path'],item['sha256']))
    independent=pd.DataFrame(list(provider_rows(receipt,item['query']).values()))
    independent.date=pd.to_datetime(independent.date)
    series=independent.set_index('date')['close'].sort_index()
    normal_path=source(EVIDENCE/'official-normalized.parquet')
    anchors=pd.read_parquet(normal_path,filters=[('stock_id','==','1583'),
        ('date','>=',item['query']['start_date']),('date','<=',item['query']['end_date'])],
        columns=['date','stock_id','market','close'])
    require(not anchors.duplicated('date').any() and set(anchors.market)=={'TWSE'},'Ambiguous 1583 official anchors')
    anchors.date=pd.to_datetime(anchors.date)
    official_raw=anchors.sort_values('date').set_index('date')['close']
    derived=[]
    for index,row in additions.loc[additions.stock_id.eq('1583')].iterrows():
        require(pd.isna(row.quality_adjusted_close),'Unexpected existing 1583 adjusted repair')
        detail=derive_adjusted_close(row.date,row.close,series,official_raw,frames['close-quality']['1583'],
            events.loc[events.stock_id.eq('1583'),'event_date'])
        additions.at[index,'quality_adjusted_close']=detail['value']
        additions.at[index,'quality_alignment_method']=detail['method']
        additions.at[index,'quality_adjusted_verified']=True
        additions.at[index,'ready_for_signal_rebuild']=True
        derived.append(dict(stock_id='1583',date=row.date,source_path=item['path'],**detail))
    require(len(derived)==2,'Expected two explicit factor-derived repairs')
    complete=additions.total_daily_volume.notna() & additions.quality_adjusted_close.notna()
    unresolved=additions.loc[~complete].copy()
    require(set(unresolved.stock_id)=={'4415'} and len(unresolved)==9,'Unexpected unresolved quote scope')
    quotes,frames=merge_quote_repairs(quotes,frames,additions.loc[complete])

    print('Restore listing histories while keeping account access separate',flush=True)
    overlay=load_identity_repair(ROOT/'docs/historical_identity_repair_20261002.json',ROOT,refs)
    repaired_identity=apply_identity_repair(identity,overlay)
    from skills.historical_trading_repair import load_trading_repair,apply_trading_repair
    trading=load_trading_repair(ROOT/'docs/historical_trading_repair_20261002.json',ROOT,refs)
    repaired_identity=apply_trading_repair(repaired_identity,trading)
    for entry in overlay['entries']:
        require(companies.stock_id.eq(entry['stock_id']).sum()==1,'Repaired identity missing from companies')
        companies.loc[companies.stock_id.eq(entry['stock_id']),'listed_date']=pd.Timestamp(entry['start'])
    days,ids=frames['raw-close'].index,frames['raw-close'].columns
    mask=eligibility_matrix(repaired_identity,companies,days).reindex(columns=ids)
    changed=mask & ~old_frames['eligibility']
    removed=old_frames['eligibility'] & ~mask
    # Quote history stays intact; only dated listing validity masks signal frames.
    raw=quotes.pivot(index='date',columns='stock_id',values='close').reindex(index=days,columns=ids)
    volume=quotes.pivot(index='date',columns='stock_id',values='volume').reindex_like(raw)
    require(raw.index.equals(frames['raw-close'].index),'Changed strategy calendar')
    quality=frames['close-quality'].copy()
    early_path=source(ROOT/'artifacts/adj_prices/adj_prices_10y.parquet')
    changed_ids=list(changed.columns[changed.any()])
    early=pd.read_parquet(early_path,filters=[('stock_id','in',changed_ids)])
    early.trading_date=pd.to_datetime(early.trading_date)
    restored_quality=[];restoration_missing=[]
    for sid in changed_ids:
        observations=early.loc[early.stock_id.eq(sid)].sort_values('trading_date').set_index('trading_date')['close']
        require(observations.index.is_unique,'Duplicate independent listing-prefix series')
        needed=changed[sid] & raw[sid].gt(0) & ~quality[sid].gt(0)
        for day in days[needed]:
            if day not in observations.index or not pd.notna(observations.at[day]) or observations.at[day]<=0:
                restoration_missing.append(dict(stock_id=sid,date=str(day.date()),reason='independent_listing_history_missing'))
                continue
            detail=align_independent_close(observations,old_frames['close-quality'][sid],str(day.date()))
            quality.at[day,sid]=detail['value']
            restored_quality.append(dict(stock_id=sid,date=str(day.date()),value=detail['value'],
                scale=detail['scale'],method=detail['method'],overlap_count=detail['overlap_count'],
                source_path=str(early_path.relative_to(ROOT))))
    require(not restoration_missing,'Newly eligible positive quote lacks independent adjusted history')
    frames={'raw-close':raw.where(mask),'raw-volume':volume.where(mask),'close-quality':quality.where(mask),
            'close-official':official_adjusted(raw,events).where(mask),'eligibility':mask}
    affected=set(additions.stock_id)|{e['stock_id'] for e in overlay['entries']}
    affected.update(e['stock_id'] for e in repaired_identity['trading_exclusions']
                    if e not in identity['trading_exclusions'])
    untouched=[sid for sid in ids if sid not in affected]
    require(all(frames[name][untouched].equals(old_frames[name][untouched]) for name in frames),
            'Unrelated stock frame changed during repair')
    print('Recompute fixed candidates and record account-only rejections',flush=True)
    after=generate(frames,companies,groups,CUTOFF)['liquid_universe']
    filtered,decisions=filter_candidates(frames['raw-close'],frames['raw-volume'],after,CUTOFF)
    accepted,blocked={},{}
    for name,entries in filtered.items():
        accepted[name],blocked[name]=apply_entry_policy(entries,repaired_identity,account_entry_decision,
            research_risk_notice_assumed=False)
    diffs={name:candidate_diff(baseline[name],accepted[name]) for name in accepted}
    print('Verify two complete candidate prefixes with future observations removed',flush=True)
    prefix_audit=audit_candidate_prefixes(frames,companies,groups,repaired_identity,after,accepted)

    print('Audit required daily rosters and newly needed history',flush=True)
    required_dates=set(required_history_days(days,accepted['median50m']))
    official=pd.read_parquet(normal_path,columns=['market','date','stock_id'])
    by_sid=defaultdict(list)
    for ep in repaired_identity['episodes']:by_sid[ep['stock_id']].append(ep)
    classifications=load_nonordinary_overlay(ROOT/'docs/official_market_nonordinary_overlay_20261002.json',ROOT,refs)
    classified=defaultdict(list)
    for row in classifications:classified[row['stock_id']].append(row)
    roster=observed_roster_check(official,repaired_identity,
        lambda report,sid,stamp:resolve_episode(by_sid[sid],sid,stamp),sorted(required_dates),ids,
        nonordinary=lambda sid,stamp,market:any(r['market'].upper()==market and r['start']<=stamp<r['end']
                                               for r in classified[sid]))
    prior_audit=read(source(ROOT/'artifacts/forward_simulation/three_black_market_supplement_20261002.json'))
    old_required={r['date'] for r in prior_audit['request_plan']}
    history=dict(required_dates=sorted(required_dates),newly_required_dates=sorted(required_dates-old_required),
        dates_before_previous_start=sorted(d for d in required_dates if d<'2018-08-15'),
        missing_market_dates=roster['missing_market_dates'],lookback_sessions=126,
        source_scope='all_accepted_median50m_candidates_and_entire_account_period')
    known_exclusions=deepcopy(repaired_identity['trading_exclusions'])
    unresolved_scope=[]
    for row in unresolved.itertuples(index=False):
        matches=[e for e in known_exclusions if e['stock_id']==row.stock_id and
                 e['start']<=str(row.date)[:10] and (e['end'] is None or str(row.date)[:10]<e['end'])]
        unresolved_scope.append(dict(stock_id=row.stock_id,date=str(row.date)[:10],
            raw_ohlc_preserved=True,total_volume_missing=pd.isna(row.total_daily_volume),
            independent_adjusted_missing=pd.isna(row.quality_adjusted_close),
            dated_exclusions=matches,in_strategy_listing_scope=bool(mask.at[pd.Timestamp(row.date),row.stock_id])))

    output.mkdir(parents=True)
    for name,frame in frames.items():frame.rename_axis('date').reset_index().to_parquet(output/(name+'.parquet'),index=False)
    quotes.to_parquet(output/'quotes-unmasked.parquet',index=False)
    companies.to_parquet(output/'companies.parquet',index=False)
    events.to_parquet(output/'events.parquet',index=False)
    for name,value in [('identity.json',repaired_identity),('candidate-diff.json',diffs),
        ('raw-candidates.json',dict(entries=filtered)),('account-entry-rejections.json',blocked),
        ('roster-audit.json',roster),('required-history.json',history),('derived-adjusted-repairs.json',derived),
        ('restored-quality-history.json',restored_quality),('original-unresolved-quote-issues.json',initial_unresolved),
        ('known-trading-exclusions.json',known_exclusions),('unresolved-quote-scope.json',unresolved_scope),
        ('candidate-prefix-audit.json',prefix_audit),
        ('liquidity-decisions.json',decisions)]:
        (output/name).write_text(encode(value))
    additions.to_parquet(output/'quote-repair-evidence.parquet',index=False)
    unresolved.to_parquet(output/'unresolved-quote-evidence.parquet',index=False)
    for name in ('scripts/prepare_repaired_market_inputs.py','skills/repaired_market_inputs.py',
                 'skills/independent_factor_repair.py','skills/historical_identity_repair.py',
                 'skills/historical_trading_repair.py',
                 'skills/historical_selector_replay.py','skills/official_quote_repair_enrichment.py',
                 'skills/stock_universe_2019.py','skills/liquidity_candidates.py',
                 'scripts/prepare_million_signals.py','skills/official_adj_factors.py'):
        path=ROOT/name;digest=sha(path)
        require(name not in refs or refs[name]==digest,'Changed source code in existing closure: '+name)
        refs[name]=digest
    (output/'signals.json').write_text(encode(dict(entries=accepted,source_sha256=refs,
        candidate_parameters_changed=False,account_policy='ordinary_retail_no_risk_notice_assumption',
        live_qualified=False,unseen_validation=False)))
    report=dict(schema='repaired_market_input_bundle_v1',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs,files_sha256={p.name:sha(p) for p in output.iterdir()},
        baseline_reproduced=True,quote_repairs=len(additions.loc[complete]),
        directly_aligned_adjusted_repairs=int(complete.sum())-len(derived),derived_adjusted_repairs=len(derived),
        unresolved_quote_rows=len(unresolved),unresolved_stock_ids=sorted(set(unresolved.stock_id)),
        unresolved_quote_rows_in_strategy_scope=sum(r['in_strategy_listing_scope'] for r in unresolved_scope),
        initial_unresolved_quote_rows=len(initial_unresolved),
        restored_quality_rows=len(restored_quality),
        untouched_stock_frames_identical=True,
        candidate_prefix_checks=len(prefix_audit['checks']),candidate_prefix_exact=prefix_audit['all_exact'],
        listing_history_days_added={sid:int(v) for sid,v in changed.sum().items() if v},
        listing_history_days_removed={sid:int(v) for sid,v in removed.sum().items() if v},
        median50m_candidate_diff=diffs['median50m'],
        account_rejections={name:len(items) for name,items in blocked.items()},
        required_history=history,roster_summary={k:v for k,v in roster.items() if k not in ('issues','missing_market_dates')},
        roster_issue_count=len(roster['issues']),complete_verified_data=False,complete_historical_universe=False,
        live_qualified=False,actual_fill_verified=False,unseen_validation=False,return_recomputed=False,
        full_account_replay_required=True,frozen_inputs_changed=False,database_mutations=False,
        network_requests=0,finmind_requests=0)
    (output/'manifest.json').write_text(encode(report))
    (output/'manifest.sha256').write_text(sha(output/'manifest.json')+'\n')
    # Complete source identity check occurs after writing, before returning success.
    require(all(sha(ROOT/name)==digest for name,digest in refs.items()),'Source changed during input rebuild')
    print(encode({k:v for k,v in report.items() if k not in ('source_sha256','files_sha256',
        'median50m_candidate_diff','required_history')}),flush=True)
    print('median50m', {k:v for k,v in diffs['median50m'].items() if not isinstance(v,list)},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.output)
