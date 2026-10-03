#!/usr/bin/env python3
"""Audit raw profile inputs, then test three preregistered pilot comparisons."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_volume_profile import BUNDLE, PREREG, RANKS, local, validate_cohort, verify_saved, write
from scripts.export_signal_explorer import verify_evidence
from scripts.research_early_signal_losses import digest, evaluate_filter, statistics, PERIODS
from scripts.research_staged_entry_20261003 import _load_paths
from skills.board_tape_reconciliation import summarize_ticks, reconcile
from skills.intraday_limit_replay import normalize_ticks
from skills.independent_three_black import path_issue, black_at_close
from skills.independent_signals import net_unit_return
from skills.exit_policy import decide_exit
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.trial_registry import append_trial_registry
from skills.volume_profile import build_volume_profile

RANK_DIR = RANKS.parent
BASELINE = ROOT/'.cache/all-signals-2019-20261002/research-v1'
OFFICIAL = ROOT/'.cache/market-input-repair-20261002/quote-evidence/official-sources.json'
INVENTORY = ROOT/'.cache/volume-profile-20261003/inventory'
TAPES = ROOT/'.cache/volume-profile-20261003/tapes-v1'


def constant_scale(adjusted, raw):
    """All rounded prices must admit one common positive adjustment factor."""
    a, r = np.asarray(adjusted, float), np.asarray(raw, float)
    if len(a) != len(r) or not len(a) or not (np.isfinite(a)&np.isfinite(r)&(a>.005)&(r>.005)).all():
        return False
    return bool(np.max((a-.005)/(r+.005)) <= np.min((a+.005)/(r-.005)))


def audit_day(raw, sid, day, official, exact=None):
    """Provider prints are not exchange-verified by a loose all-session bound."""
    market = official['market']
    tape = summarize_ticks(raw, sid, day, market)
    out = dict(stock_id=sid, date=day, market=market, tape=tape,
               official_volume=float(official['volume']), volume_scope=official['volume_scope'],
               official_source_id=official['source_id'], status='usable_provider_diagnostic',
               ordinary_volume_matched=False, ordinary_amount_matched=False, tick_sequence_complete=False)
    if official.get('table_category') == '管理股票':
        return out | dict(status='managed_board')
    if tape['unknown_session_rows'] or tape['shares'] <= 0:
        return out | dict(status='unknown_session_or_no_regular_volume')
    for key in ('open', 'high', 'low', 'close'):
        value = official[key]
        if not np.isfinite(value) or value <= 0 or abs(tape[key+'_cents']/100-value) > 1e-6:
            return out | dict(status='official_ohlc_conflict')
    volume = official['volume']
    if not np.isfinite(volume) or volume <= 0:
        return out | dict(status='official_volume_missing')
    if exact is not None:
        if (exact['stock_id'],exact['date'],exact['market']) != (sid,day,market):
            raise ValueError('Ordinary reference identity mismatch')
        match=reconcile(tape,exact)
        if not match['same_scope_aggregate_matched']:
            return out | dict(status='official_ordinary_aggregate_conflict', differences=match['differences'])
        return out | dict(status='usable_ordinary_daily_matched',ordinary_volume_matched=True,ordinary_amount_matched=True)
    if official['volume_scope'] == 'ordinary_session':
        if tape['shares'] != volume:
            return out | dict(status='official_ordinary_volume_conflict')
        return out | dict(status='usable_ordinary_daily_matched', ordinary_volume_matched=True)
    if official['volume_scope'] != 'all_daily_sessions':
        return out | dict(status='official_volume_scope_unknown')
    if tape['shares']+tape['fixed_price_shares'] > volume:
        return out | dict(status='exceeds_official_all_session_volume')
    return out


def early_exit(row, path, feature):
    """Frozen VAH: close trigger, following session HL2; original exits win ties."""
    result = dict(row)
    def unknown(reason):
        out=dict(row,status='unresolved',outcome='unknown',data_issue=reason,
                 exit_reason=None,exit_date=None,exit_trigger_date=None)
        for key in ('net_return','gross_return','unrealized_net_return','exit_price','adjusted_end_price','mark_price',
                    'exit_volume','exit_single_price',
                    'holding_days','holding_days_inclusive','calendar_days','mfe','mfe_date',
                    'peak_close_return','peak_close_date','confirmed_high_return','confirmed_high_date'):
            out[key]=None
        return out
    if feature['status'] != 'known':
        return unknown('volume_profile_unavailable')
    if not feature['above_vah'] or not row.get('entry_date'):
        return result
    start = int(path.days.get_loc(pd.Timestamp(row['entry_date'])))
    last = len(path.days)-1
    original_trigger = (int(path.days.get_loc(pd.Timestamp(row['exit_trigger_date'])))
                        if row.get('exit_trigger_date') else last+1)
    # A support decision is evaluated sequentially, with no peek at a later low.
    for j in range(start, min(last, original_trigger-1)+1):
        if path_issue(path, start, j):
            return unknown('volume_profile_decision_path_invalid')
        prior=decide_exit(dict(held_sessions=j+1-start,has_signal=True,
            entry_return=float(path.close[j]/path.close[start]-1),peak_return=None,
            peak_drawdown=None,relative20=None,below_ma20_two=False,
            market_off_two=False,strong_trend=False),'loss12')
        if prior['exit'] or black_at_close(path,start,j):
            # Even an unfilled original stop has priority. Do not walk past it
            # and manufacture a later successful support exit.
            return result
        if path.close[j] >= feature['vah_adjusted']: continue
        if j == last:
            return result | dict(status='pending_exit', outcome='unrealized', net_return=None,
                exit_reason='volume_value_area_failure', exit_trigger_date=str(path.days[j].date()), exit_date=None)
        end = j+1
        if path_issue(path, start, end) or path.volume[end] <= 0:
            return unknown('volume_profile_exit_path_invalid')
        entry = (path.high[start]+path.low[start])/2*path.close[start]/path.raw_close[start]
        raw_exit = float((path.high[end]+path.low[end])/2)
        end_price = raw_exit*path.close[end]/path.raw_close[end]
        net = float(net_unit_return(end_price/entry))
        result.update(status='closed', outcome='profit' if net>0 else 'loss' if net<0 else 'flat',
            exit_reason='volume_value_area_failure', exit_trigger_date=str(path.days[j].date()),
            exit_date=str(path.days[end].date()), exit_price=raw_exit, adjusted_end_price=float(end_price),
            exit_volume=float(path.volume[end]),exit_single_price=bool(path.high[end]==path.low[end]),
            unrealized_net_return=None,mark_price=None,
            net_return=net, gross_return=float(end_price/entry-1), holding_days=end-start,
            holding_days_inclusive=end-start+1, calendar_days=int((path.days[end]-path.days[start]).days),
            data_issue=None, observed_end_date=str(path.days[end].date()))
        # Old post-exit extrema cannot survive an earlier exit as holding metrics.
        for key in ('mfe','mfe_date','peak_close_return','peak_close_date','confirmed_high_return','confirmed_high_date'):
            result[key] = None
        return result
    return result


def paired(originals, changed):
    lookup = {r['signal_id']:r for r in changed}
    if len(lookup)!=len(changed) or set(lookup)!={r['signal_id'] for r in originals}:
        raise ValueError('Paired outcome identities differ')
    common = [r for r in originals if r['status']=='closed' and lookup[r['signal_id']]['status']=='closed']
    new = [lookup[r['signal_id']] for r in common]
    big = [r for r in common if r['net_return']>=.30]
    return dict(baseline=statistics(common), changed=statistics(new),
        original_count=len(originals), common_closed=len(common),
        unpaired_statuses=dict(Counter(lookup[r['signal_id']]['status'] for r in originals if r not in common)),
        mean_difference=float(np.mean([b['net_return']-a['net_return'] for a,b in zip(common,new)])) if common else None,
        changed_exit_count=sum(a['exit_date']!=b['exit_date'] for a,b in zip(common,new)),
        winners_helped=sum(b['net_return']>a['net_return'] for a,b in zip(common,new)),
        winners_hurt=sum(b['net_return']<a['net_return'] for a,b in zip(common,new)),
        return30_retention=sum(lookup[r['signal_id']]['net_return']>=.30 for r in big)/len(big) if big else None)


def run(output, record=False):
    output=local(output)
    if output.exists(): raise ValueError('Choose new output to retain every experiment')
    # Validate publication prerequisites before reading any outcome columns.
    for name in ('tests/test_volume_profile.py','tests/test_volume_profile_research.py'):
        if not (ROOT/name).is_file():raise ValueError('Missing research verification source: '+name)
    refs, manifest = verify_evidence(BUNDLE, RANK_DIR)
    cohort_file=INVENTORY/'cohort.json'
    cohort=json.loads(cohort_file.read_text())
    plan=json.loads((TAPES/'plan.json').read_text())
    acquisition=json.loads((TAPES/'report.json').read_text())
    if acquisition['plan_sha256']!=digest(TAPES/'plan.json') or acquisition['script_sha256']!=digest(ROOT/'scripts/prepare_volume_profile.py'):
        raise ValueError('Acquisition source or plan changed')
    if digest(cohort_file)!=plan['cohort_sha256'] or digest(PREREG)!=plan['prereg_sha256']:
        raise ValueError('Acquisition preregistration or cohort changed')
    if digest(INVENTORY/'reuse-index.json')!=plan['reuse_index_sha256']:
        raise ValueError('Acquisition reuse inventory changed')
    ids=sorted({r['stock_id'] for r in cohort['rows']})
    paths=_load_paths(BUNDLE, ids)
    days=next(iter(paths.values())).days
    ranks=pd.read_parquet(RANKS,columns=['signal_id','stock_id','signal_date'])
    coordinates=validate_cohort(cohort,ranks,days.strftime('%Y-%m-%d').tolist())
    if plan['coordinates'] != [list(k) for k in coordinates]: raise ValueError('Acquisition footprint changed')
    official_meta=json.loads(OFFICIAL.read_text())
    if digest(OFFICIAL)!=OFFICIAL.with_suffix('.sha256').read_text().strip():
        raise ValueError('Official source receipt changed')
    normalized=OFFICIAL.parent/'official-normalized.parquet'
    if digest(normalized)!=official_meta['output_sha256'][str(normalized.relative_to(ROOT))]:
        raise ValueError('Official normalized table changed')
    official=pd.read_parquet(normalized)
    official=official.loc[official.stock_id.isin(ids)]
    official_index={k:g for k,g in official.groupby(['stock_id','date'],sort=False)}
    peer=INVENTORY.parent/'peer-audit'
    exact_refs=json.loads((peer/'source-hashes.json').read_text())
    peer_manifest=json.loads((peer/'manifest.json').read_text())
    if peer_manifest['files_sha256']['ordinary-exact-reference.json']!='7660401aea0b46f615c48bd3c83fc787407d5c340a36525e9a4c187504b2ab9f':
        raise ValueError('Unexpected reviewed ordinary reference')
    for name,expected in peer_manifest['files_sha256'].items():
        if digest(local(peer/name))!=expected:raise ValueError('Peer audit output changed')
    for name,expected in exact_refs.items():
        p=local(ROOT/name)
        if digest(p)!=expected:raise ValueError('Ordinary aggregate source changed')
        refs[str(p.relative_to(ROOT))]=expected
    for p in peer.iterdir():
        if p.suffix in ('.json','.py'):refs[str(p.relative_to(ROOT))]=digest(p)
    exact_rows=json.loads((peer/'ordinary-exact-reference.json').read_text())
    exact_index={(r['stock_id'],r['date'],r['market']):r for r in exact_rows}
    if len(exact_index)!=len(exact_rows):raise ValueError('Duplicate ordinary aggregate reference')
    for p in (cohort_file,INVENTORY/'reuse-index.json',TAPES/'plan.json',TAPES/'report.json',OFFICIAL,OFFICIAL.with_suffix('.sha256'),normalized,PREREG):
        refs[str(p.relative_to(ROOT))]=digest(p)
    daily,selected_ticks,used_sources=[],{},set()
    for sid,day in coordinates:
        receipt=TAPES/'receipts'/(sid+'-'+day+'.json')
        if not receipt.exists():
            daily.append(dict(stock_id=sid,date=day,status='not_requested')); continue
        if acquisition['receipt_sha256'].get(str(receipt.relative_to(ROOT)))!=digest(receipt):
            raise ValueError('Acquisition receipt changed')
        refs[str(receipt.relative_to(ROOT))]=digest(receipt)
        item=verify_saved(json.loads(receipt.read_text()),dict(dataset='TaiwanStockPriceTick',data_id=sid,start_date=day))
        if item['status'] not in ('cached','received'):
            daily.append(dict(stock_id=sid,date=day,status=item['status']));continue
        p=local(item['raw_path']);refs[str(p.relative_to(ROOT))]=item['raw_sha256']
        if item.get('metadata_path'):
            mp=local(item['metadata_path'])
            if digest(mp)!=item['metadata_sha256']:raise ValueError('Cached metadata changed')
            refs[str(mp.relative_to(ROOT))]=item['metadata_sha256']
        o=official_index.get((sid,day))
        if o is None or len(o)!=1:
            daily.append(dict(stock_id=sid,date=day,status='official_identity_missing_or_ambiguous'));continue
        source=o.iloc[0].to_dict();used_sources.add(source['source_id'])
        raw=pd.read_parquet(p)
        try:
            checked=audit_day(raw,sid,day,source,exact_index.get((sid,day,source['market'])))
        except (ValueError,KeyError,TypeError,OverflowError,ReplayDataUnavailable):
            checked=dict(stock_id=sid,date=day,status='invalid_provider_tape')
        daily.append(checked)
        if checked['status'].startswith('usable_'):
            ticks=normalize_ticks(raw,sid,day,source['market'])
            regular=ticks.time.ge(pd.Timedelta('09:00:00')) & ticks.time.lt(pd.Timedelta('13:34:00'))
            ticks=ticks.loc[regular & ticks.shares.gt(0)].copy()
            ticks['timestamp']=pd.Timestamp(day)+ticks.time
            selected_ticks[(sid,day)]=ticks[['timestamp','price','shares']]
    for key in used_sources:
        source=official_meta['sources'][key]
        for pk,hk in (('path','sha256'),('receipt','receipt_sha256')):
            p=local(source[pk])
            if digest(p)!=source[hk]:raise ValueError('Used official source changed')
            refs[str(p.relative_to(ROOT))]=source[hk]
    audit={(r['stock_id'],r['date']):r for r in daily}
    events=pd.read_parquet(BUNDLE/'events.parquet');events['event_date']=pd.to_datetime(events.event_date)
    features=[]
    for row in cohort['rows']:
        sid,day=row['stock_id'],row['signal_date'];path=paths[sid];i=int(days.get_loc(pd.Timestamp(day)))
        f=dict(signal_id=row['signal_id'],stock_id=sid,signal_date=day,status='unknown',
            above_vah=None,poc_up=None,ordinary_daily_matched=False,issues=[])
        prior=row['prior_dates'];checks=[audit[(sid,d)] for d in prior]
        issues=[r['status'] for r in checks if not r['status'].startswith('usable_')]
        if path_issue(path,i-20,i):issues.append('pre_signal_daily_path_invalid')
        actions=events.loc[events.stock_id.eq(sid)&events.event_date.between(prior[0],day)]
        if len(actions) or not constant_scale(path.close[i-20:i+1],path.raw_close[i-20:i+1]):
            issues.append('corporate_action_or_nonconstant_price_scale')
        f['issues']=sorted(set(issues))
        if not issues:
            ticks=pd.concat([selected_ticks[(sid,d)] for d in prior],ignore_index=True)
            profile=build_volume_profile(ticks,signal_date=day,session_dates=prior,bins=40,
                value_fraction=.70,source_kind='authentic_regular_board_trade_ticks')
            if profile['status']!='available':
                f['issues']=['profile_unavailable']
            else:
                f.update(status='known', profile=profile,
                    above_vah=bool(path.raw_close[i]>profile['full']['vah']),
                    poc_up=profile['poc_up'],
                    vah_adjusted=float(profile['full']['vah']*path.close[i]/path.raw_close[i]),
                    ordinary_daily_matched=all(c['ordinary_volume_matched'] and c['ordinary_amount_matched'] for c in checks))
        features.append(f)
    # Outcome columns are joined only after every profile and decision is frozen.
    payload=json.loads((BASELINE/'workbook-data.json').read_text())
    feature_map={r['signal_id']:r for r in features}
    original=[r for r in payload['rows'] if r['signal_id'] in feature_map]
    if len(original)!=160:raise ValueError('Pilot outcomes do not match all cohort identities')
    enriched=[dict(r,above_vah=feature_map[r['signal_id']]['above_vah'],poc_up=feature_map[r['signal_id']]['poc_up']) for r in original]
    changed=[early_exit(r,paths[r['stock_id']],feature_map[r['signal_id']]) for r in original]
    scopes={'all':lambda r:True}
    scopes.update({str(y):lambda r,y=y:r['signal_date'].startswith(str(y)) for y in range(2019,2027)})
    scopes.update({p:lambda r,lo=lo,hi=hi:lo<=r['signal_date']<=hi for p,(lo,hi) in PERIODS.items()})
    results={}
    for name,keep in scopes.items():
        group=[r for r in enriched if keep(r)]
        known=[r for r in group if feature_map[r['signal_id']]['status']=='known']
        matched=[r for r in known if feature_map[r['signal_id']]['ordinary_daily_matched']]
        results[name]=dict(full_cohort=statistics(group),profile_known=statistics(known),
            ordinary_daily_matched=statistics(matched),
            filters={field:evaluate_filter(known,field) for field in ('above_vah','poc_up')},
            all_cohort_filters={field:evaluate_filter(group,field) for field in ('above_vah','poc_up')},
            ordinary_matched_filters={field:evaluate_filter(matched,field) for field in ('above_vah','poc_up')},
            ordinary_matched_support_exit=paired(matched,[r for r in changed if r['signal_id'] in {z['signal_id'] for z in matched}]),
            support_exit=paired([r for r in original if keep(r)],[r for r in changed if keep(r)]))
    output.mkdir(parents=True)
    write(output/'features.json',features);write(output/'daily-audit.json',daily)
    write(output/'outcomes.json',dict(baseline=enriched,support_exit=changed))
    write(output/'comparisons.json',results)
    trial_rows=[]
    for arm in ('baseline','above_vah','poc_up','support_exit'):
        trial=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='volume_profile_pilot_20261003',
            command=' '.join(sys.argv),status='completed',sharpe=None,params=dict(arm=arm,sample=160,
            lookback=20,bins=40,value_fraction=.70,unseen_validation=False,cash_account=False),
            prereg_sha256=digest(PREREG),result=results['all']['support_exit'] if arm=='support_exit'
            else results['all']['filters'].get(arm,results['all']['full_cohort']))
        if record:trial['registry_row']=append_trial_registry(trial)
        trial_rows.append(trial)
    write(output/'trials.json',trial_rows)
    for p in (Path(__file__),ROOT/'scripts/prepare_volume_profile.py',ROOT/'skills/volume_profile.py',
              ROOT/'skills/board_tape_reconciliation.py',ROOT/'skills/intraday_limit_replay.py',
              ROOT/'scripts/research_staged_entry_20261003.py',ROOT/'skills/independent_three_black.py',
              ROOT/'skills/independent_signals.py',ROOT/'skills/exit_policy.py',ROOT/'skills/trial_registry.py',
              ROOT/'tests/test_volume_profile.py',ROOT/'tests/test_volume_profile_research.py'):
        refs[str(p.relative_to(ROOT))]=digest(p)
    report=dict(schema='volume_profile_pilot_v1',sample_count=160,
        acquisition=dict(required_stock_days=acquisition['required_stock_days'],adapter_attempts=acquisition['adapter_attempts'],
                         raw_rows=acquisition['raw_rows'],counts=acquisition['counts']),
        known_profiles=sum(r['status']=='known' for r in features),
        ordinary_daily_matched_profiles=sum(r['ordinary_daily_matched'] for r in features),
        feature_issues=dict(Counter(i for r in features for i in r['issues'])),
        daily_statuses=dict(Counter(r['status'] for r in daily)),results=results,
        source_sha256=refs,output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        live_qualified=False,unseen_validation=False,cash_account=False,
        all_signal_intraday_complete=False,tick_sequence_complete=False,registry_recorded=record,
        limitations=['Stratified 160-signal pilot; not the full 30188-signal universe.',
            'Previously researched periods, overlapping independent opportunities; no account NAV or account drawdown.',
            'HL2 is an end-of-day hindsight fill assumption, not an executable opening price.',
            'TWSE all-session volume only bounds regular volume; unmatched profiles are provider diagnostics.',
            'Matching OHLC and volume does not certify full tick sequence or historical available revisions.',
            'No use of this indicator to infer owner holdings, big-money costs, or future profits.'])
    write(output/'report.json',report);(output/'report.sha256').write_text(digest(output/'report.json')+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','output_sha256','results')},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record-trials',action='store_true')
    args=parser.parse_args()
    try:
        run(args.output,args.record_trials)
    except Exception as exc:
        failure=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='volume_profile_pilot_20261003',
            command=' '.join(sys.argv),status='failed',sharpe=None,error_type=type(exc).__name__,
            params=dict(arms=['baseline','above_vah','poc_up','support_exit'],cash_account=False),
            output=str(args.output),results_must_not_be_promoted=True)
        if args.record_trials:failure['registry_row']=append_trial_registry(failure)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        write(local(args.output).parent/('failed-attempt-'+stamp+'.json'),failure)
        raise
