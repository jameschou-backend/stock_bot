#!/usr/bin/env python3
"""Re-audit frozen inputs with separately hash-bound official supplements.

This v2 entrypoint preserves the original sealed auditor and strategy files.
It reads local sources only; it does not download, filter candidates or trade.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from itertools import islice
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.market_input_validation import (parse_market_day, resolve_episode, excluded,
    require, require_complete)
from skills.official_market_supplement import (collect_supplement, compare_quote, execution_scope)
from skills.official_market_classification import load_nonordinary_overlay, resolve_nonordinary
from scripts.validate_three_black_market_inputs import collect_sources

BASE = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
RUN = ROOT/'.cache/three-black-20261001/final-c'
SIGNALS = ROOT/'.cache/liquidity-universe-20261001/signals-v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'


def coverage(keys, verified, day_index, id_index):
    keys = set(keys)
    n = sum(bool(verified[day_index[d], id_index[s]]) for d,s in keys
            if d in day_index and s in id_index)
    return dict(required=len(keys),verified=n,missing=len(keys)-n,
                fraction=n/len(keys) if keys else None,complete=bool(keys) and n==len(keys))


def iter_source_quotes(official, quote_index, quote_columns, batch_size=4096):
    """Batch the lookup only; preserve source order and original numeric values."""
    entries = iter(official.items())
    while batch := list(islice(entries, batch_size)):
        locations = quote_index.get_indexer([(stamp,sid) for (market,stamp,sid),row in batch])
        for (key, source), location in zip(batch, locations):
            # A mixed numeric pandas row coerced share volume to float in v1.
            # Keep that exact representation, without allocating a Series/row.
            local = None if location < 0 else {k:float(v[location]) for k,v in quote_columns.items()}
            yield key, source, local


def build(supplement_manifest, repair_bundle=None, classification_overlay=None):
    seal = read(RUN/'report.json')
    case_path = RUN/'three_black.json'
    require(sha(case_path) == seal['cases']['three_black']['sha256'], 'Account source changed')
    case = read(case_path)
    a = case['account']
    refs = {str((RUN/'report.json').relative_to(ROOT)):sha(RUN/'report.json'),
            str(case_path.relative_to(ROOT)):sha(case_path)}
    classifications = (load_nonordinary_overlay(classification_overlay, ROOT, refs)
                       if classification_overlay is not None else [])
    for p in [BASE/'quotes-unmasked.parquet', BASE/'eligibility.parquet',BASE/'companies.parquet',
              BASE/'identity.json',SIGNALS]:
        name = str(p.relative_to(ROOT))
        require(sha(p) == seal['source_sha256'][name], 'Changed frozen input: '+name)
        refs[name] = seal['source_sha256'][name]
    identity = read(BASE/'identity.json')
    episodes, exclusions = identity['episodes'], list(identity['trading_exclusions'])
    for name in ('stock_universe_2019_corporate_terms','stock_universe_five_corporate_terms'):
        path = ROOT/'docs'/(name+'.json')
        relative = str(path.relative_to(ROOT))
        require(sha(path) == seal['source_sha256'][relative], 'Changed corporate exclusion evidence')
        refs[relative] = sha(path)
        exclusions.extend(read(path).get('verified_halts', []))
    by_sid = defaultdict(list)
    for e in episodes:
        by_sid[e['stock_id']].append(e)
    print('Normalize hash-bound official daily tables', flush=True)
    official, sources, rejected = collect_sources(seal['source_sha256'], refs)
    original_source_days = {(s['market'], s['date']) for s in sources}
    official, supplemental_sources = collect_supplement(supplement_manifest, ROOT, refs, official)
    sources.extend(supplemental_sources)
    repair_summary = None
    if repair_bundle is not None:
        repair_path = Path(repair_bundle).resolve()/'report.json'
        require(repair_path.is_relative_to(ROOT), 'Repair bundle escapes repository')
        require(sha(repair_path) == repair_path.with_suffix('.sha256').read_text().strip(), 'Repair report changed')
        repair = read(repair_path)
        require(repair['schema'] == 'market_input_repairs_v1' and repair['live_qualified'] is False,
                'Unsupported repair evidence')
        for name,expected in {**repair['source_sha256'],**repair['output_sha256']}.items():
            require((ROOT/name).resolve().is_relative_to(ROOT) and sha(ROOT/name) == expected, 'Repair source changed: '+name)
            refs[name] = expected
        refs[str(repair_path.relative_to(ROOT))] = sha(repair_path)
        quotes = pd.read_parquet(repair_path.parent/'quotes-unmasked.parquet')
        repair_summary = {k:repair[k] for k in ('missing_quotes_repaired','provider_requests',
            'original_candidate_count','repaired_candidate_count','added_candidates','removed_candidates',
            'changed_candidates','earlier_candidates_unchanged','affected_holdings_after_repair','full_account_replay_required')}
    else:
        quotes = pd.read_parquet(BASE/'quotes-unmasked.parquet')
    # Format each distinct calendar value once and share those immutable
    # strings. Formatting millions of repeated dates separately costs memory.
    date_codes, date_values = pd.factorize(quotes.date,sort=False,use_na_sentinel=False)
    date_labels = pd.Series(date_values).astype(str).to_numpy()
    quotes.date = date_labels[date_codes]
    del date_codes, date_values, date_labels
    require(not quotes.duplicated(['date','stock_id']).any(), 'Duplicate local quote')
    quote_index = pd.MultiIndex.from_arrays([quotes.date,quotes.stock_id],names=['date','stock_id'])
    quote_columns = {k:quotes[k].to_numpy(copy=False) for k in ('open','high','low','close','volume')}
    eligibility = pd.read_parquet(BASE/'eligibility.parquet').set_index('date')
    eligibility.index = pd.to_datetime(eligibility.index).strftime('%Y-%m-%d')
    days, ids = list(eligibility.index),list(eligibility.columns)
    require(days == sorted(set(days)), 'Invalid input calendar')
    day_index, id_index = {d:i for i,d in enumerate(days)},{s:i for i,s in enumerate(ids)}
    counts, field_counts = Counter(),defaultdict(Counter)
    differences, absent_positive, identity_issues, no_quote = [], [], [], []
    # Matrix axes are already validated/frozen above. Compact masks avoid
    # millions of duplicate tuple objects without changing any coverage gate.
    verified = np.zeros(eligibility.shape,dtype=bool)
    verified_closes = np.zeros(eligibility.shape,dtype=bool)
    verified_totals = np.zeros(eligibility.shape,dtype=bool)
    present = defaultdict(set)
    ordinary_seen = set()
    for (market, stamp, sid), source, local in iter_source_quotes(official,quote_index,quote_columns):
        if sid.startswith('0') and sid != '0050':
            continue
        episode = resolve_episode(by_sid[sid],sid,stamp)
        nonordinary = resolve_nonordinary(classifications,sid,stamp,market)
        if nonordinary is not None:
            require(episode is None or episode['category'] in ('TDR','臺灣存託憑證','臺灣存託憑證(TDR)'),
                    'Official nonordinary classification conflicts with historical identity')
            counts['identified_nonordinary_rows'] += 1
            counts['supplemental_nonordinary_rows'] += 1
            continue
        if source.get('table_category') == '管理股票':
            counts['managed_table_rows'] += 1
            if episode and not excluded(exclusions,sid,stamp,market):
                identity_issues.append(dict(stock_id=sid,date=stamp,market=market,issue='managed_board_without_exclusion'))
            continue
        if episode is None or episode['market'].upper() != market:
            identity_issues.append(dict(stock_id=sid,date=stamp,market=market,issue='official_stock_absent_from_dated_identity'))
        elif episode['category'] != '股票' and sid != '0050':
            counts['identified_nonordinary_rows'] += 1
            continue
        key = stamp,sid
        if sid != '0050':
            ordinary_seen.add(sid)
        present[(market,stamp)].add(sid)
        count_name = 'official_benchmark_rows' if sid == '0050' else (
            'official_ordinary_rows' if episode else 'official_unclassified_rows')
        counts[count_name] += 1
        if local is None:
            positive = any(source[f] is not None for f in ('open','high','low','close'))
            no_quote.append(dict(stock_id=sid,date=stamp,market=market,positive_official_price=positive,
                                 source_volume=source['volume']))
            counts['local_quote_missing'] += 1
            counts['positive_local_quote_missing'] += positive
            continue
        result = compare_quote(local,source)
        counts['local_rows_compared'] += 1
        for field,status in result.items():
            field_counts[field][status] += 1
        if stamp in day_index and sid in id_index:
            i,j = day_index[stamp],id_index[sid]
            verified[i,j] |= all(result[f] == 'matched' for f in ('open','high','low','close'))
            verified_closes[i,j] |= result['close'] == 'matched'
            verified_totals[i,j] |= result['total_volume'] == 'matched'
        conflicts = [f for f,s in result.items() if s == 'conflict']
        if conflicts:
            differences.append(dict(stock_id=sid,date=stamp,market=market,fields=conflicts,
                local={k:local['volume' if k=='total_volume' else k] for k in conflicts},
                official={k:source['volume' if k=='total_volume' else k] for k in conflicts}))
    print('Compare all candidates and known historical listing intervals', flush=True)
    candidates = read(SIGNALS)['entries']['median50m']
    candidate_issues, history = [],np.zeros(eligibility.shape,dtype=bool)
    for e in candidates:
        sid,stamp = e['members'][0],e['signal_date']
        episode = resolve_episode(by_sid[sid],sid,stamp)
        reason = None
        if not episode or episode['category'] != '股票':
            reason = 'missing_or_nonordinary_identity'
        elif resolve_nonordinary(classifications,sid,stamp,episode['market'].upper()) is not None:
            reason = 'official_nonordinary_classification'
        elif excluded(exclusions,sid,stamp,episode['market'].upper()):
            reason = 'known_trading_exclusion'
        elif not eligibility.at[stamp,sid]:
            reason = 'signal_outside_frozen_eligibility'
        if reason:
            candidate_issues.append(dict(stock_id=sid,date=stamp,event_id=e['event_id'],reason=reason))
        i,j = day_index[stamp],id_index[sid]
        require(i >= 126, 'Candidate lacks 126 prior sessions')
        history[i-126:i+1,j] = True
        history[i-126:i+1,id_index['0050']] = True
    # Listing masks may legitimately contain missing-price or nontrading rows;
    # report those independently instead of assuming a missing quote delisted it.
    legal = np.zeros(eligibility.shape,dtype=bool)
    known_ids = set()
    for e in episodes:
        if e['category'] not in ('股票','ETF') or e['stock_id'] not in id_index or not e['start']:
            continue
        sid,j = e['stock_id'],id_index[e['stock_id']]
        known_ids.add(sid)
        legal[:,j] |= np.array([e['start'] <= d and (not e['end'] or d < e['end']) for d in days])
    bad_mask = eligibility.to_numpy(dtype=bool) & ~legal
    known_omitted = [e for e in episodes if e['category']=='股票' and e.get('start')
                     and e['start'] <= case['summary']['end']
                     and (not e['end'] or e['end'] > case['summary']['start']) and e['stock_id'] not in id_index]
    source_days = {(s['market'],s['date']) for s in sources}
    positive_by_day = (quotes.loc[quotes.close.gt(0) & quotes.volume.gt(0),['date','stock_id']]
        .groupby('date',sort=False)['stock_id'].agg(list).to_dict())
    # Detect positive quotes not present in a full official source-day table.
    for market,stamp in source_days:
        if stamp not in days:
            continue
        for sid in positive_by_day.get(stamp,[]):
            ep = resolve_episode(by_sid[sid],sid,stamp)
            if (ep and ep['category']=='股票' and ep['market'].upper()==market
                    and sid not in present[(market,stamp)]):
                absent_positive.append(dict(stock_id=sid,date=stamp,market=market))
    required_price_count = int(history.sum())
    signal_price_count = int((history & verified_closes).sum())
    signal_total_count = int((history & verified_totals).sum())
    # Benchmark observations participate in the required history coverage too.
    required_dates = {days[i] for i in np.flatnonzero(history.any(axis=1))}
    required_dates.update(d for d in days if case['summary']['start'] <= d <= case['summary']['end'])
    scoped_identity_issues = [r for r in identity_issues if r['date'] in required_dates]
    scoped_positive_missing = [r for r in no_quote if r['date'] in required_dates and r['positive_official_price']]
    request_plan = [dict(market=m,date=d,scope='full_daily_prices_and_roster',
                         status='cached' if (m,d) in source_days else 'source_missing')
                    for d in sorted(required_dates) for m in ('TWSE','TPEX')]
    account_keys = {(r['date'],r['stock_id']) for r in a['trades']}
    board_keys = {(r['date'],r['stock_id']) for r in a['trades'] if r['channel']=='board'}
    holding_keys = {(r['mark_date'],r['stock_id']) for r in a['holdings']}
    # Only ordinary fills use these price envelopes. Odd-lot fills use separate
    # official channels and are not incorrectly certified from ordinary bars.
    capacities = execution_scope(a,official,days,episodes,exclusions)
    eligibility_issues = [dict(date=days[i],stock_id=ids[j]) for i,j in np.argwhere(bad_mask)]
    result = dict(schema='market_input_validation_v2',created_at=datetime.now(timezone.utc).isoformat(),
        study='three_black_20261001',start=case['summary']['start'],end=case['summary']['end'],
        historical_return=case['summary']['total_return'],return_recomputed=False,repair_summary=repair_summary,
        sources=sources,source_days=len(source_days),source_rejections=rejected,
        supplement=dict(manifest=str(Path(supplement_manifest).resolve().relative_to(ROOT)),
            added_source_days=len(source_days-original_source_days),
            source_count=len(supplemental_sources),
            legacy_status_unknown=sum(s.get('http_status_evidence')=='legacy_status_unknown' for s in supplemental_sources)),
        classification_overlay=(dict(manifest=str(Path(classification_overlay).resolve().relative_to(ROOT)),
            entries=len(classifications),observed_rows=counts['supplemental_nonordinary_rows'],
            changes_strategy_universe=False) if classification_overlay is not None else None),
        counts=dict(counts),field_checks={k:dict(v) for k,v in field_counts.items()},
        price_conflicts=differences,local_quote_missing=no_quote,
        positive_local_absent_official=absent_positive,
        coverage=dict(ordinary_fill_prices=coverage(board_keys,verified,day_index,id_index),
            traded_stock_day_prices=coverage(account_keys,verified,day_index,id_index),
            holding_marks=coverage(holding_keys,verified_closes,day_index,id_index),
            candidate_signal_prices=coverage({(e['signal_date'],e['members'][0]) for e in candidates},verified,day_index,id_index),
            signal_history=dict(required=required_price_count,close_verified=int(signal_price_count),
                                total_volume_verified=int(signal_total_count),includes_benchmark=True)),
        identity=dict(known_episodes=len(episodes),terminated_episodes=sum(bool(e.get('end')) for e in episodes),
            candidate_count=len(candidates),candidate_issues=candidate_issues,
            official_presence_issues=identity_issues,eligible_outside_known_listing=eligibility_issues,
            in_scope_official_presence_issues=scoped_identity_issues,
            observations_outside_required_dates=[r for r in identity_issues if r['date'] not in required_dates],
            known_ordinary_episodes_missing_from_frame=known_omitted,
            observed_ordinary_ids=len(ordinary_seen),official_days_by_market=dict(Counter(m for m,d in source_days)),
            basis=identity['historical_membership_basis'],complete_historical_universe=False),
        execution=capacities,request_plan=request_plan,
        requests_lower_bound=sum(r['status']=='source_missing' for r in request_plan),
        request_estimate_note='Market/day price-roster tables only; ordinary-session volume components require additional sources.',
        missing_positive_quotes_in_scope=scoped_positive_missing,
        network_requests=0,finmind_requests=0,
        live_qualified=False,actual_fill_verified=False,unseen_validation=False,
        source_sha256=refs)
    result['checks'] = dict(all_execution_prices_verified=result['coverage']['ordinary_fill_prices']['complete'],
        all_holding_marks_verified=result['coverage']['holding_marks']['complete'],
        all_ordinary_capacity_verified=capacities['all_capacity_verified'],
        all_signal_histories_verified=bool(required_price_count) and signal_price_count==required_price_count
            and signal_total_count==required_price_count,
        all_historical_market_days_observed=all(r['status']=='cached' for r in request_plan),
        known_identity_checks_passed=not(candidate_issues or scoped_identity_issues or eligibility_issues
            or known_omitted or capacities['identity_issues'] or absent_positive),
        all_observed_positive_quotes_present=not scoped_positive_missing,
        complete_historical_universe=identity.get('complete_historical_universe') is True,
        no_observed_price_conflicts=not differences)
    result['complete_verified_data'] = all(result['checks'].values())
    result['research_status'] = 'observed_conflict' if (differences or candidate_issues or eligibility_issues) else 'partial_primary_verification'
    for path in (Path(__file__),ROOT/'skills/market_input_validation.py',
                 ROOT/'skills/official_market_supplement.py',ROOT/'scripts/validate_three_black_market_inputs.py',
                 ROOT/'skills/official_market_classification.py'):
        result['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--repair-bundle',type=Path)
    parser.add_argument('--supplement',type=Path,required=True)
    parser.add_argument('--classification-overlay',type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve published evidence; choose a new output')
    result = build(args.supplement, args.repair_bundle, args.classification_overlay)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(encode(result))
    args.output.with_suffix('.sha256').write_text(sha(args.output)+'\n')
    if args.require_complete:
        require_complete(result)
    print(encode({k:result[k] for k in ('counts','coverage','checks','requests_lower_bound','research_status')}))


if __name__ == '__main__':
    main()
