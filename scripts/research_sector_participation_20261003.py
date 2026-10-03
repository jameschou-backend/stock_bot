#!/usr/bin/env python3
"""One frozen, causal co-movement participation screen; not historical industries."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.research_early_signal_losses import PERIODS, digest, evaluate_filter, statistics, verified_sources
from skills.surge_sector import _exclusive_sums
from skills.trial_registry import append_trial_registry

EXPECTED_PREREG = '7a8247ef6304e684e085d6360d9e2bf2a4649c1b1e84da48f7afebac0576efa1'
EXPECTED_MANIFEST = '6cd6ef3cbf9ebfced4741b2e60fea34212a2dc5605c6b6b5895c003a34bb65bf'
EXPECTED_RANK_REPORT = 'deead8ba56001627867a0af188fa0ad4a2201fbbbac881b2a8f83dae0e6f68bb'
EXPECTED_RANKS = 'e9c3e1826d012bf5c960f53d9ab1fb7a172d39c7a92e3c24d9fc054abd36752a'
PREKNOWN = ['signal_id', 'stock_id', 'signal_date', 'rank_priority', 'rank_score_bin', 'turnover_mean20']
ARM = 'group_participation'


def finite(value):
    return float(value) if np.isfinite(value) else None


def validate_frames(close, raw, volume, eligibility):
    if (not isinstance(close.index, pd.DatetimeIndex) or not close.index.is_unique
            or not close.index.is_monotonic_increasing or not close.columns.is_unique
            or '0050' not in close):
        raise ValueError('Unique ordered calendar, stock columns and0050 required')
    if any(not isinstance(s, str) or len(s) != 4 or not s.isdigit() for s in close.columns):
        raise ValueError('Four-digit stock ids required')
    for frame in (raw, volume, eligibility):
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError('Input axes differ')
    if any(dtype != bool for dtype in eligibility.dtypes):
        raise ValueError('Eligibility must contain explicit booleans')


def monthly_peers(returns, eligible, days, ids, observations):
    """Pairwise correlation uses only120returns ending before the signal month."""
    columns = {sid: i for i, sid in enumerate(ids)}
    first = {}
    for i, day in enumerate(days):
        first.setdefault(str(day.to_period('M')), i)
    targets = defaultdict(set)
    for row in observations:
        targets[row['signal_date'][:7]].add(row['stock_id'])
    records = []
    for month, symbols in sorted(targets.items()):
        cutoff = first[month]-1
        selected_days = returns[max(0, cutoff-119):cutoff+1]
        for sid in sorted(symbols):
            j = columns[sid]
            record = dict(month=month, stock_id=sid, cutoff_date=str(days[cutoff].date()) if cutoff >= 0 else None,
                          peer_ids=[], correlations=[], paired_counts=[], issue=None)
            if cutoff < 119 or not eligible[cutoff, j]:
                record['issue'] = 'insufficient_history_or_ineligible_cutoff'
                records.append(record)
                continue
            target = selected_days[:, j]
            valid = np.isfinite(selected_days) & np.isfinite(target[:, None])
            count = valid.sum(axis=0)
            x = np.where(valid, target[:, None], 0.)
            y = np.where(valid, selected_days, 0.)
            safe_n = np.maximum(count, 1)
            sx, sy = x.sum(axis=0), y.sum(axis=0)
            covariance = (x*y).sum(axis=0)-sx*sy/safe_n
            vx = (x*x).sum(axis=0)-sx*sx/safe_n
            vy = (y*y).sum(axis=0)-sy*sy/safe_n
            denominator = np.sqrt(np.maximum(vx, 0)*np.maximum(vy, 0))
            corr = np.divide(covariance, denominator, out=np.full(len(ids), np.nan), where=denominator > 0)
            candidates = [k for k in range(len(ids)) if k != j and ids[k] != '0050'
                          and eligible[cutoff, k] and count[k] >= 100 and np.isfinite(corr[k]) and corr[k] >= .5]
            candidates.sort(key=lambda k: (-float(corr[k]), ids[k]))
            selected = candidates[:10]
            record.update(peer_ids=[ids[k] for k in selected], correlations=[float(corr[k]) for k in selected],
                          paired_counts=[int(count[k]) for k in selected],
                          issue=None if len(selected) >= 3 else 'fewer_than_three_correlated_peers')
            records.append(record)
    return records


def participation_features(close, raw, volume, eligibility, observations):
    """No outcome field is read; candidate itself is excluded from both amounts."""
    validate_frames(close, raw, volume, eligibility)
    keys = [r['signal_id'] for r in observations]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate signal ids')
    days, ids = close.index, list(close.columns)
    cols = {sid: i for i, sid in enumerate(ids)}
    if any(r['stock_id'] not in cols or r['stock_id'] == '0050' or pd.Timestamp(r['signal_date']) not in days for r in observations):
        raise ValueError('Candidate outside frozen calendar/cohort')
    e = eligibility.to_numpy()
    c, p, v = [f.to_numpy(dtype=float) for f in (close, raw, volume)]
    known_price = np.isfinite(c) & (c > 0) & e
    returns = np.full(c.shape, np.nan)
    good = known_price[1:] & known_price[:-1]
    np.divide(c[1:], c[:-1], out=returns[1:], where=good)
    returns[1:] -= 1
    returns[~np.isfinite(returns)] = np.nan
    groups = monthly_peers(returns, e, days, ids, observations)
    lookup = {(r['month'], r['stock_id']): r for r in groups}
    with np.errstate(over='ignore', invalid='ignore'):
        amounts = p*v
    amount_valid = e & np.isfinite(p) & (p > 0) & np.isfinite(v) & (v > 0) & np.isfinite(amounts) & (amounts > 0)
    benchmark = cols['0050']
    market_eligible = e.copy()
    market_eligible[:, benchmark] = False
    market_valid = amount_valid.copy()
    market_valid[:, benchmark] = False
    denominators = _exclusive_sums(np.where(market_valid, amounts, 0.))
    expected = market_eligible.sum(axis=1)[:, None]-market_eligible
    observed = market_valid.sum(axis=1)[:, None]-market_valid
    market_coverage = np.divide(observed, expected, out=np.full(c.shape, np.nan), where=expected > 0)
    output = []
    for source in observations:
        row = {key: source[key] for key in PREKNOWN if key in source}
        day = pd.Timestamp(row['signal_date'])
        i, j = int(days.get_loc(day)), cols[row['stock_id']]
        group = lookup[(row['signal_date'][:7], row['stock_id'])]
        row.update(group_cutoff_date=group['cutoff_date'], peer_ids=group['peer_ids'],
                   expected_peers=len(group['peer_ids']), observed_peers=0, observed_peer_ids=[], peer_coverage=None,
                   peer_above_ma20_fraction=None, peer_share5=None, peer_share_prior20=None, peer_share_multiple=None,
                   market_min_coverage25=None, own_return20=None, peer_median_return20=None, own_minus_peer_return20=None,
                   leader_description=None, breadth_confirmed=None, turnover_confirmed=None,
                   group_participation=None, feature_issue=group['issue'],
                   feature_available_at=row['signal_date']+' after completed close', historical_industry_claimed=False)
        if group['issue'] is not None or i < 24:
            if row['feature_issue'] is None:
                row['feature_issue'] = 'insufficient_twentyfive_day_window'
            output.append(row)
            continue
        peers = np.array([cols[s] for s in group['peer_ids']], dtype=int)
        window = slice(i-24, i+1)
        # Same observed peers for all25days; no missing observation becomes zero.
        valid = (amount_valid[window][:, peers] & known_price[window][:, peers]).all(axis=0)
        common = peers[valid]
        row['observed_peer_ids'] = [ids[k] for k in common]
        row['observed_peers'] = len(common)
        row['peer_coverage'] = len(common)/len(peers)
        cov = market_coverage[window, j]
        row['market_min_coverage25'] = finite(np.min(cov))
        if len(common) < 3 or len(common)/len(peers) < .8:
            row['feature_issue'] = 'insufficient_common_peer_coverage'
            output.append(row)
            continue
        ma = c[i-19:i+1, common].mean(axis=0)
        breadth = float(np.mean(c[i, common] > ma))
        median_return = float(np.median(c[i, common]/c[i-20, common]-1))
        own = float(c[i, j]/c[i-20, j]-1) if known_price[[i-20, i], j].all() else None
        row.update(peer_above_ma20_fraction=breadth, breadth_confirmed=bool(breadth >= .6),
                   peer_median_return20=median_return, own_return20=own,
                   own_minus_peer_return20=own-median_return if own is not None else None,
                   leader_description=('ahead_of_peer_median' if own > median_return else 'at_or_below_peer_median') if own is not None else None)
        denominator = denominators[window, j]
        if not (np.isfinite(cov).all() and (cov >= .95).all() and np.isfinite(denominator).all() and (denominator > 0).all()):
            row['feature_issue'] = 'insufficient_market_amount_coverage'
            output.append(row)
            continue
        shares = amounts[window][:, common].sum(axis=1)/denominator
        recent, prior = float(shares[-5:].mean()), float(shares[:-5].mean())
        multiple = recent/prior
        if not (np.isfinite(multiple) and prior > 0):
            row['feature_issue'] = 'invalid_turnover_share'
            output.append(row)
            continue
        row.update(peer_share5=recent, peer_share_prior20=prior, peer_share_multiple=multiple,
                   turnover_confirmed=bool(multiple >= 1.2), group_participation=bool(breadth >= .6 and multiple >= 1.2))
        output.append(row)
    return output, groups


def turnover_bin(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite T0 turnover required')
    return 'below100m' if value < 100e6 else '100m_to300m' if value < 300e6 else 'at_least300m'


def fixed_pairs(rows):
    """Greedy nearest-score matching before any future outcome is consulted."""
    cells = defaultdict(list)
    excluded = []
    for row in rows:
        if row[ARM] is None:
            excluded.append(dict(signal_id=row['signal_id'], reason='unknown_group_participation'))
            continue
        if type(row[ARM]) is not bool or not math.isfinite(row['rank_priority']):
            raise ValueError('Explicit boolean screen and finite score required')
        key = (row['signal_date'], row['rank_score_bin'], turnover_bin(row['turnover_mean20']))
        cells[key].append(row)
    pairs = []
    for (day, score, turnover), group in sorted(cells.items()):
        yes, no = [r for r in group if r[ARM]], [r for r in group if not r[ARM]]
        candidates = sorted((abs(a['rank_priority']-b['rank_priority']), a['signal_id'], b['signal_id']) for a in yes for b in no)
        used = set()
        for delta, a, b in candidates:
            if a in used or b in used:
                continue
            pairs.append(dict(signal_date=day, rank_score_bin=score, turnover_bin=turnover,
                              pass_signal_id=a, fail_signal_id=b, score_distance=delta))
            used.update((a, b))
        excluded.extend(dict(signal_id=r['signal_id'], reason='no_unmatched_opposite_in_same_cell') for r in group if r['signal_id'] not in used)
    return dict(pairs=pairs, excluded=sorted(excluded, key=lambda r: r['signal_id']))


def scopes(rows):
    result = {'all': rows}
    result.update({str(y): [r for r in rows if r['signal_date'].startswith(str(y))] for y in range(2019, 2027)})
    result.update({name: [r for r in rows if lo <= r['signal_date'] <= hi] for name, (lo, hi) in PERIODS.items()})
    return result


def summarize(rows):
    result = evaluate_filter(rows, ARM)
    closed = [r for r in rows if r['status'] == 'closed']
    known = [r for r in closed if r[ARM] is not None]
    kept = [r for r in known if r[ARM]]
    selected_sum = sum(r['net_return'] for r in kept)
    def mean(group):
        return float(np.mean([r['net_return'] for r in group])) if group else None
    # Unknown screen retains its original outcome for a conservative operational
    # comparison, and separately blocks a claim about the fully observed filter.
    unknown = [r for r in closed if r[ARM] is None]
    practical_sum = selected_sum+sum(r['net_return'] for r in unknown)
    result['opportunities'] = dict(original_closed=len(closed), observable_closed=len(known), unknown_closed=len(unknown),
        selected_closed=len(kept), observable_fraction=len(known)/len(closed) if closed else None,
        original_mean=mean(closed), known_original_mean=mean(known),
        selected_plus_cash_mean_on_known=selected_sum/len(known) if known else None,
        known_opportunity_improvement=(selected_sum/len(known)-mean(known)) if known else None,
        unknown_retains_original_mean=practical_sum/len(closed) if closed else None,
        unknown_retains_original_improvement=(practical_sum/len(closed)-mean(closed)) if closed else None,
        fully_observed_all_original_mean=selected_sum/len(closed) if closed and not unknown else None,
        unknown_not_imputed_zero=True, cash_account=False)
    result['observable_signal_fraction'] = sum(r[ARM] is not None for r in rows)/len(rows) if rows else None
    result['leader_descriptions'] = {name: statistics([r for r in rows if r['leader_description'] == name]) for name in ('ahead_of_peer_median', 'at_or_below_peer_median')}
    result['component_counts_only'] = {field: dict(Counter(str(r[field]) for r in rows)) for field in ('breadth_confirmed', 'turnover_confirmed')}
    return result


def matched_outcomes(pairing, outcomes):
    lookup = {r['signal_id']: r for r in outcomes}
    rows = []
    for pair in pairing['pairs']:
        a, b = lookup[pair['pass_signal_id']], lookup[pair['fail_signal_id']]
        known = a['status'] == b['status'] == 'closed'
        rows.append(dict(pair, pass_status=a['status'], fail_status=b['status'], both_closed=known,
                         pass_return=a['net_return'] if known else None, fail_return=b['net_return'] if known else None,
                         paired_return_difference=a['net_return']-b['net_return'] if known else None,
                         paired_win_difference=int(a['net_return'] > 0)-int(b['net_return'] > 0) if known else None))
    out = {}
    for name, group in scopes(rows).items():
        known = [r for r in group if r['both_closed']]
        dates = defaultdict(list)
        for row in known:
            dates[row['signal_date']].append(row['paired_return_difference'])
        out[name] = dict(fixed_pairs=len(group), both_closed_pairs=len(known), incomplete_pairs=len(group)-len(known),
            mean_paired_return_difference=float(np.mean([r['paired_return_difference'] for r in known])) if known else None,
            median_paired_return_difference=float(np.median([r['paired_return_difference'] for r in known])) if known else None,
            mean_paired_win_difference=float(np.mean([r['paired_win_difference'] for r in known])) if known else None,
            same_day_equal_weight_return_difference=float(np.mean([np.mean(v) for v in dates.values()])) if dates else None,
            matched_dates=len(dates), incomplete_statuses=dict(Counter(r['pass_status']+'|'+r['fail_status'] for r in group if not r['both_closed'])))
    return dict(pairs=rows, scopes=out, excluded_feature_or_matching=pairing['excluded'])


def promotion_gate(results):
    all_result = results['all']
    coverage = all_result['observable_signal_fraction']
    retained = all_result['return30_retention']
    period_improvement = {k: results[k]['opportunities']['known_opportunity_improvement'] for k in PERIODS}
    checks = dict(observable_at_least80=coverage is not None and coverage >= .8,
                  winner30_retention_at_least80=retained is not None and retained >= .8,
                  every_period_opportunity_improves=all(v is not None and v > 0 for v in period_improvement.values()))
    return dict(passed=all(checks.values()), checks=checks, observable_fraction=coverage,
                return30_retention=retained, period_known_opportunity_improvement=period_improvement,
                unknown_policy='unknown screens preserved, never cash0; improvement measured on same observable original opportunities',
                live_qualified=False)


def run(output, prereg, *, record_trials=False):
    started = time.monotonic()
    output, prereg = Path(output).resolve(), Path(prereg).resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError('Require a new repository output directory')
    if not prereg.is_file() or digest(prereg) != EXPECTED_PREREG:
        raise ValueError('Preregistration differs from frozen2a67030')
    bundle = ROOT/'.cache/all-signals-2019-20261002/inputs'
    research = ROOT/'.cache/all-signals-2019-20261002/research-v1'
    rank = ROOT/'.cache/signal-rank-20261003/rank-v2'
    refs, manifest, _ = verified_sources(bundle, research)
    for path, expected in [(bundle/'manifest.json', EXPECTED_MANIFEST), (rank/'report.json', EXPECTED_RANK_REPORT),
                           (rank/'signal-ranks.parquet', EXPECTED_RANKS)]:
        if digest(path) != expected:
            raise ValueError('Frozen source differs: '+str(path))
        refs[str(path.relative_to(ROOT))] = expected
    for name, expected in json.loads((rank/'report.json').read_text())['output_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Rank output differs: '+name)
        refs[name] = expected
    for path in [prereg, Path(__file__).resolve(), ROOT/'tests/test_sector_participation_research.py', ROOT/'skills/surge_sector.py']:
        refs[str(path.relative_to(ROOT))] = digest(path)
    observations = pd.read_parquet(rank/'signal-ranks.parquet', columns=PREKNOWN).to_dict('records')
    if len(observations) != 30188 or len({r['signal_id'] for r in observations}) != 30188:
        raise ValueError('Require original30188signals')
    frames = {name: pd.read_parquet(bundle/(name+'.parquet')).set_index('date')
              for name in ('close-official', 'raw-close', 'raw-volume', 'eligibility')}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    print('Computing frozen monthly peers and T0participation', flush=True)
    features, groups = participation_features(*(frames[n] for n in ('close-official', 'raw-close', 'raw-volume', 'eligibility')), observations)
    pairing = fixed_pairs(features)
    output.mkdir(parents=True)
    def write(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    # These files are materialized before reading future outcomes.
    write('features.json', features)
    write('monthly-peers.json', groups)
    write('fixed-pairs.json', pairing)
    preoutcome_hashes = {name: digest(output/name) for name in ('features.json', 'monthly-peers.json', 'fixed-pairs.json')}
    outcomes = pd.read_parquet(rank/'signal-ranks.parquet').to_dict('records')
    lookup = {r['signal_id']: r for r in features}
    rows = [dict(r, **lookup[r['signal_id']]) for r in outcomes]
    results = {scope: summarize(group) for scope, group in scopes(rows).items()}
    matched = matched_outcomes(pairing, outcomes)
    gate = promotion_gate(results)
    write('comparisons.json', results)
    write('matched-comparisons.json', matched)
    pd.DataFrame(rows).to_parquet(output/'signals.parquet', index=False)
    trials = []
    for arm, value in [('baseline', statistics(rows)), (ARM, results['all']), ('matched_descriptive', matched['scopes']['all'])]:
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), source='sector_participation_20261003', status='completed',
                      command=' '.join(sys.argv), params=dict(arm=arm, fixed_new_strategy_arms=1, cash_account=False, unseen_validation=False),
                      prereg_sha256=EXPECTED_PREREG, source_sha256=refs, result=value, sharpe=None)
        if record_trials:
            record['registry_row'] = append_trial_registry(record)
        trials.append(record)
    write('trials.json', trials)
    for name, expected in refs.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Source changed during research: '+name)
    report = dict(schema='causal_peer_participation_v1', created_at=datetime.now(timezone.utc).isoformat(),
                  sample_count=len(rows), source_sha256=refs, preoutcome_output_sha256=preoutcome_hashes,
                  output_sha256={p.name: digest(p) for p in output.iterdir()},
                  elapsed_seconds=time.monotonic()-started, all_result=results['all'], matched_summary=matched['scopes']['all'],
                  promotion=gate, feature_issue_counts=dict(Counter(r['feature_issue'] or 'known' for r in features)),
                  same_original_outcomes=True, entry_signal_lag_sessions=1, cost_formula_unchanged=True,
                  historical_industry_claimed=False, live_qualified=False, unseen_validation=False, cash_account=False,
                  new_network_requests=0, model_training=0, research_database_writes=0,
                  limitations=manifest['limitations']+[
                      'Causal correlation peers are a price-co-movement proxy, not historical industry or news/theme membership.',
                      'All history has been studied; repeated stocks and overlapping holds are dependent.',
                      'Signal outcomes assume next-session HL2 and original proportionate costs; no cash, slot or execution-depth audit.',
                      'Unknown feature rows remain unknown; full original-denominator filter return is not asserted when any are unknown.',
                      'Matched pairs are fixed before outcomes; only both-closed pairs enter descriptive differences, with incomplete pairs disclosed.'])
    write('report.json', report)
    (output/'report.sha256').write_text(digest(output/'report.json')+'\n')
    print(json.dumps(dict(output=str(output), sample_count=len(rows), promotion=gate,
                          issues=report['feature_issue_counts'], matched=report['matched_summary']), ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prereg', type=Path, default=ROOT/'docs/research_sequential_prereg_20261003.md')
    args = parser.parse_args()
    try:
        run(args.output, args.prereg, record_trials=True)
    except Exception as exc:
        append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(), source='sector_participation_20261003',
            status='failed', command=' '.join(sys.argv), prereg_sha256=EXPECTED_PREREG, error=str(exc), sharpe=None,
            params=dict(arm=ARM, cash_account=False, unseen_validation=False)))
        raise
