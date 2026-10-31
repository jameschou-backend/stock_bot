#!/usr/bin/env python3
"""Isolated one-date POC ablation; never edits the original candidate ledger.

Only the signal-date 0050 MA120 gate is bypassed in a copied feature mapping.
The relative-strength comparison to 0050 and every other original rule remain.
Missing authentic tick coordinates are exported for shared-quota acquisition.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.extend_scanner_daily_inputs import MATRICES, original_candidates, digest
from scripts.extend_scanner_daily_poc import Continuation
from skills.candidate_quality import candidate_features
from skills.regime_state import build_trend
from skills.strategy_scanner.data import _Inputs, load_bundle
from skills.strategy_scanner.engine import scan_market


def candidate_arms(frames, companies, identity, signal_day, *, features=None):
    """Reuse the original predicate; change one gate value on a private copy."""
    stamp = pd.Timestamp(signal_day)
    if any(frame.index.max() > stamp for frame in frames.values()):
        raise ValueError('Comparison inputs must stop at the signal date')
    features = features if features is not None else candidate_features(frames, companies)
    if features['trend'].at[stamp] not in ('ON', 'OFF'):
        raise ValueError('A known original market gate is required for this comparison')
    baseline, baseline_rejected = original_candidates(frames, companies, identity, signal_day, features=features)
    relaxed = dict(features)
    relaxed['trend'] = features['trend'].copy(deep=True)
    relaxed['trend'].at[stamp] = 'ON'
    variant, variant_rejected = original_candidates(frames, companies, identity, signal_day, features=relaxed)
    return baseline, variant, dict(baseline=baseline_rejected, without_market_gate=variant_rejected)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare_profiles(bundle, signal_day, candidates, official_reports, tick_directories):
    # A same-end continuation starts with no original events. Populate only the
    # new experiment's red candidates, without replacing any bundle file.
    provider = Continuation(bundle, signal_day, official_reports=official_reports,
                            tick_directories=tick_directories)
    if provider.rows or provider.signals or provider.allowed:
        raise ValueError('Isolated profile provider must start empty')
    provider.signals = [dict(signal_id='no-market-gate:' + row['event_id'],
                            stock_id=row['members'][0], signal_date=signal_day) for row in candidates]
    provider._seed()
    provider.offline()
    rows = list(provider.rows.values())
    missing = sorted({(row['stock_id'], day) for row in rows if row['status'] == 'pending_data'
                      for day in row['missing_dates']})
    coordinates = [dict(stock_id=sid, date=day, dataset='TaiwanStockPriceTick',
                        previous_status=provider.unavailable.get(sid + '-' + day, 'not_requested'))
                   for sid, day in missing]
    return provider, rows, coordinates


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new, empty experiment output directory')
    bundle = args.bundle.resolve()
    reader = _Inputs(bundle)
    if reader.manifest['end'] != args.date:
        raise ValueError('Use the frozen bundle ending on the requested date')
    frames = {name: pd.read_parquet(reader.verify(name + '.parquet')).set_index('date') for name in MATRICES}
    companies = pd.read_parquet(reader.verify('companies.parquet'))
    identity = reader.read_json('identity.json')
    f = candidate_features(frames, companies)
    baseline, variant, rejected = candidate_arms(frames, companies, identity, args.date, features=f)
    data = load_bundle(bundle, args.date, args.date)
    sealed = [r for r in data['original_signals'] if r['signal_date'] == args.date]
    if sorted(baseline, key=lambda r: r['event_id']) != sorted(sealed, key=lambda r: r['event_id']):
        raise ValueError('Reconstructed baseline differs from sealed original candidates')
    stamp = pd.Timestamp(args.date)
    quotes = data['bars'].loc[data['bars'].date.eq(stamp)].set_index('stock_id')
    red = [r for r in variant if quotes.at[r['members'][0], 'close'] > quotes.at[r['members'][0], 'open']]
    provider, profiles, missing = prepare_profiles(bundle, args.date, red, args.official_report, args.tick_directory)
    write_new(output / 'profiles.json', profiles)
    write_new(output / 'missing-ticks.json', missing)
    write_new(output / 'day-audits.json', {k: {n: v for n, v in r.items() if n != 'prices'}
                                        for k, r in provider.days_cache.items()})
    provenance = dict(data['provenance'], original_signal_start=args.date, original_signal_end=args.date,
                      original_candidates_complete=True)
    def scan(events):
        return scan_market(data['bars'], data['calendar'], start=args.date, end=args.date,
                           names=data['names'], original_signals=events, poc=profiles,
                           provenance=provenance, strategies=['poc_up_red'])
    original_scan, variant_scan = scan(baseline), scan(variant)
    write_new(output / 'original-scan.json', original_scan)
    write_new(output / 'without-market-gate-scan.json', variant_scan)
    baseline_publication = json.loads(args.baseline_publication.read_text())
    if digest(args.baseline_publication) != args.baseline_publication.with_suffix('.sha256').read_text().strip():
        raise ValueError('Original publication digest mismatch')
    original_status = Counter(s['results']['poc_up_red']['status'] for s in original_scan['days'][0]['stocks'])
    published = next(row for row in baseline_publication['per_strategy_daily_summary'] if row['strategy_id'] == 'poc_up_red')
    if baseline_publication['signal_date'] != args.date or any(original_status[k] != published[k] for k in ('matched', 'not_matched', 'unknown', 'ineligible')):
        raise ValueError('Original full-universe scan differs from the published baseline')
    variant_stocks = variant_scan['days'][0]['stocks']
    variant_status = Counter(s['results']['poc_up_red']['status'] for s in variant_stocks)
    lookup = {row['stock_id']: row for row in profiles}
    events = {row['members'][0]: row for row in variant}
    amounts = (frames['raw-close'] * frames['raw-volume'])
    amount20 = amounts.rolling(20, min_periods=20).mean().loc[stamp]
    median20 = amounts.rolling(20, min_periods=20).median().loc[stamp]
    prior_high = f['close'].shift(1).rolling(60, min_periods=60).max().loc[stamp]
    previous = frames['raw-close'].iloc[-2]
    official = provider.official.xs(args.date, level='date')
    rows = []
    for stock in variant_stocks:
        sid = stock['stock_id']
        if sid not in events:
            continue
        q = quotes.loc[sid]
        p = lookup.get(sid, {})
        official_match = sid in official.index and all(abs(float(q[k])-float(official.at[sid, k])) < 1e-8 for k in ('open', 'high', 'low', 'close'))
        row = dict(stock_id=sid, name=data['names'].get(sid, sid), market=str(official.at[sid, 'market']) if sid in official.index else None,
                   status=stock['results']['poc_up_red']['status'], red_candle=bool(q.close > q.open),
                   open=float(q.open), high=float(q.high), low=float(q.low), close=float(q.close),
                   daily_change_pct=float((q.close/previous[sid]-1)*100),
                   adjusted_close=float(f['close'].at[stamp, sid]), prior_high60=float(prior_high[sid]),
                   amount20=float(amount20[sid]), median_amount20=float(median20[sid]),
                   volume_ratio=events[sid]['leader_evidence']['leader_volume_ratio'],
                   return20_pct=events[sid]['leader_evidence']['leader_return20']*100,
                   relative20_pct=events[sid]['priority']*100,
                   poc_status=p.get('status', 'not_evaluated_non_red'), poc_before=p.get('poc_before'), poc_after=p.get('poc_after'),
                   poc_window_start=p.get('window_start'), poc_window_end=p.get('window_end'),
                   poc_unknown_reason=p.get('reason'), ordinary_daily_matched=p.get('ordinary_daily_matched'),
                   official_ohlc_matched=official_match)
        if row['status'] == 'matched' and not official_match:
            raise ValueError('A matched candidate has conflicting signal-day official prices')
        rows.append(row)
    rows.sort(key=lambda r: (-r['amount20'], r['stock_id']))
    refs = dict(provider.refs)
    refs.update({str((bundle / name).relative_to(ROOT)): value for name, value in reader.hashes.items()})
    for path in (Path(__file__), args.baseline_publication, ROOT / 'scripts/extend_scanner_daily_inputs.py',
                 ROOT / 'scripts/extend_scanner_daily_poc.py', ROOT / 'skills/candidate_quality.py',
                 ROOT / 'skills/regime_state.py', ROOT / 'skills/liquidity_diagnostics.py',
                 ROOT / 'skills/poc_daily_opportunities.py', ROOT / 'skills/volume_profile.py',
                 ROOT / 'skills/strategy_scanner/data.py', ROOT / 'skills/strategy_scanner/engine.py'):
        refs[str(path.resolve().relative_to(ROOT))] = digest(path)
    reader.finish()
    gate = build_trend(frames['close-official']['0050']).loc[stamp]
    result = dict(schema='poc_market_gate_comparison_v1', created_at=datetime.now(timezone.utc).isoformat(),
        signal_date=args.date, baseline_strategy='poc_up_red', variant_id='poc_up_red_without_0050_ma120_research',
        single_change='bypass only signal-date 0050 adjusted close > MA120 gate; preserve relative strength versus 0050',
        experiment_type='same_day_candidate_ablation_not_return_backtest', original_gate=dict(state=str(gate.state), close=float(gate.close), ma120=float(gate.ma120)),
        universe_count=len(variant_stocks), original_status_counts=dict(original_status), variant_status_counts=dict(variant_status),
        baseline_publication_exact=True, original_price_volume_candidates=len(baseline), variant_price_volume_candidates=len(variant),
        variant_red_candidates=len(red), candidates=[row for row in rows if row['status'] == 'matched'],
        variant_price_volume_rows=rows, identity_rejections=rejected, profiles_status_counts=dict(Counter(r['status'] for r in profiles)),
        missing_tick_coordinates=len(missing), scan_complete=not missing, ranking='amount20 descending, stock_id ascending; not a win-probability rank',
        amount_basis='raw close times reported share volume proxy; rolling 20 sessions include signal day',
        decision_timing='T close; any entry evaluation is T+1 or later; no same-day fill claim',
        backtest_run=False, live_qualified=False, original_strategy_changed=False, orders_or_fills_created=False,
        network_requests=0, source_sha256=refs,
        outputs_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(output.glob('*.json'))},
        limitations=['This is one-date selection sensitivity, not evidence of improved returns or drawdown.',
                     'The existing frozen universe, provisional identity, adjustment and source limitations remain.',
                     'POC uses exactly the preceding 20 market sessions, split in halves with 40 common bins; signal-day ticks are excluded.',
                     'Original provider diagnostic policy retained; all-session daily totals do not fully certify ordinary-session ticks.',
                     'Unknown observations remain unknown. No first-trigger lifecycle is inferred for this one-date variant.',
                     'Reconstructed after the signal date; historical real-time publication was not verified.'])
    write_new(output / 'comparison.json', result)
    (output / 'comparison.sha256').write_text(digest(output / 'comparison.json') + '\n')
    print(json.dumps({key: result[key] for key in ('universe_count', 'original_gate', 'original_status_counts', 'variant_status_counts', 'variant_price_volume_candidates', 'variant_red_candidates', 'profiles_status_counts', 'missing_tick_coordinates', 'candidates')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--baseline-publication', type=Path, required=True)
    parser.add_argument('--official-report', type=Path, action='append', required=True)
    parser.add_argument('--tick-directory', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
