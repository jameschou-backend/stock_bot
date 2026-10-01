#!/usr/bin/env python3
"""Offline independent accounting and prefix-only signal audit of a sealed case."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.return_claim_audit import audit_account, require, same
# Reuse source-schema readers only, never replay/NAV/cost/signal implementations.
from skills.historical_odd_regime import parse_after_hours
from skills.replay_market_feeds import parse_odd


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_check(left, right):
    require(left != right, 'Two distinct sealed runs are required')
    reports = [read(p/'report.json') for p in (left, right)]
    require(reports[0]['source_sha256'] == reports[1]['source_sha256'], 'Source sets differ')
    for i, (path, digest) in enumerate(reports[0]['source_sha256'].items(), 1):
        source = (ROOT/path).resolve()
        require(source.is_relative_to(ROOT) and sha(source) == digest, 'Changed source: '+path)
        if i % 10000 == 0:
            print('Verified source files:', i, flush=True)
    for directory, report in zip((left, right), reports):
        require(report['validated'] and report['all_completed'] and not report['preparation'], 'Unsealed case')
        require(sha(directory/'three_black.json') == report['cases']['three_black']['sha256'], 'Case hash differs')
    require(sha(left/'three_black.json') == sha(right/'three_black.json'), 'Offline cases differ')
    publication = read(ROOT/'artifacts/forward_simulation/three_black_exit_20261001.json')
    for path, digest in publication['exports_sha256'].items():
        require(sha(ROOT/path) == digest, 'Published export changed: '+path)
    return reports[0], dict(verified_source_files=len(reports[0]['source_sha256']),
        verified_exports=len(publication['exports_sha256']), independent_runs_identical=True)


def prefix_signals(account, frames, companies, quotes):
    """Recalculate bought-stock predicates using truncated two-symbol histories.

    This checks executed entry conditions and logged exits, not the completeness
    of the historical universe or whether all losing candidates were present.
    """
    days = frames['raw-close'].index
    listed = companies.set_index('stock_id').listed_date
    cohorts = {r['event_id']: r for r in account['cohorts']}
    output = []
    for c in cohorts.values():
        sid, signal = c['stock_id'], pd.Timestamp(c['signal_date'])
        i = days.get_loc(signal)
        require(str(days[i+1].date()) == c['entry_date'], 'Entry not immediately after signal')
        require(c['group_cutoff_date'] < c['signal_date'], 'Group date is not causal')
        ids = [sid, '0050']
        eligible = frames['eligibility'].loc[:signal, ids].copy()
        eligible[sid] &= eligible.index >= pd.Timestamp(listed[sid])
        price = frames['close-official'].loc[:signal, ids]
        other = frames['close-quality'].loc[:signal, ids]
        raw = frames['raw-close'].loc[:signal, ids]
        volume = frames['raw-volume'].loc[:signal, ids]
        price = price.where(eligible & price.gt(0))
        other = other.where(eligible & other.gt(0))
        volume = volume.where(eligible & volume.gt(0))
        ret, alt = price.pct_change(fill_method=None), other.pct_change(fill_method=None)
        common = ret['0050'].notna() & alt['0050'].notna()
        window = ret.tail(126)
        alt_window = alt.tail(126)
        require(len(window) == 126 and common.tail(126).sum() >= 100, 'Insufficient quality history')
        require(days[i-126] >= pd.Timestamp(listed[sid]), 'Listing history too short')
        require(not ((window.abs() > .2) | (alt_window.abs() > .2)
                     | ((window-alt_window).abs() > .005)).any().any(), 'Historical anomaly filter differs')
        require(not ((window.isna() | alt_window.isna())
                     .mul(common.tail(126), axis=0)).any().any(), 'Missing common return history')
        prior = price[sid].iloc[-61:-1]
        prior_vol = volume[sid].iloc[-21:-1]
        require(len(prior) == 60 and prior.notna().all() and prior_vol.notna().all(), 'Incomplete technical lookback')
        r20 = price.iloc[-1]/price.iloc[-21]-1
        ratio = volume[sid].iloc[-1]/prior_vol.mean()
        amount = (raw[sid]*volume[sid]).tail(20)
        bench = frames['close-official'].loc[:signal, '0050']
        bench = bench.where(np.isfinite(bench) & bench.gt(0))
        recent_bench = bench.dropna().tail(120)
        require(price[sid].iloc[-1] > prior.max() and r20[sid] > 0
                and r20[sid] > r20['0050'] and ratio >= 1.5, 'Technical buy predicate failed')
        require(amount.notna().all() and amount.mean() >= 50_000_000
                and amount.median() >= 50_000_000, 'Liquidity predicate failed')
        require(len(recent_bench) == 120 and bench.iloc[-1] > recent_bench.mean(), 'Benchmark trend predicate failed')
        same(c['priority'], r20[sid]-r20['0050'], 'Entry ranking score')
        output.append(dict(event_id=c['event_id'], stock_id=sid, signal_date=c['signal_date'],
            entry_date=c['entry_date'], prior60_high=float(prior.max()), signal_close=float(price[sid].iloc[-1]),
            volume_ratio=float(ratio), median20_amount=float(amount.median()), relative20=float(r20[sid]-r20['0050'])))
    triggers = {}
    indexed = quotes.set_index(['date', 'stock_id'])
    for r in account['black_log']:
        c = cohorts[r['event_id']]
        i, entry = days.get_loc(pd.Timestamp(r['date'])), days.get_loc(pd.Timestamp(c['entry_date']))
        require(r['signal_date'] == str(days[i-1].date()), 'Exit decision is not prior-close')
        require(r['stock_id'] == c['stock_id'], 'Exit identity mismatch')
        sid = r['stock_id']
        expected = i >= 4 and i-entry >= 3
        if expected:
            for j in range(i-3, i):
                if (days[j], sid) not in indexed.index:
                    expected = False
                    continue
                bar = indexed.loc[(days[j], sid)]
                prev, now = frames['close-official'][sid].iloc[j-1:j+1]
                values = [bar.open, bar.close, bar.volume, prev, now]
                expected &= bool(all(np.isfinite(v) and v > 0 for v in values)
                    and bar.close < bar.open and now < prev-max(1e-12, max(abs(prev), abs(now))*1e-12))
        require(bool(r['trigger']) == expected, 'Three-black exit predicate differs')
        if expected:
            require(r['event_id'] not in triggers, 'Duplicate exit trigger')
            triggers[r['event_id']] = r['signal_date']
    for t in account['trades']:
        if t['reason'] == 'three_black':
            require(triggers[t['event_id']] == t['signal_date'] < t['date'], 'Fill without earlier exit trigger')
    return dict(executed_entry_conditions_checked=len(output), black_decisions_checked=len(account['black_log']),
        black_triggers_checked=len(triggers), entries=output,
        complete_candidate_universe_verified=False, unseen_validation=False)


def run(left, right):
    report, integrity = source_check(left, right)
    case = read(left/'three_black.json')
    require(case['completed'], 'Incomplete account cannot substantiate a return claim')
    published = read(ROOT/'artifacts/forward_simulation/three_black_exit_20261001.json')
    comparison = next(r for r in published['comparison'] if r['arm'] == 'three_black')
    for field in ('final_nav', 'total_return', 'max_drawdown'):
        same(comparison[field], case['summary'][field], 'Published comparison '+field)
    a = case['account']
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    ids = sorted({c['stock_id'] for c in a['cohorts']} | {'0050'})
    frames = {}
    for name in ('raw-close', 'raw-volume', 'close-official', 'close-quality', 'eligibility'):
        frame = pd.read_parquet(base/(name+'.parquet'), columns=['date', *ids]).set_index('date')
        frame.index = pd.to_datetime(frame.index)
        frames[name] = frame
    days = frames['raw-close'].index
    companies = pd.read_parquet(base/'companies.parquet')
    quotes = pd.read_parquet(base/'quotes-unmasked.parquet', filters=[('stock_id', 'in', ids)])
    quotes.date = pd.to_datetime(quotes.date)
    require(not quotes.duplicated(['date','stock_id']).any(), 'Duplicate raw quotes')
    by_quote = quotes.set_index(['date', 'stock_id'])
    volumes = frames['raw-volume'].where(frames['eligibility'])
    fractional_rounding = {}
    for name in ('stock_universe_2019_corporate_terms', 'stock_universe_five_corporate_terms',
                 'three_black_corporate_terms_20261001'):
        terms = read(ROOT/'docs'/(name+'.json'))
        for key, value in terms['overrides'].items():
            if 'stock' in value.get('kind', '') or 'shares_per_share' in value:
                fractional_rounding[key+'-stock'] = value.get('fractional_rounding', 'half_up_cents')
        for halt in terms.get('verified_halts', []):
            sid = halt['stock_id']
            if sid in volumes:
                require(sha(ROOT/halt['source_path']) == halt['source_sha256'], 'Halt source changed')
                volumes.loc[(days >= halt['start']) & (days < halt['end']), sid] = 0.
    # The zero-value fractional 6409 right is explicitly cash-unavailable.
    fractional_rounding['6409-2019-09-02-stock'] = 'floor_ntd'
    adv = volumes.rolling(20, min_periods=20).mean().shift(1)
    marks = {}
    for sid in ids:
        positive = frames['raw-close'][sid].gt(0) & frames['raw-volume'][sid].gt(0)
        prices = frames['raw-close'][sid].where(positive).ffill()
        dates = pd.Series(days, index=days).where(positive).ffill()
        for day in a['daily']:
            dt = pd.Timestamp(day['date'])
            if pd.notna(prices.at[dt]):
                marks[(day['date'], sid)] = float(prices.at[dt]), str(dates.at[dt].date())
    odd_paths = {}
    for path in report['source_sha256']:
        match = re.fullmatch(r'odd-(twse|tpex)-(\d{4}-\d{2}-\d{2})(?:\.raw)?\.json', Path(path).name)
        if match:
            odd_paths.setdefault(match.groups(), []).append(ROOT/path)
    episodes = read(base/'identity.json')['episodes']
    market_days = {}
    for t in a['trades']:
        matches = [e for e in episodes if e['stock_id'] == t['stock_id']
                   and e['start'] <= t['date'] and (e['end'] is None or t['date'] < e['end'])]
        require(len(matches) == 1, 'Ambiguous dated market identity')
        market_days[(t['date'], t['stock_id'])] = matches[0]['market'].lower()
    parsed, execution = {}, {}
    for t in a['trades']:
        day, sid, channel = t['date'], t['stock_id'], t['channel']
        if channel == 'board':
            r = by_quote.loc[(pd.Timestamp(day), sid)]
            high, low, volume = float(r.high), float(r.low), float(r.volume)
        else:
            key = market_days[(day, sid)], day
            if key not in parsed:
                paths = odd_paths[key]
                parser = parse_after_hours if day < '2020-10-26' else parse_odd
                values = [parser(read(p), key[0], day) for p in paths]
                require(all(v == values[0] for v in values), 'Conflicting cached odd quotes')
                parsed[key] = values[0]
            r = parsed[key][sid]
            high, low, volume = r['odd_high'], r['odd_low'], r['odd_shares']
        execution[(day, sid, channel)] = high, low, volume, float(adv.at[pd.Timestamp(day), sid])
    print('Reconstruct cash, shares, receivables and NAV', flush=True)
    result = audit_account(case, marks, execution, fractional_rounding=fractional_rounding)
    print('Recalculate prefix-only entry and exit conditions', flush=True)
    result['signal_audit'] = prefix_signals(a, frames, companies, quotes)
    result['integrity'] = integrity
    result['audit_date'] = '2026-10-02'
    result['external_price_reverification'] = dict(status='not_completed',
        reason='Official web endpoints did not return accessible historical price data; existing TWSE origin hold preserved.',
        cached_odd_sources_reparsed=len(parsed), raw_daily_prices='frozen FinMind source; no independent full-market re-download')
    result['limitations'] = ['HL2 is a full-day price proxy, not a known opening price or verified fill.',
        'Daily capacity is not order book depth or priority.',
        'Historical universe and source revisions are not completely point-in-time verified.',
        'All these years have already informed research; this is not unseen validation.',
        'Personal dividend taxes, supplemental premiums and some remittance costs are not fully modeled.',
        'Exact-decimal shadow NAV holds recorded fills fixed; it is not a new order-sizing replay.']
    result['source_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in (
        left/'report.json', left/'three_black.json', right/'report.json', right/'three_black.json',
        ROOT/'skills/return_claim_audit.py', Path(__file__),
        ROOT/'skills/replay_market_feeds.py', ROOT/'skills/historical_odd_regime.py')}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=Path, default=ROOT/'.cache/three-black-20261001/final-c')
    parser.add_argument('--right', type=Path, default=ROOT/'.cache/three-black-20261001/final-d')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing audit output')
    result = run(args.left.resolve(), args.right.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('daily', 'signal_audit', 'source_sha256')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
