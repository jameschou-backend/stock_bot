#!/usr/bin/env python3
"""Read-only, hindsight case audit; never adds signals or hypothetical profits."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.diffusion_signals import LOOKBACK, MIN_COMMON, MIN_TURNOVER, _returns, _rolling


def monthly_block(month, sid):
    reasons = [reason for reason, ids in month['exclusions'].items() if sid in ids]
    if reasons:
        return 'monthly_quality_or_liquidity', reasons, None
    if sid not in month['selected_ids']:
        return 'outside_top300', [], None
    discarded = next((g for g in month['discarded_clusters'] if sid in g['members']), None)
    if discarded:
        return 'cluster_size', [len(discarded['members'])], None
    group = next((g for g in month['clusters'] if sid in g['members']), None)
    if group is None:
        raise ValueError('Unexplained monthly exclusion: ' + sid)
    return None, [], group


def group_block(day, sid, event, breadth):
    # The actual builder consumes a group before considering later candidates.
    if event and event['leader_date'] < day:
        return 'group_already_used'
    if breadth is None:
        return 'missing_peer_prices'
    if breadth > .4:
        return 'already_broad'
    if event and event['leader_date'] == day:
        return 'selected_leader' if event['leader_id'] == sid else 'lower_priority_same_day'
    raise ValueError('Valid candidate missing from sealed event ledger')


def main():
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    publication = ROOT/'artifacts/forward_simulation/partial_risk_2019_20260929.json'
    output = ROOT/'artifacts/forward_simulation/missed_winners_2024_20260929.json'
    if output.exists():
        raise ValueError('Preserve existing research; choose a new version')
    refs = {}

    def bind(path):
        refs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    manifest = json.loads(bind(base/'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        path = bind(base/name)
        if refs[str(path.relative_to(ROOT))] != digest:
            raise ValueError('Frozen input changed: ' + name)
    pub = json.loads(bind(publication).read_text())
    if not pub['all_completed'] or not pub['offline_identical']:
        raise ValueError('Completed reproducible account required')
    for rel, digest in pub['export_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != digest:
            raise ValueError('Export changed: ' + rel)
    for path in [Path(__file__), ROOT/'skills/historical_diffusion_signals.py',
                 ROOT/'skills/diffusion_signals.py', ROOT/'skills/historical_selector_replay.py']:
        bind(path)

    def frame(name):
        return pd.read_parquet(base/(name+'.parquet')).set_index('date')

    close, other, raw, volume = [frame(name) for name in
        ('close-official', 'close-quality', 'raw-close', 'raw-volume')]
    companies = pd.read_parquet(base/'companies.parquet').set_index('stock_id')
    names = companies['name']
    eligibility = frame('eligibility').reindex(columns=close.columns)
    days, ids = close.index, list(close.columns)
    listing = np.ones(close.shape, dtype=bool)
    for j, sid in enumerate(ids):
        if sid != '0050':
            listing[:, j] = days >= companies.at[sid, 'listed_date']
    listing &= eligibility.to_numpy()
    price, other_price, vol, amount = [f.to_numpy(dtype=float, copy=True)
        for f in (close, other, volume, raw*volume)]
    for values in (price, other_price, vol, amount):
        values[(values <= 0) | ~listing] = np.nan
    amount[~np.isfinite(vol)] = np.nan
    benchmark = ids.index('0050')
    ret, ret_other = _returns(price), _returns(other_price)
    common = np.isfinite(ret[:, benchmark]) & np.isfinite(ret_other[:, benchmark])
    count = _rolling(common[:, None].astype(float), LOOKBACK, 'sum')[:, 0]
    incomplete = _rolling((common[:, None] & ~(np.isfinite(ret) & np.isfinite(ret_other))).astype(float), LOOKBACK, 'sum') > 0
    anomaly = _rolling(((abs(ret) > .2) | (abs(ret_other) > .2) | (abs(ret-ret_other) > .005)).astype(float), LOOKBACK, 'sum') > 0
    mature = np.zeros(close.shape, dtype=bool)
    for j, sid in enumerate(ids):
        mature[LOOKBACK:, j] = True if sid == '0050' else days[:-LOOKBACK] >= companies.at[sid, 'listed_date']
    enough = (np.arange(len(days)) >= LOOKBACK) & (count >= MIN_COMMON)
    liquid = _rolling(amount, 20)
    quality = enough[:, None] & mature & ~incomplete & ~anomaly & np.isfinite(liquid) & (liquid >= MIN_TURNOVER)
    quality &= (enough & ~anomaly[:, benchmark])[:, None] & listing
    r5, r20 = _returns(price, 5), _returns(price, 20)
    high, meanvol = np.full_like(price, np.nan), np.full_like(price, np.nan)
    high[1:] = _rolling(price, 60, 'max')[:-1]
    meanvol[1:] = _rolling(vol, 20)[:-1]
    technical = quality & (price > high) & (r20 > 0) & (r20 > r20[:, [benchmark]]) & (vol >= meanvol*1.5)
    technical[:, benchmark] = False

    signals = json.loads((base/'signals.json').read_text())
    monthly = {g['month']: g for g in signals['diffusion']['groups'] if g['month'].startswith('2024')}
    events = {e['group_id']: e for e in signals['diffusion']['events']}
    accepted = {e['event_id']: e for e in signals['entries']}
    rejected = {e['event_id']: e for e in signals['rejections']}
    # Validate reconstructed daily conditions against every recorded 2024 leader.
    for event in events.values():
        if event['leader_date'].startswith('2024'):
            assert technical[days.get_loc(event['leader_date']), ids.index(event['leader_id'])]
    folder = publication.with_suffix('')
    orders = pd.read_csv(bind(folder/'original-orders.csv'), dtype={'stock_id': str})
    holdings = pd.read_csv(bind(folder/'original-holdings.csv'), dtype={'stock_id': str})
    trades = pd.read_csv(bind(folder/'original-trades.csv'), dtype={'stock_id': str})
    # Hindsight annual price ranking is ONLY the case-selection rule.
    returns = (close.loc['2024-12-31']/close.loc['2023-12-29']-1).drop('0050').dropna().sort_values(ascending=False)
    winners = list(returns.head(20).index)
    bought = list(trades.loc[trades.side.eq('buy') & trades.date.str.startswith('2024'), 'stock_id'].unique())
    cohort = list(dict.fromkeys(winners + ['2365', '2486', '2330', '2317', '2467', '3363'] + bought))
    records = []
    for sid in cohort:
        j = ids.index(sid)
        months, checks = [], []
        for month, audit in monthly.items():
            block, details, group = monthly_block(audit, sid)
            event = events.get(group['group_id']) if group else None
            months.append(dict(month=month, block=block, details=details, group=group,
                               consumed_by=event['leader_id'] if event else None,
                               consumed_on=event['leader_date'] if event else None))
        for i in np.flatnonzero(technical[:, j] & (days.year == 2024)):
            day = str(days[i].date())
            audit = monthly[day[:7]]
            block, details, group = monthly_block(audit, sid)
            event, breadth = None, None
            if group:
                event = events.get(group['group_id'])
                peers = [ids.index(s) for s in group['members'] if s != sid]
                values, market = r5[i, peers], r5[i, benchmark]
                if np.isfinite(values).all() and np.isfinite(market):
                    breadth = float(((values > 0) & (values > market)).mean())
                block = group_block(day, sid, event, breadth)
            row = dict(date=day, block=block, details=details, peer_breadth=breadth,
                return20=float(r20[i, j]), excess20=float(r20[i, j]-r20[i, benchmark]),
                volume_ratio=float(vol[i, j]/meanvol[i, j]))
            if event:
                row.update(group_leader=event['leader_id'], group_leader_date=event['leader_date'])
            if block == 'selected_leader':
                eid = event['event_id']
                row['trend_gate'] = 'accepted' if eid in accepted else rejected[eid]['reason']
                if eid in accepted:
                    planned = accepted[eid]['entry_date']
                    matching = orders[orders.event_id.eq(eid) & orders.side.eq('buy')]
                    row['orders'] = json.loads(matching.to_json(orient='records'))
                    previous = str(days[days.get_loc(planned)-1].date())
                    row['prior_holdings'] = json.loads(holdings[holdings.date.eq(previous)].to_json(orient='records'))
            checks.append(row)
        fills = trades[trades.stock_id.eq(sid) & trades.date.str.startswith('2024')]
        records.append(dict(stock_id=sid, name=names[sid], annual_adjusted_price_return=float(returns[sid]),
            retrospective_top20=sid in winners, months=months, technical_days=checks,
            technical_blocks=dict(Counter(r['block'] for r in checks)),
            trades_2024=json.loads(fills.to_json(orient='records'))))
    report = dict(year=2024, methodology='Retrospective cases; forward-known gates from frozen inputs; no new backtest',
        annual_return_basis='2023-12-29 to 2024-12-31 official-adjusted close ratio, not portfolio net return',
        full_historical_market=False, live_qualified=False, unseen_validation=False,
        cases=records, source_sha256=refs)
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    print(json.dumps({r['stock_id']: r['technical_blocks'] for r in records}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
