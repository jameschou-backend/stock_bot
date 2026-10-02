"""Explicit input revisions for a fixed strategy; never mutate sealed research."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd


def merge_quote_repairs(quotes, frames, additions):
    """Add independently sourced OHLC, daily total shares and adjusted closes.

    Each replacement may fill a missing cell or reproduce its existing value.
    Ordinary-session shares are deliberately not accepted as daily totals.
    The official adjusted series is rebuilt separately from corporate events.
    """
    required = {'date', 'stock_id', 'open', 'high', 'low', 'close',
                'total_daily_volume', 'quality_adjusted_close'}
    if not required <= set(additions.columns):
        raise ValueError('Repair needs explicit daily-total and independent adjusted-price fields')
    a = additions.copy()
    a['date'] = pd.to_datetime(a.date)
    if a.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate repaired stock/date')
    keys = pd.MultiIndex.from_frame(a[['date', 'stock_id']])
    q = quotes.copy()
    q['date'] = pd.to_datetime(q.date)
    if q.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate original stock/date')
    if keys.isin(pd.MultiIndex.from_frame(q[['date', 'stock_id']])).any():
        raise ValueError('Repair would replace a recorded quote')
    out = {name: frame.copy() for name, frame in frames.items()}
    fields = {'raw-close': 'close', 'raw-volume': 'total_daily_volume',
              'close-quality': 'quality_adjusted_close'}
    for row in a.to_dict('records'):
        values = [row[k] for k in ('open', 'high', 'low', 'close', 'quality_adjusted_close')]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Missing independently verified positive repair prices')
        vol = row['total_daily_volume']
        if (isinstance(vol, bool) or not isinstance(vol, (int, float))
                or not math.isfinite(vol) or vol <= 0 or vol != int(vol)):
            raise ValueError('Missing positive integer daily-total shares')
        if not row['low'] <= min(row['open'], row['close']) <= max(row['open'], row['close']) <= row['high']:
            raise ValueError('Impossible repair OHLC')
        for name, field in fields.items():
            frame = out[name]
            if row['date'] not in frame.index or row['stock_id'] not in frame.columns:
                raise ValueError('Repair is outside explicit input axes')
            previous = frame.at[row['date'], row['stock_id']]
            if pd.notna(previous) and previous > 0 and not math.isclose(float(previous), float(row[field]), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError('Repair conflicts with existing input: '+name)
            frame.at[row['date'], row['stock_id']] = row[field]
    new = a[['date', 'stock_id', 'open', 'high', 'low', 'close', 'total_daily_volume']].rename(
        columns={'total_daily_volume': 'volume'})
    return pd.concat([q, new], ignore_index=True).sort_values(['date', 'stock_id']), out


def candidate_diff(before, after):
    def index(rows):
        result = {r['event_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate candidate identities')
        return result
    old, new = index(before), index(after)
    return dict(original_candidate_count=len(old), repaired_candidate_count=len(new),
                added_candidates=[deepcopy(new[k]) for k in sorted(new.keys()-old.keys())],
                removed_candidates=[deepcopy(old[k]) for k in sorted(old.keys()-new.keys())],
                changed_candidates=[k for k in sorted(old.keys() & new.keys()) if old[k] != new[k]],
                ordering_unchanged=[r['event_id'] for r in before] == [r['event_id'] for r in after])


def apply_entry_policy(entries, identity, decide, *, research_risk_notice_assumed):
    """Keep source candidates; log account rejections separately from signals.

    Historical price windows use listing eligibility, not this account mask.
    A stock may supply useful past prices before this account could buy it.
    """
    accepted, blocked = [], []
    for entry in entries:
        result = decide(identity, entry['members'][0], entry['entry_date'],
                        channel='regular', research_risk_notice_assumed=research_risk_notice_assumed)
        if type(result.get('allowed')) is not bool:
            raise ValueError('Account eligibility must explicitly accept or reject')
        if result['allowed']:
            accepted.append(deepcopy(entry))
        else:
            blocked.append(dict(event_id=entry['event_id'], stock_id=entry['members'][0],
                                signal_date=entry['signal_date'], entry_date=entry['entry_date'],
                                decision=deepcopy(result)))
    return accepted, blocked


def observed_roster_check(official, identity, resolver, days, stock_ids, nonordinary=None):
    """Certify exact observed daily rosters, not unobserved intraday restrictions.

    Daily source coverage and security classifications are both mandatory. A
    local missing stock column remains a failure rather than shrinking scope.
    """
    days, ids = set(map(str, days)), set(stock_ids)
    required = {'market', 'date', 'stock_id'}
    if not required <= set(official.columns):
        raise ValueError('Official roster lacks identity columns')
    seen, issues, observed = set(), [], set()
    rows = official.loc[official.date.isin(days)].drop_duplicates(['market', 'date', 'stock_id'])
    for r in rows.itertuples(index=False):
        seen.add((r.market, r.date))
        if str(r.stock_id).startswith('0') and r.stock_id != '0050':
            continue
        if nonordinary and nonordinary(r.stock_id, r.date, r.market):
            continue
        ep = resolver(identity, r.stock_id, r.date)
        if not ep or ep.get('market', '').upper() != r.market:
            issues.append(dict(market=r.market, date=r.date, stock_id=r.stock_id, reason='dated_identity_missing'))
            continue
        if ep.get('category') not in ('股票', 'ETF'):
            continue
        if ep['category'] == 'ETF' and r.stock_id != '0050':
            continue
        observed.add(r.stock_id)
        if r.stock_id not in ids:
            issues.append(dict(market=r.market, date=r.date, stock_id=r.stock_id, reason='stock_absent_from_input_axes'))
    missing = sorted({(m, d) for d in days for m in ('TWSE', 'TPEX')}-seen)
    return dict(complete_observed_daily_rosters=not missing and not issues,
                scope='official_daily_rosters_on_explicit_required_dates',
                observed_ordinary_or_benchmark_ids=len(observed), issues=issues,
                missing_market_dates=[dict(market=m, date=d) for m,d in missing],
                intraday_restrictions_certified=False, live_qualified=False)
