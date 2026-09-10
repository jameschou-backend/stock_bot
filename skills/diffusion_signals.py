"""Fixed, causal price-peer diffusion signals; no downloads or portfolio returns.

See docs/prereg_diffusion_20260910.md. ``volume`` is shares and ``turnover``
is raw close times shares, in TWD. The latter is an estimate, not exchange
reported trading money. The supplied current-company cohort is not a historical
universe. Missing observations are never filled or interpreted as zero returns.
"""
from __future__ import annotations

from collections import Counter
import re
import time

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


ARMS = ('leader_now', 'leader_after', 'follower_after', 'basket_after')
LOOKBACK = 126
MIN_COMMON = 100
MIN_TURNOVER = 50_000_000.


def _date(value, label):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f'{label} must be a timezone-naive date')
    return result


def _validate(close, other_close, volume, turnover, companies):
    if (not isinstance(close.index, pd.DatetimeIndex) or close.index.hasnans
            or close.index.tz is not None or not close.index.is_unique
            or not close.index.is_monotonic_increasing
            or not close.index.equals(close.index.normalize())
            or not close.columns.is_unique or '0050' not in close.columns
            or any(not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid)
                   for sid in close.columns)):
        raise ValueError('Prices require ordered unique dates, four-digit string tickers, and 0050')
    for label, frame in [('close', close), ('other_close', other_close),
                         ('volume', volume), ('turnover', turnover)]:
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError(f'{label} must have exactly aligned dates and tickers')
        try:
            values = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label} must be numeric') from exc
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError(f'{label} cannot contain infinity or negative values')
    for label, frame, expected in [('volume', volume, 'shares'), ('turnover', turnover, 'TWD')]:
        if frame.attrs.get('unit', expected) != expected:
            raise ValueError(f'{label} unit must be {expected}; convert explicitly before research')
    if not {'stock_id', 'listed_date'}.issubset(companies.columns):
        raise ValueError('companies requires stock_id and listed_date')
    if (companies.stock_id.duplicated().any()
            or any(not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid)
                   or sid == '0050' for sid in companies.stock_id)):
        raise ValueError('companies must contain unique four-digit ordinary-stock ids, excluding 0050')
    missing = set(close.columns) - {'0050'} - set(companies.stock_id)
    if missing:
        raise ValueError(f'Price stocks missing from company cohort: {sorted(missing)}')
    listed = {row.stock_id: _date(row.listed_date, 'listed_date')
              for row in companies[['stock_id', 'listed_date']].itertuples(index=False)}
    return listed


def _rolling(values, window, operation='mean'):
    rolling = pd.DataFrame(values).rolling(window, min_periods=window)
    return getattr(rolling, operation)().to_numpy()


def _returns(values, lag=1):
    result = np.full_like(values, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        result[lag:] = values[lag:] / values[:-lag] - 1.
    result[~np.isfinite(result)] = np.nan
    return result


def build_diffusion(close: pd.DataFrame, other_close: pd.DataFrame,
                    volume: pd.DataFrame, turnover: pd.DataFrame,
                    companies: pd.DataFrame, *, start='2022-01-03',
                    signal_end='2025-12-31') -> dict:
    """Build monthly fixed groups and all four preregistered signal ledgers.

    Every row in the input index is one market session. The two price versions
    share the benchmark observation mask; ``close`` determines residuals and
    signals, while both versions determine only-past quality exclusions. Group
    fitting never reads the first session of its own month. Pending leaders can
    finish their ten-session window after signal_end and across month changes.

    ``groups`` contains monthly audit rows with nested ``clusters``. ``events``
    contains selected leaders, including unsuccessful/incomplete observations.
    ``entries`` maps each arm to next-session portfolio-engine input records.
    A missing next session is recorded, never invented. All output is JSON safe.
    """
    started = time.perf_counter()
    listed_dates = _validate(close, other_close, volume, turnover, companies)
    start, signal_end = _date(start, 'start'), _date(signal_end, 'signal_end')
    if signal_end < start:
        raise ValueError('signal_end must not precede start')
    days, ids = close.index, list(close.columns)
    if len(days) < 2:
        raise ValueError('At least two market sessions are required')
    n, p = close.shape
    benchmark = ids.index('0050')
    stocks = np.array([j for j, sid in enumerate(ids) if sid != '0050'], dtype=int)
    price, other, vol, amount = [frame.to_numpy(dtype=float, copy=True)
                                for frame in (close, other_close, volume, turnover)]
    # Enforce the listing boundary even if the caller already masked raw data.
    listed = np.ones((n, p), dtype=bool)
    for j in stocks:
        listed[:, j] = days >= listed_dates[ids[j]]
    for values in (price, other, vol, amount):
        values[(values <= 0) | ~listed] = np.nan
    # A positive supplied amount cannot rescue a missing/zero share count.
    # Both are necessary observations of the raw close-times-shares estimate.
    amount[~np.isfinite(vol)] = np.nan
    ret, other_ret = _returns(price), _returns(other)
    common = np.isfinite(ret[:, benchmark]) & np.isfinite(other_ret[:, benchmark])
    common_count = _rolling(common[:, None].astype(float), LOOKBACK, 'sum')[:, 0]
    invalid_common = common[:, None] & ~(np.isfinite(ret) & np.isfinite(other_ret))
    incomplete = _rolling(invalid_common.astype(float), LOOKBACK, 'sum') > 0
    anomalies = ((np.abs(ret) > .20) | (np.abs(other_ret) > .20)
                 | (np.abs(ret - other_ret) > .005))
    anomalous_window = _rolling(anomalies.astype(float), LOOKBACK, 'sum') > 0
    liquid20 = _rolling(amount, 20)
    complete_listing = np.zeros((n, p), dtype=bool)
    for j in stocks:
        complete_listing[LOOKBACK:, j] = days[:-LOOKBACK] >= listed_dates[ids[j]]
    complete_listing[LOOKBACK:, benchmark] = True
    enough = (np.arange(n) >= LOOKBACK) & (common_count >= MIN_COMMON)
    quality = (enough[:, None] & complete_listing & ~incomplete & ~anomalous_window
               & np.isfinite(liquid20) & (liquid20 >= MIN_TURNOVER))
    benchmark_ok = enough & ~anomalous_window[:, benchmark]
    quality &= benchmark_ok[:, None]
    r5, r20 = _returns(price, 5), _returns(price, 20)
    prior_high60 = np.full_like(price, np.nan)
    prior_vol20 = np.full_like(vol, np.nan)
    prior_high60[1:] = _rolling(price, 60, 'max')[:-1]
    prior_vol20[1:] = _rolling(vol, 20)[:-1]
    technical_leader = (quality & (price > prior_high60) & (r20 > 0)
                        & (r20 > r20[:, [benchmark]]) & (vol >= prior_vol20 * 1.5))
    technical_leader[:, benchmark] = False

    # Every listed company in the supplied cohort counts as expected, including
    # companies without any price column. Missing history cannot shrink this
    # denominator and make observed turnover look like full market coverage.
    raw_valid = np.isfinite(amount) & np.isfinite(vol) & listed
    raw_valid[:, benchmark] = False
    listing_times = np.sort(np.array([value.to_datetime64() for value in listed_dates.values()]))
    expected_count = np.searchsorted(listing_times, days.to_numpy(), side='right')
    coverage = np.divide(raw_valid.sum(axis=1), expected_count,
                         out=np.full(n, np.nan), where=expected_count > 0)
    denominator = np.nansum(np.where(raw_valid, amount, np.nan), axis=1)
    denominator[(coverage < .95) | ~np.isfinite(coverage) | (denominator <= 0)] = np.nan

    def as_date(i):
        return str(days[i].date())

    def breadth(i, peers):
        values, base = r5[i, peers], r5[i, benchmark]
        if not np.isfinite(base) or not np.isfinite(values).all():
            return None, None
        response = (values > 0) & (values > base)
        return float(response.mean()), response

    def share_evidence(i, peers):
        result = {'last5_mean': None, 'prior20_mean': None, 'rising': None,
                  'coverage_min': None, 'valid': False}
        if i < 24:
            return result
        rows = slice(i - 24, i + 1)
        required = amount[rows][:, peers]
        cov = coverage[rows]
        if np.isfinite(cov).all():
            result['coverage_min'] = float(cov.min())
        den = denominator[rows]
        if not np.isfinite(required).all() or not np.isfinite(den).all():
            return result
        shares = required.sum(axis=1) / den
        short, previous = float(shares[-5:].mean()), float(shares[:-5].mean())
        result.update(last5_mean=short, prior20_mean=previous,
                      rising=bool(short > previous), valid=True)
        return result

    monthly, events = [], []
    entries = {arm: [] for arm in ARMS}
    pending = []
    counts = Counter()
    invalid_windows = Counter()
    month_lookup = {}

    def fit_month(first_i):
        month = str(days[first_i].to_period('M'))
        cutoff = first_i - 1
        audit = {'month': month, 'cutoff_date': as_date(cutoff) if cutoff >= 0 else None,
                 'common_observations': int(common_count[cutoff])
                 if cutoff >= 0 and np.isfinite(common_count[cutoff]) else 0,
                 'eligible_count': 0, 'selected_ids': [], 'clusters': [],
                 'discarded_clusters': [], 'exclusions': {}, 'leader_rejections': [],
                 'status': 'insufficient_history'}
        monthly.append(audit)
        month_lookup[month] = audit
        if cutoff < LOOKBACK or not enough[cutoff]:
            invalid_windows['insufficient_common_history'] += 1
            return []
        if not benchmark_ok[cutoff]:
            audit['status'] = 'benchmark_price_anomaly'
            invalid_windows['benchmark_price_anomaly'] += 1
            return []
        exclusions = {
            'listing_window': ~complete_listing[cutoff, stocks],
            'incomplete_common_returns': incomplete[cutoff, stocks],
            'price_anomaly_in_past126': anomalous_window[cutoff, stocks],
            'incomplete_turnover20': ~np.isfinite(liquid20[cutoff, stocks]),
            'turnover_below_50000000': np.isfinite(liquid20[cutoff, stocks])
            & (liquid20[cutoff, stocks] < MIN_TURNOVER),
        }
        audit['exclusions'] = {reason: sorted(ids[j] for j in stocks[mask])
                               for reason, mask in exclusions.items() if mask.any()}
        eligible = [j for j in stocks if quality[cutoff, j]]
        audit['eligible_count'] = len(eligible)
        selected = sorted(eligible, key=lambda j: (-liquid20[cutoff, j], ids[j]))[:300]
        # Sorted symbols fix the linkage input order, including exact ties.
        selected.sort(key=lambda j: ids[j])
        audit['selected_ids'] = [ids[j] for j in selected]
        audit['status'] = 'fitted'
        if len(selected) < 4:
            audit['status'] = 'too_few_eligible'
            invalid_windows['too_few_eligible'] += 1
            return []
        rows = np.arange(cutoff - LOOKBACK + 1, cutoff + 1)
        rows = rows[common[rows]]
        x = ret[rows, benchmark]
        y = ret[np.ix_(rows, selected)]
        x, y = x - x.mean(), y - y.mean(axis=0)
        variance = float(x @ x)
        residual = y - np.outer(x, (x @ y) / variance) if variance > 0 else y
        norm = np.linalg.norm(residual, axis=0)
        nonconstant = norm > np.finfo(float).eps * np.sqrt(len(rows))
        audit['exclusions']['zero_residual_variance'] = [ids[j] for j, ok
                                                        in zip(selected, nonconstant) if not ok]
        selected = [j for j, ok in zip(selected, nonconstant) if ok]
        residual = residual[:, nonconstant]
        if len(selected) < 4:
            audit['status'] = 'too_few_nonconstant'
            invalid_windows['too_few_nonconstant'] += 1
            return []
        unit = residual / np.linalg.norm(residual, axis=0)
        distance = np.clip(1. - unit.T @ unit, 0., 2.)
        distance = (distance + distance.T) / 2.
        np.fill_diagonal(distance, 0.)
        labels = fcluster(linkage(squareform(distance, checks=False), method='average'),
                          t=.65, criterion='distance')
        groups = sorted([sorted(j for j, label in zip(selected, labels) if label == value)
                         for value in np.unique(labels)],
                        key=lambda members: tuple(sorted(ids[j] for j in members)))
        retained = []
        for members in groups:
            symbols = sorted(ids[j] for j in members)
            if not 4 <= len(members) <= 20:
                audit['discarded_clusters'].append({'members': symbols,
                                                    'reason': 'size_outside_4_to_20'})
                continue
            group_id = f'{month}-g{len(retained) + 1:03d}'
            record = {'group_id': group_id, 'members': symbols}
            audit['clusters'].append(record)
            retained.append({'group_id': group_id, 'columns': members, 'used': False})
        return retained

    def add_entry(arm, event, i, members):
        if i + 1 >= n:
            event['entry_date_missing'].append(arm)
            return
        entries[arm].append({'event_id': event['event_id'], 'signal_date': as_date(i),
                             'entry_date': as_date(i + 1), 'members': sorted(members),
                             'priority': event['priority']})

    first_by_month = {}
    for i, day in enumerate(days):
        first_by_month.setdefault(str(day.to_period('M')), i)
    relevant = np.flatnonzero((days >= start) & (days <= signal_end))
    if len(relevant):
        last_observation = min(n - 1, int(relevant[-1]) + 10)
        current_month, active_groups = None, []
        for i in range(int(relevant[0]), last_observation + 1):
            month = str(days[i].to_period('M'))
            if days[i] <= signal_end and month != current_month:
                active_groups = fit_month(first_by_month[month])
                current_month = month
            for item in list(pending):
                event, leader_i = item['event'], item['leader_i']
                if i <= leader_i or i > leader_i + 10:
                    continue
                peers = item['peers']
                peer_breadth, response = breadth(i, peers)
                share = share_evidence(i, peers)
                check = {'date': as_date(i), 'session_after_leader': i - leader_i,
                         'peer_breadth': peer_breadth, 'new_responders': None,
                         'turnover_share': share, 'follower_id': None,
                         'leader_quality': bool(quality[i, item['leader']]),
                         'follower_quality': None, 'reason': None}
                if response is None:
                    check['reason'] = 'missing_peer_or_benchmark_prices'
                else:
                    newcomers = [j for j, now, then in zip(peers, response, item['response'])
                                 if now and not then]
                    check['new_responders'] = sorted(ids[j] for j in newcomers)
                    if peer_breadth < .6 or len(newcomers) < 2:
                        check['reason'] = 'breadth_or_new_responders_below_threshold'
                    elif not share['valid']:
                        check['reason'] = 'missing_turnover_share_window'
                    elif not share['rising']:
                        check['reason'] = 'peer_turnover_share_not_rising'
                    else:
                        follower = sorted(newcomers, key=lambda j: (-(r5[i, j] - r5[i, benchmark]),
                                                                   ids[j]))[0]
                        check['follower_id'] = ids[follower]
                        check['follower_quality'] = bool(quality[i, follower])
                        if not quality[i, item['leader']] or not quality[i, follower]:
                            check['reason'] = 'leader_or_follower_quality_window_invalid'
                        else:
                            check['reason'] = 'confirmed'
                            event.update(status='confirmed', confirmation_date=as_date(i),
                                         follower_id=ids[follower], confirmation_peer_breadth=peer_breadth,
                                         confirmation_turnover_share=share,
                                         confirmation_new_responders=check['new_responders'])
                            add_entry('leader_after', event, i, [event['leader_id']])
                            add_entry('follower_after', event, i, [ids[follower]])
                            add_entry('basket_after', event, i, event['members'])
                            pending.remove(item)
                event['confirmation_checks'].append(check)
                if check['reason'] in {'missing_peer_or_benchmark_prices',
                                       'missing_turnover_share_window',
                                       'leader_or_follower_quality_window_invalid'}:
                    event['insufficient_data_sessions'] += 1
                if i == leader_i + 10 and event['status'] != 'confirmed':
                    event['status'] = ('data_insufficient' if event['insufficient_data_sessions']
                                       else 'no_confirmation')
                    pending.remove(item)

            if days[i] > signal_end:
                continue
            for group in active_groups:
                if group['used']:
                    continue
                candidates = [j for j in group['columns'] if technical_leader[i, j]]
                accepted = []
                for leader in candidates:
                    peers = [j for j in group['columns'] if j != leader]
                    peer_breadth, response = breadth(i, peers)
                    if peer_breadth is None or peer_breadth > .4:
                        reason = 'missing_peer_or_benchmark_prices' if peer_breadth is None else 'already_broad'
                        month_lookup[month]['leader_rejections'].append(
                            {'date': as_date(i), 'group_id': group['group_id'],
                             'leader_id': ids[leader], 'peer_breadth': peer_breadth, 'reason': reason})
                        counts[f'leader_rejected_{reason}'] += 1
                    else:
                        accepted.append((leader, peers, peer_breadth, response))
                if not accepted:
                    continue
                leader, peers, peer_breadth, response = sorted(
                    accepted, key=lambda item: (-(r20[i, item[0]] - r20[i, benchmark]), ids[item[0]]))[0]
                group['used'] = True
                event = {'event_id': f"{group['group_id']}-{as_date(i)}-{ids[leader]}",
                         'group_id': group['group_id'], 'group_month': month,
                         'group_cutoff_date': month_lookup[month]['cutoff_date'],
                         'status': 'pending', 'leader_id': ids[leader], 'follower_id': None,
                         'leader_date': as_date(i), 'confirmation_date': None,
                         'deadline_date': as_date(i + 10) if i + 10 < n else None,
                         'members': sorted(ids[j] for j in group['columns']),
                         'leader_peer_breadth': peer_breadth,
                         'leader_responders': sorted(ids[j] for j, yes in zip(peers, response) if yes),
                         'leader_turnover_share': share_evidence(i, peers),
                         'leader_return20': float(r20[i, leader]),
                         'benchmark_return20': float(r20[i, benchmark]),
                         'priority': float(r20[i, leader] - r20[i, benchmark]),
                         'leader_volume_ratio': float(vol[i, leader] / prior_vol20[i, leader]),
                         'confirmation_peer_breadth': None, 'confirmation_turnover_share': None,
                         'confirmation_new_responders': None, 'confirmation_checks': [],
                         'insufficient_data_sessions': 0, 'entry_date_missing': []}
                events.append(event)
                add_entry('leader_now', event, i, [ids[leader]])
                pending.append({'event': event, 'leader_i': i, 'leader': leader,
                                'peers': peers, 'response': response.copy()})
        for item in pending:
            item['event']['status'] = 'window_incomplete'

    status_counts = Counter(event['status'] for event in events)
    return {'groups': monthly, 'events': events, 'entries': entries,
            'stats': {'seconds': float(time.perf_counter() - started),
                      'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
                      'months': len(monthly), 'groups': sum(len(row['clusters']) for row in monthly),
                      'leaders': len(events), 'statuses': dict(status_counts),
                      'entry_counts': {arm: len(batch) for arm, batch in entries.items()},
                      'counts': dict(counts), 'invalid_windows': dict(invalid_windows),
                      'input_sessions': n, 'input_stocks': len(stocks),
                      'turnover_coverage_under_95_sessions': int(np.sum(coverage < .95)),
                      'cohort_limitation': 'current_company_cohort_not_historical_universe',
                      'turnover_unit': 'TWD_raw_close_times_shares_estimate',
                      'volume_unit': 'shares', 'live_qualified': False}}
