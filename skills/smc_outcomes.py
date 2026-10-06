"""Paired FVG formation-cohort outcomes, independent of signal construction.

The direct and retest arms share the formation date's T+h close. A known expired
or invalidated setup that never entered remains in the waiting arm as cash (0),
including any rally missed by waiting. Unknown setup histories are never cash.
This is an overlapping event study, not a portfolio or executable-fill backtest.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from skills.strategy_scanner.engine import _day
from skills.strategy_scanner.outcomes import COSTS, _net

WAIT_SESSIONS = 10
RALLY_THRESHOLDS = {20: .30, 60: .50}
SETUP_STATES = {'pending', 'retested', 'expired', 'invalidated', 'data_missing'}
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 20261006
OHLC_ROUNDING_RTOL = 1e-12

COLUMNS = [
    'setup_id', 'stock_id', 'formation_date', 'signal_date', 'horizon', 'status',
    'outcome_status', 'mature', 'retest_date', 'direct_entry_date', 'wait_entry_date',
    'entry_date', 'exit_date', 'wait_entered', 'wait_delay_sessions', 'direct_gross',
    'wait_gross', 'direct_net', 'wait_net', 'common_benchmark_net',
    'wait_exposed_benchmark_net', 'delta', 'direct_rally', 'missed_rally',
    'direct_mae', 'direct_mfe', 'wait_mae', 'wait_mfe',
]


def _cluster_interval(frame):
    """Resample whole formation-date clusters; retain event-weighted mean delta.

    Date clustering does not remove overlapping holding-period or same-stock
    dependence. The interval is descriptive, not multiplicity-adjusted evidence.
    """
    clusters = frame.groupby('formation_date', sort=True).delta.agg(['sum', 'count'])
    if len(clusters) < 2:
        return None
    totals, counts = clusters['sum'].to_numpy(), clusters['count'].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        draws[i] = totals[chosen].sum() / counts[chosen].sum()
    low, high = np.quantile(draws, [.025, .975])
    return dict(low=float(low), high=float(high), confidence=.95,
                samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                cluster='formation_date', clusters=len(clusters),
                event_weighted=True, multiple_testing_adjusted=False)


def _summary(frame, horizon, year):
    evaluated = frame[frame.outcome_status.eq('evaluated')]
    executed = evaluated[evaluated.wait_entered.eq(True)]
    counts = Counter(frame.outcome_status)
    rallies = evaluated[evaluated.direct_rally.eq(True)]
    cash = evaluated[evaluated.wait_entered.eq(False)]

    def mean(col, rows=evaluated):
        return float(rows[col].mean()) if len(rows) else None

    def rate(col, rows=evaluated):
        return float(rows[col].gt(0).mean()) if len(rows) else None

    missed = int(rallies.missed_rally.eq(True).sum())
    missing = sum(counts.get(state, 0) for state in (
        'stock_path_missing', 'benchmark_path_missing', 'setup_data_missing'))
    return dict(horizon=horizon, year=year, formations=len(frame), evaluated=len(evaluated),
        immature=counts.get('immature', 0), missing=missing,
        pending=counts.get('setup_pending', 0), outcome_counts=dict(counts),
        setup_status_counts=dict(Counter(frame.status)),
        wait_entered=len(executed), cash_nonentries=len(cash),
        take_rate=len(executed) / len(evaluated) if len(evaluated) else None,
        mean_direct_net=mean('direct_net'), mean_wait_net=mean('wait_net'),
        mean_common_benchmark_net=mean('common_benchmark_net'),
        mean_paired_delta=mean('delta'),
        direct_win_rate=rate('direct_net'), wait_win_rate_including_cash=rate('wait_net'),
        wait_cash_is_win=False,
        direct_beat_benchmark_rate=float((evaluated.direct_net > evaluated.common_benchmark_net).mean()) if len(evaluated) else None,
        wait_beat_benchmark_rate=float((evaluated.wait_net > evaluated.common_benchmark_net).mean()) if len(evaluated) else None,
        executed_only_mean_wait_net=mean('wait_net', executed),
        executed_only_mean_direct_net=mean('direct_net', executed),
        executed_only_mean_common_benchmark_net=mean('common_benchmark_net', executed),
        executed_only_mean_exposed_benchmark_net=mean('wait_exposed_benchmark_net', executed),
        executed_only_wait_win_rate=rate('wait_net', executed),
        executed_only_mean_delay_sessions=mean('wait_delay_sessions', executed),
        mean_direct_mae=mean('direct_mae'), mean_direct_mfe=mean('direct_mfe'),
        executed_only_mean_wait_mae=mean('wait_mae', executed),
        executed_only_mean_wait_mfe=mean('wait_mfe', executed),
        excursion_definition='adjusted_intraday_low_high_vs_entry_open_per_trade_not_account_drawdown',
        direct_rallies=len(rallies), missed_rallies=missed,
        missed_share_of_direct_rallies=missed / len(rallies) if len(rallies) else None,
        rally_gross_threshold=RALLY_THRESHOLDS[horizon],
        paired_delta_cluster_bootstrap=_cluster_interval(evaluated),
        comparison='all_formation_cohort_same_T_plus_h_exit',
        returns_are_fractions=True, costs=dict(COSTS),
        study_type='descriptive_overlapping_event_study_not_portfolio_backtest',
        dependence_warning='Formation-date clustering leaves overlapping-window and repeated-stock dependence.',
        live_qualified=False)


def _validate_inputs(f, days, ids, start, end, horizons):
    days = pd.DatetimeIndex(days)
    ids = list(ids)
    if (not len(days) or days.hasnans or days.tz is not None or days.has_duplicates
            or not days.is_monotonic_increasing or not days.equals(days.normalize())):
        raise ValueError('Complete market calendar must be sorted unique naive dates')
    if len(ids) != len(set(ids)) or '0050' not in ids:
        raise ValueError('Unique stock IDs and 0050 are required for paired comparison')
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or h not in RALLY_THRESHOLDS for h in horizons)):
        raise ValueError('FVG comparison supports unique fixed 20/60-session horizons')
    start, end = _day(start), _day(end)
    if start > end or start not in days or end not in days:
        raise ValueError('Study start/end must be ordered observed market sessions')
    for name in ('valid', 'eligible', 'open', 'close', 'c', 'h', 'l', 'volume'):
        if name not in f or not f[name].index.equals(days) or list(f[name].columns) != ids:
            raise ValueError('Input '+name+' axes must match the complete market calendar')
    cutoff = int(days.get_loc(end)) + 1
    return days[:cutoff], ids, start, end, cutoff


def _formation(setup):
    candidates = [setup[key] for key in ('formation_date', 'setup_date', 'formation')
                  if setup.get(key) is not None]
    if not candidates:
        raise ValueError('FVG setup needs formation_date or setup_date')
    dates = [_day(value) for value in candidates]
    if any(date != dates[0] for date in dates[1:]):
        raise ValueError('Conflicting FVG formation dates')
    return dates[0]


def compare_fvg(f, days, ids, setups, start, end, horizons=(20, 60)):
    """Return summary rows and all FVG formation/horizon outcome rows.

    ``f`` uses the scanner's prepared aligned matrices; ``end`` also bounds all
    price evidence. Setups are supplied by the causal SMC state machine and are
    not selected based on eventual entry, outcome, or account capacity. Waiting
    signals confirm at retest close and enter next session; both arms then exit
    at formation+h close. Expired/invalidated nonentries stay cash with no fees.
    ``status`` preserves setup state; ``outcome_status`` controls comparability.
    """
    days, ids, start, end, cutoff = _validate_inputs(f, days, ids, start, end, horizons)
    open_adj = (f['open'] * (f['c'] / f['close'])).iloc[:cutoff].to_numpy(float)
    closes = f['c'].iloc[:cutoff].to_numpy(float)
    highs = f['h'].iloc[:cutoff].to_numpy(float)
    lows = f['l'].iloc[:cutoff].to_numpy(float)
    volume = f['volume'].iloc[:cutoff].to_numpy(float)
    good = (f['valid'].eq(True) & f['eligible'].eq(True)).iloc[:cutoff].fillna(False).to_numpy(bool)
    good &= np.isfinite(open_adj) & (open_adj > 0) & np.isfinite(closes) & (closes > 0)
    good &= np.isfinite(volume) & (volume > 0)
    # Same raw OHLC equality can differ by one ULP after adjustment. Tolerance
    # is solely a data-consistency check; entry, return and rally rules stay exact.
    tolerance = np.maximum.reduce([np.abs(highs), np.abs(lows), np.abs(closes), np.abs(open_adj)]) * OHLC_ROUNDING_RTOL
    good &= np.isfinite(highs) & np.isfinite(lows) & (lows > 0) & (highs + tolerance >= lows)
    good &= ((highs + tolerance >= closes) & (highs + tolerance >= open_adj)
             & (lows - tolerance <= closes) & (lows - tolerance <= open_adj))
    missing = np.vstack([np.zeros((1, len(ids)), dtype=np.int64), (~good).cumsum(axis=0)])
    column = {sid: i for i, sid in enumerate(ids)}
    benchmark = column['0050']
    locations = {day: i for i, day in enumerate(days)}
    records, seen = [], set()
    for setup in setups:
        if setup.get('kind') != 'fvg':
            continue
        formed = _formation(setup)
        if not start <= formed <= end:
            continue
        sid, setup_id, state = setup.get('stock_id'), setup.get('setup_id'), setup.get('status')
        if sid not in column or not isinstance(sid, str) or sid.startswith('0'):
            raise ValueError('FVG formation must identify an individual stock in source axes')
        if not isinstance(setup_id, str) or not setup_id or setup_id in seen:
            raise ValueError('FVG setup IDs must be nonempty and unique')
        seen.add(setup_id)
        if formed not in locations or state not in SETUP_STATES:
            raise ValueError('FVG formation date or state is invalid')
        anchor, stock = locations[formed], column[sid]
        # Core state is an as-of observation. Reject a caller that supplies a
        # terminal state learned after the requested evidence cutoff.
        state_date_key = {'retested': 'retest_date', 'expired': 'expiry_date',
                          'invalidated': 'invalidation_date',
                          'data_missing': 'data_missing_date'}.get(state)
        if state_date_key and setup.get(state_date_key) is not None:
            observed_at = _day(setup[state_date_key])
            if observed_at not in locations or observed_at <= formed:
                raise ValueError('Setup state date is outside its formation/evidence cutoff')
        retest = _day(setup['retest_date']) if setup.get('retest_date') is not None else None
        if state == 'retested':
            if (retest not in locations or not anchor < locations[retest] <= anchor+WAIT_SESSIONS):
                raise ValueError('Retested FVG needs a confirmed next-10-session retest date within evidence cutoff')
        elif retest is not None:
            raise ValueError('Only retested FVG setups may carry a retest date')
        for h in horizons:
            exit_index, entry_index = anchor+h, anchor+1
            mature = exit_index < len(days)
            wait_index = locations[retest]+1 if retest is not None else None
            row = dict.fromkeys(COLUMNS)
            row.update(setup_id=setup_id, stock_id=sid, formation_date=formed.strftime('%Y-%m-%d'),
                signal_date=formed.strftime('%Y-%m-%d'), horizon=h, status=state,
                mature=mature, retest_date=retest.strftime('%Y-%m-%d') if retest is not None else None,
                direct_entry_date=days[entry_index].strftime('%Y-%m-%d') if entry_index < len(days) else None,
                entry_date=days[entry_index].strftime('%Y-%m-%d') if entry_index < len(days) else None,
                wait_entry_date=days[wait_index].strftime('%Y-%m-%d') if wait_index is not None and wait_index < len(days) else None,
                exit_date=days[exit_index].strftime('%Y-%m-%d') if mature else None,
                wait_entered=wait_index is not None and wait_index < len(days),
                wait_delay_sessions=wait_index-entry_index if wait_index is not None else None)
            if state == 'data_missing':
                row['outcome_status'] = 'setup_data_missing'
            elif not mature:
                row['outcome_status'] = 'immature'
            elif state == 'pending':
                row['outcome_status'] = 'setup_pending'
            elif missing[exit_index+1, stock] - missing[entry_index, stock] > 0:
                row['outcome_status'] = 'stock_path_missing'
            elif missing[exit_index+1, benchmark] - missing[entry_index, benchmark] > 0:
                row['outcome_status'] = 'benchmark_path_missing'
            else:
                direct_gross = closes[exit_index, stock] / open_adj[entry_index, stock]-1
                direct_net = float(_net(open_adj[entry_index, stock], closes[exit_index, stock], COSTS['stock_sell_tax']))
                wait_gross = closes[exit_index, stock] / open_adj[wait_index, stock]-1 if wait_index is not None else 0.
                wait_net = float(_net(open_adj[wait_index, stock], closes[exit_index, stock], COSTS['stock_sell_tax'])) if wait_index is not None else 0.
                common_benchmark = float(_net(open_adj[entry_index, benchmark], closes[exit_index, benchmark], COSTS['benchmark_sell_tax']))
                exposed_benchmark = float(_net(open_adj[wait_index, benchmark], closes[exit_index, benchmark], COSTS['benchmark_sell_tax'])) if wait_index is not None else None
                rally = bool(direct_gross >= RALLY_THRESHOLDS[h])
                row.update(outcome_status='evaluated', direct_gross=float(direct_gross),
                    wait_gross=float(wait_gross), direct_net=direct_net, wait_net=wait_net,
                    common_benchmark_net=common_benchmark, wait_exposed_benchmark_net=exposed_benchmark,
                    delta=wait_net-direct_net, direct_rally=rally,
                    missed_rally=rally and wait_index is None,
                    direct_mae=float(lows[entry_index:exit_index+1, stock].min()/open_adj[entry_index, stock]-1),
                    direct_mfe=float(highs[entry_index:exit_index+1, stock].max()/open_adj[entry_index, stock]-1),
                    wait_mae=float(lows[wait_index:exit_index+1, stock].min()/open_adj[wait_index, stock]-1) if wait_index is not None else None,
                    wait_mfe=float(highs[wait_index:exit_index+1, stock].max()/open_adj[wait_index, stock]-1) if wait_index is not None else None)
            records.append(row)
    frame = pd.DataFrame(records, columns=COLUMNS)
    numeric = ['horizon', 'wait_delay_sessions', 'direct_gross', 'wait_gross', 'direct_net',
               'wait_net', 'common_benchmark_net', 'wait_exposed_benchmark_net', 'delta',
               'direct_mae', 'direct_mfe', 'wait_mae', 'wait_mfe']
    frame[numeric] = frame[numeric].apply(pd.to_numeric)
    summaries = []
    for h in horizons:
        group = frame[frame.horizon.eq(h)]
        summaries.append(_summary(group, h, 'all'))
        for year in range(start.year, end.year+1):
            annual = group[group.formation_date.str.startswith(str(year)).fillna(False)]
            summaries.append(_summary(annual, h, str(year)))
    return summaries, frame
