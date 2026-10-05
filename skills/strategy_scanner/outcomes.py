"""Separate, descriptive forward outcomes; never used to construct a signal.

Not a portfolio backtest: overlapping events have no capital, fill, board-lot or
cash constraints. Adjusted open/close ratios are return proxies. Bad or missing
holding-path data are counted explicitly, never silently shortened or imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import _compile_rules, _day, _prepare

COSTS = dict(buy_fee=.001425, sell_fee=.001425, buy_slippage=.001,
             sell_slippage=.001, stock_sell_tax=.003, benchmark_sell_tax=.001)


def _net(entry, exit_price, tax):
    return (exit_price * (1-COSTS['sell_slippage']) * (1-COSTS['sell_fee']-tax)
            / (entry * (1+COSTS['buy_slippage']) * (1+COSTS['buy_fee']))) - 1


def measure_events(f, days, ids, event_mask, *, start, end, horizons=(5, 20, 60)):
    """Measure pre-existing event coordinates with T+1 open/T+h close only."""
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or h < 1 for h in horizons)):
        raise ValueError('Horizons must be unique positive market-session integers')
    if '0050' not in ids:
        raise ValueError('0050 is required for a paired benchmark comparison')
    if not event_mask.index.equals(days) or list(event_mask.columns) != list(ids):
        raise ValueError('Event axes must match the complete market calendar')
    if (not all(pd.api.types.is_bool_dtype(dtype) for dtype in event_mask.dtypes)
            or event_mask.isna().to_numpy().any()):
        raise ValueError('Events must be known booleans, not missing or numeric flags')
    selected = event_mask.to_numpy(dtype=bool).copy()
    selected[(days < _day(start)) | (days > _day(end))] = False
    selected[:, [i for i, sid in enumerate(ids) if sid.startswith('0')]] = False
    rows, cols = np.where(selected)
    # The same-day adjustment factor is a return proxy, not actual broker cash.
    open_adj = (f['open'] * f['c'] / f['close']).to_numpy(float)
    closes = f['c'].to_numpy(float)
    good = (f['valid'] & f['eligible'].eq(True).fillna(False)).to_numpy(bool)
    good &= np.isfinite(open_adj) & (open_adj > 0) & np.isfinite(closes) & (closes > 0)
    good &= np.isfinite(f['volume'].to_numpy(float)) & (f['volume'].to_numpy(float) > 0)
    # Cumulative missing counts detect every session, not just the endpoints.
    missing = np.vstack([np.zeros((1, len(ids)), dtype=np.int64), (~good).cumsum(axis=0)])
    benchmark = ids.index('0050')
    outputs = []
    for h in horizons:
        exits = rows+h
        mature = exits < len(days)
        bounded = np.minimum(exits, len(days)-1)
        entries = np.minimum(rows+1, len(days)-1)
        stock_ok = mature & ((missing[bounded+1, cols] - missing[rows+1, cols]) == 0)
        bench_ok = mature & ((missing[bounded+1, benchmark] - missing[rows+1, benchmark]) == 0)
        paired = stock_ok & bench_ok
        net = np.full(len(rows), np.nan)
        gross = np.full(len(rows), np.nan)
        bench = np.full(len(rows), np.nan)
        gross[stock_ok] = closes[bounded[stock_ok], cols[stock_ok]] / open_adj[entries[stock_ok], cols[stock_ok]] - 1
        net[stock_ok] = _net(open_adj[entries[stock_ok], cols[stock_ok]], closes[bounded[stock_ok], cols[stock_ok]], COSTS['stock_sell_tax'])
        bench[bench_ok] = _net(open_adj[entries[bench_ok], benchmark], closes[bounded[bench_ok], benchmark], COSTS['benchmark_sell_tax'])
        status = np.where(~mature, 'immature', np.where(~stock_ok, 'stock_path_missing',
                          np.where(~bench_ok, 'benchmark_path_missing', 'evaluated')))
        outputs.append(pd.DataFrame(dict(
            signal_date=days[rows].strftime('%Y-%m-%d'), stock_id=np.asarray(ids)[cols], horizon=h,
            entry_date=np.where(rows+1 < len(days), days[entries].strftime('%Y-%m-%d'), None),
            exit_date=np.where(mature, days[bounded].strftime('%Y-%m-%d'), None),
            status=status, gross_return=gross, net_return=net, benchmark_net_return=bench,
            excess_vs0050=np.where(paired, net-bench, np.nan))))
    return pd.concat(outputs, ignore_index=True)


def _stats(frame):
    ok = frame[frame.status.eq('evaluated')]
    def stat(column, method):
        return float(getattr(ok[column], method)()) if len(ok) else None
    counts = frame.status.value_counts().to_dict()
    return dict(events=len(frame), evaluated=len(ok), immature=counts.get('immature', 0),
        stock_path_missing=counts.get('stock_path_missing', 0),
        benchmark_path_missing=counts.get('benchmark_path_missing', 0),
        win_rate=float(ok.net_return.gt(0).mean()) if len(ok) else None,
        mean_gross_return=stat('gross_return', 'mean'), mean_net_return=stat('net_return', 'mean'),
        median_net_return=stat('net_return', 'median'),
        mean_benchmark_net_return=stat('benchmark_net_return', 'mean'),
        mean_excess_vs0050=stat('excess_vs0050', 'mean'),
        beat_benchmark_rate=float(ok.excess_vs0050.gt(0).mean()) if len(ok) else None)


def study_signals(bars, calendar, *, start, end, original_signals=(), poc=None,
                  provenance=None, strategies=None, horizons=(5, 20, 60)):
    """Evaluate all entry rules once with fixed horizons, no outcome-based tuning."""
    start, end = _day(start), _day(end)
    if start > end:
        raise ValueError('Study start exceeds end')
    if provenance and provenance.get('source_end') and end > _day(provenance['source_end']):
        raise ValueError('Study exceeds frozen source coverage')
    f, days, ids = _prepare(bars, calendar, end)
    if start not in days or end not in days:
        raise ValueError('Study endpoints must be observed market sessions')
    z, masks, catalog, evidence = _compile_rules(f, days, ids,
        original_signals=original_signals, poc=poc, provenance=provenance)
    entries = {s['id']: s for s in catalog if s['status']=='active' and s['kind']=='entry'}
    chosen = list(entries if strategies is None else strategies)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen)-set(entries):
        raise ValueError('Study requires unique active entry strategies, not filters or rankings')
    # Free display-only matrices before materializing event outcomes.
    del z, evidence
    eligible = f['eligible'].eq(True).fillna(False)
    records, summary, counts = [], [], {}
    in_window = (days >= start) & (days <= end)
    individual = [sid for sid in ids if not sid.startswith('0')]
    for sid in chosen:
        match, known, fields, rule = masks[sid]
        available = known & eligible
        previous_known = available.shift(1, fill_value=False)
        matched = match & available
        first = matched & previous_known & ~match.shift(1, fill_value=False)
        counts[sid] = dict(
            matching_stock_days=int(matched.loc[in_window, individual].to_numpy().sum()),
            first_events=int(first.loc[in_window, individual].to_numpy().sum()),
            matched_prior_unknown=int((matched & ~previous_known).loc[in_window, individual].to_numpy().sum()),
            unknown_stock_days=int((~available & ~f['eligible'].eq(False).fillna(False)).loc[in_window, individual].to_numpy().sum()))
        results = measure_events(f, days, ids, first, start=start, end=end, horizons=horizons)
        results.insert(0, 'strategy_id', sid)
        records.append(results)
        for h in horizons:
            group = results[results.horizon.eq(h)]
            summary.append(dict(strategy_id=sid, name=entries[sid]['name'], version=entries[sid]['version'],
                                horizon=h, year='all', **_stats(group)))
            for year in range(start.year, end.year+1):
                subset = group[group.signal_date.str.startswith(str(year))]
                summary.append(dict(strategy_id=sid, name=entries[sid]['name'], version=entries[sid]['version'],
                                    horizon=h, year=str(year), **_stats(subset)))
    return dict(schema='strategy_scanner_signal_outcomes_v1', start=str(start.date()), end=str(end.date()),
        horizons=list(horizons), costs=COSTS, strategies=chosen,
        strategy_definitions=[entries[sid] for sid in chosen], summary=summary, signal_counts=counts,
        hypothesis_count=len(chosen)*len(horizons), source_provenance=provenance or {},
        signal_policy='known_first_day_only_T_close', entry_price='T+1_adjusted_open_proxy',
        exit_price='T+h_adjusted_close_proxy_including_entry_session',
        missing_policy='complete_valid_stock_and_benchmark_holding_paths_required',
        study_type='descriptive_overlapping_event_study_not_portfolio_backtest',
        execution_capacity_verified=False, historical_period_already_researched=True,
        multiple_testing_adjusted=False, live_qualified=False, cumulative_return=None,
        max_drawdown=None, account_independent=True), pd.concat(records, ignore_index=True)
