"""Frozen 600-lot holder stability hypothesis, independent of production rules."""
from __future__ import annotations
import numpy as np
import pandas as pd
from skills.theme_chips import aggregate_week, changes, align

ARMS = ('drawdown', 'stable', 'confirmed', 'stable_confirmed', 'decreasing', 'decreasing_confirmed')
NAMED = ('6442', '6223')
LAGS = (8, 15)
EXIT_RULES = ('5', '20', '60', 'stop20')
COSTS = dict(fee=.001425, stock_tax=.003, benchmark_tax=.001, slippage=.0045, stress_slippage=.009)


def aggregate_600(raw, day, cohort):
    result = aggregate_week(raw, day, cohort)
    if raw.empty:
        return result
    frame = raw.copy()
    frame['stock_id'] = frame.stock_id.astype(str)
    labels = frame.HoldingSharesLevel.astype(str).str.replace(',', '', regex=False).str.strip().str.lower()
    selected = labels.isin(['600001-800000', '800001-1000000', '1000001+', 'more than 1000001', 'over 1000001'])
    units = pd.to_numeric(frame.unit, errors='coerce').where(selected).groupby(frame.stock_id).sum(min_count=3)
    result['large_units'] = result.stock_id.map(units).where(result.valid)
    result['large_pct'] = result.large_units / result.total_units
    return result


def holder_changes(weekly):
    result = changes(weekly)
    grouped = result.groupby('stock_id', sort=False)
    prior = grouped.large_units.shift(4)
    whole_change = (result.total_units / grouped.total_units.shift(4) - 1).abs()
    reject = whole_change.gt(.01)
    result.loc[reject, 'change_known'] = False
    result.loc[reject, 'change_reason'] = 'four_week_inventory_denominator_change'
    result['large_units_relative4'] = (result.large_units / prior.where(prior.gt(0)) - 1).where(result.change_known)
    result['change_known'] &= np.isfinite(result.large_units_relative4)
    result.loc[~result.change_known, ['large_pct_delta4', 'large_units_delta4', 'large_units_relative4']] = np.nan
    return result


def with_holders(observations, weekly, lag):
    result = align(observations, weekly, lag)
    known = result.chip_known & np.isfinite(result.large_units_relative4)
    result['chip_known'] = known
    result['holder_state'] = np.select(
        [~known, result.large_pct_delta4.ge(-.005) & result.large_units_relative4.ge(-.01),
         result.large_pct_delta4.lt(-.005) & result.large_units_relative4.lt(-.01)],
        ['unknown', 'stable', 'decreasing'], default='mixed')
    base = result.price_setup & known
    stable, decreasing = result.holder_state.eq('stable'), result.holder_state.eq('decreasing')
    confirmed = result.price_confirmed
    for name, mask in zip(ARMS, [base, base & stable, base & confirmed, base & stable & confirmed,
                                 base & decreasing, base & decreasing & confirmed]):
        result[name] = mask.fillna(False)
    result['holder_lag'] = lag
    return result


def observations(f, days, companies, *, start, end):
    """All stocks on fixed weekly anchors; no holder or outcome inputs accepted."""
    c = f['c']
    valid = f['valid'] & f['eligible'].eq(True).fillna(False)
    complete = valid.rolling(60, min_periods=60).sum().eq(60)
    high = c.rolling(60, min_periods=60).max()
    drawdown = c / high - 1
    prior5 = c.shift(1).rolling(5, min_periods=5).max()
    stop = c.shift(1).rolling(5, min_periods=5).min()
    amount = f['a'].rolling(20, min_periods=20).mean()
    listed = companies.set_index('stock_id').listed_date
    age = pd.DataFrame(False, index=days, columns=c.columns)
    for sid in c:
        if sid == '0050':
            age.loc[:, sid] = True
        else:
            first = days.searchsorted(pd.Timestamp(listed[sid]))
            age.loc[:, sid] = np.arange(len(days)) - first >= 126
    known = complete & age & np.isfinite(amount)
    setup = known & drawdown.le(-.20) & amount.ge(50_000_000)
    confirms = c.gt(prior5) & known
    anchors = pd.Series(days, index=days).groupby(days.to_period('W-FRI')).max()
    anchors = anchors[(anchors >= pd.Timestamp(start)) & (anchors <= pd.Timestamp(end))]
    # Include the requested final observed session even in a partial market week.
    anchors = pd.DatetimeIndex(sorted(set(anchors) | {pd.Timestamp(end)}))
    ids = [sid for sid in c if not sid.startswith('0')]
    keys = pd.MultiIndex.from_product([anchors, ids], names=['signal_date', 'stock_id'])
    result = keys.to_frame(index=False)
    fields = dict(price_known=known, price_setup=setup, price_confirmed=confirms,
                  drawdown60=drawdown, amount20=amount, stop_close5=stop,
                  adjusted_close=c, close=f['close'], prior_high5=prior5)
    for name, frame in fields.items():
        result[name] = frame.loc[anchors, ids].to_numpy().reshape(-1)
    meta = companies.set_index('stock_id')
    result['name'] = result.stock_id.map(meta['name'])
    result['industry'] = result.stock_id.map(meta['industry']).fillna('UNKNOWN')
    result['named_case'] = result.stock_id.isin(NAMED)
    result['signal_date'] = result.signal_date.dt.strftime('%Y-%m-%d')
    result['event_id'] = result.signal_date + ':' + result.stock_id
    return result


def net_return(entry, exit_price, tax, slippage):
    return exit_price * (1-slippage) * (1-COSTS['fee']-tax) / (entry*(1+slippage)*(1+COSTS['fee'])) - 1


def event_outcomes(f, days, events):
    """Daily-price event replay, not a finite-capital or fill-certified account."""
    ids = list(f['c'].columns)
    position = {str(d.date()): i for i, d in enumerate(days)}
    columns = {sid: j for j, sid in enumerate(ids)}
    bm = columns['0050']
    close = f['c'].to_numpy(float)
    opening = (f['open'] * f['c'] / f['close']).to_numpy(float)
    low = f['l'].to_numpy(float)
    good = (f['valid'] & f['eligible'].eq(True).fillna(False)).to_numpy(bool)
    good &= np.isfinite(opening) & (opening > 0) & np.isfinite(close) & (close > 0)
    missing = np.vstack([np.zeros((1, len(ids)), dtype=np.int64), (~good).cumsum(axis=0)])
    rows = []
    for event in events.to_dict('records'):
        i, j = position[event['signal_date']], columns[event['stock_id']]
        entry = i + 1
        for rule in EXIT_RULES:
            reason = 'fixed_horizon'
            opening_exit = rule == 'stop20'
            exit_day = i + int(rule) if not opening_exit else i + 21
            status = 'evaluated'
            if opening_exit and entry < len(days):
                if good[entry, j] and opening[entry, j] <= event['stop_close5']:
                    status = 'entry_below_stop'
                hits = np.flatnonzero(close[entry:min(i+21, len(days)), j] < event['stop_close5'])
                if len(hits):
                    exit_day = entry + int(hits[0]) + 1
                    reason = 'close_stop_next_open'
                else:
                    reason = 'twenty_sessions_next_open'
            mature = entry < len(days) and exit_day < len(days)
            if status != 'entry_below_stop':
                if not mature:
                    status = 'immature'
                elif missing[exit_day+1, j] - missing[entry, j]:
                    status = 'stock_path_missing'
                elif missing[exit_day+1, bm] - missing[entry, bm]:
                    status = 'benchmark_path_missing'
            out = dict(event_id=event['event_id'], signal_date=event['signal_date'], stock_id=event['stock_id'],
                       rule=rule, status=status, exit_reason=reason,
                       entry_date=str(days[entry].date()) if entry < len(days) else None,
                       exit_date=str(days[exit_day].date()) if mature else None,
                       gross_return=None, net_return=None, stress_net_return=None,
                       benchmark_net_return=None, benchmark_stress_net_return=None,
                       excess_return=None, stress_excess_return=None, mae=None, holding_sessions=None)
            if status == 'evaluated':
                ep, xp = opening[entry, j], (opening if opening_exit else close)[exit_day, j]
                be, bx = opening[entry, bm], (opening if opening_exit else close)[exit_day, bm]
                # Exit-at-open MAE excludes the rest of that day's price range.
                lows = low[entry:exit_day if opening_exit else exit_day+1, j]
                out.update(gross_return=float(xp/ep-1), mae=float(min(0., float(lows.min()/ep-1), float(xp/ep-1))),
                           holding_sessions=exit_day-entry+(0 if opening_exit else 1))
                for prefix, slip in [('', COSTS['slippage']), ('stress_', COSTS['stress_slippage'])]:
                    own = net_return(ep, xp, COSTS['stock_tax'], slip)
                    benchmark = net_return(be, bx, COSTS['benchmark_tax'], slip)
                    out[prefix+'net_return'] = float(own)
                    out['benchmark_'+prefix+'net_return'] = float(benchmark)
                    out[prefix+'excess_return'] = float(own-benchmark)
            out['phase'] = ('2022_2024' if out['exit_date'] and out['exit_date'] <= '2024-12-31'
                            else '2025_2026' if event['signal_date'] >= '2025-01-01' else 'boundary')
            rows.append(out)
    return pd.DataFrame(rows)


def summarize(aligned, outcomes):
    if aligned.duplicated(['event_id', 'holder_lag']).any() or outcomes.duplicated(['event_id', 'rule']).any():
        raise ValueError('Duplicate event/lag or event/exit rule')
    table = aligned[~aligned.named_case].merge(outcomes, on=['event_id', 'signal_date', 'stock_id'], validate='many_to_many')
    rows = []
    for lag in LAGS:
        for arm in ARMS:
            arm_rows = table[table.holder_lag.eq(lag) & table[arm]]
            for rule in EXIT_RULES:
                group = arm_rows[arm_rows.rule.eq(rule)]
                for phase in ('all', '2022_2024', '2025_2026', 'boundary'):
                    part = group if phase == 'all' else group[group.phase.eq(phase)]
                    good = part[part.status.eq('evaluated')]
                    for cost, prefix in [('main', ''), ('stress', 'stress_')]:
                        rows.append(dict(lag=lag, arm=arm, rule=rule, phase=phase, cost=cost,
                            events=len(part), evaluated=len(good), status_counts=part.status.value_counts().to_dict(),
                            stocks=part.stock_id.nunique(), dates=part.signal_date.nunique(),
                            mean_net=float(good[prefix+'net_return'].mean()) if len(good) else None,
                            median_net=float(good[prefix+'net_return'].median()) if len(good) else None,
                            win_rate=float(good[prefix+'net_return'].gt(0).mean()) if len(good) else None,
                            mean_benchmark=float(good['benchmark_'+prefix+'net_return'].mean()) if len(good) else None,
                            mean_excess=float(good[prefix+'excess_return'].mean()) if len(good) else None,
                            mean_mae=float(good.mae.mean()) if len(good) else None,
                            worst_mae=float(good.mae.min()) if len(good) else None,
                            median_holding_sessions=float(good.holding_sessions.median()) if len(good) else None))
    return rows


def matched_pairs(aligned):
    rows = []
    base = aligned[~aligned.named_case & aligned.drawdown]
    for keys, part in base.groupby(['holder_lag', 'signal_date', 'industry', 'price_confirmed'], sort=True):
        stable, control = part[part.holder_state.eq('stable')], part[part.holder_state.eq('decreasing')].sort_values('stock_id')
        scale = np.array([part.drawdown60.std(ddof=0), np.log(part.amount20).std(ddof=0)])
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
        for r in stable.to_dict('records'):
            match, distance = None, None
            if len(control):
                values = np.column_stack([control.drawdown60-r['drawdown60'], np.log(control.amount20/r['amount20'])]) / scale
                dist = np.square(values).sum(axis=1)
                index = int(np.argmin(dist))
                match, distance = control.iloc[index].event_id, float(dist[index])
            rows.append(dict(holder_lag=keys[0], signal_date=keys[1], event_id=r['event_id'],
                             control_id=match, distance=distance, price_confirmed=bool(keys[3])))
    return pd.DataFrame(rows, columns=['holder_lag', 'signal_date', 'event_id', 'control_id', 'distance', 'price_confirmed'])


def paired_statistics(pairs, outcomes):
    own = pairs.merge(outcomes, on=['event_id', 'signal_date'], validate='many_to_many')
    control = outcomes[['event_id', 'rule', 'status', 'net_return', 'stress_net_return']].rename(
        columns={'event_id': 'control_id', 'status': 'control_status', 'net_return': 'control_net', 'stress_net_return': 'control_stress'})
    joined = own.merge(control, on=['control_id', 'rule'], how='left', validate='many_to_one')
    rows = []
    for keys, part in joined.groupby(['holder_lag', 'rule', 'price_confirmed', 'phase'], sort=True):
        good = part[part.status.eq('evaluated') & part.control_status.eq('evaluated')].copy()
        for cost, field, other in [('main', 'net_return', 'control_net'), ('stress', 'stress_net_return', 'control_stress')]:
            good['delta'] = good[field] - good[other]
            dates = good.groupby('signal_date').delta.mean().to_numpy()
            lo = hi = None
            if len(dates) >= 5:
                rng = np.random.default_rng(20261102)
                means = rng.choice(dates, size=(1000, len(dates)), replace=True).mean(axis=1)
                lo, hi = [float(x) for x in np.quantile(means, [.025, .975])]
            rows.append(dict(lag=keys[0], rule=keys[1], confirmed=bool(keys[2]), phase=keys[3], cost=cost,
                             events=len(part), matched=int(part.control_id.notna().sum()), evaluated=len(good),
                             dates=len(dates), mean_delta=float(good.delta.mean()) if len(good) else None,
                             date_mean_delta=float(dates.mean()) if len(dates) else None,
                             descriptive_low=lo, descriptive_high=hi))
    return rows
