"""Bounded causal signal/exit interventions and descriptive course diagnostics.

The original engine and its execution assumptions are deliberately unchanged.
These are exploratory comparisons on previously seen historical observations.
"""
import numpy as np
import pandas as pd

from skills.course_bundle import FAMILIES, signals


def clean_signal_close(bars):
    x = bars.copy()
    x['date'] = pd.to_datetime(x.date)
    valid = (x.quality.fillna(False) & x.source_return_known.fillna(False)
             & x.eligible.fillna(False) & ~x.source_disagreement.fillna(False))
    x['value'] = x.adjusted_close.where(valid)
    return x.pivot(index='date', columns='stock_id', values='value').sort_index()


def shift_institutional_availability(frame, calendar):
    """Shift availability by one exchange session, not by next observed record."""
    mapping = dict(zip(calendar[:-1], calendar[1:]))
    x = frame.copy()
    x['date'] = pd.to_datetime(x.date).map(mapping)
    return x.dropna(subset=['date'])


def intervene(features, config, name, clean_close):
    f = dict(features)
    f['ma'] = dict(features['ma'])
    if name == 'momentum_60':
        mean = clean_close.rolling(60, min_periods=60).mean()
        ret = clean_close / clean_close.shift(60) - 1
        bias = clean_close / mean - 1
        common = f['common'] & ret.notna() & bias.notna()
        # Re-ranking the prior money percentiles on a subset preserves their
        # ordering/ties while making all three percentile denominators equal.
        f['ranks'] = [v.where(common).rank(axis=1, pct=True, method='average') * 100
                      for v in (ret, bias, f['ranks'][2])]
        f['common'] = common
        f['trend'] = common & ret.gt(0) & clean_close.gt(mean)
    if name.startswith('market_ma'):
        window = 60 if name == 'market_ma60' else 200
        market = f['close']['0050']
        average = market.dropna().rolling(window, min_periods=window).mean().reindex(market.index)
        f['market'] = f['market'] & market.gt(average) & ~f['halt']['0050']
    if name in ('exit_ma20', 'exit_ma20_confirm2', 'market_ma200_exit_ma20'):
        f['ma'][10] = f['ma'][20]
    elif name == 'exit_ma50':
        f['ma'][10] = clean_close.rolling(50, min_periods=50).mean()
    if name.endswith('confirm2'):
        under = clean_close.lt(f['ma'][10])
        confirmed = under.rolling(2, min_periods=2).sum().eq(2)
        f['ma'][10] = f['ma'][10].where(confirmed)
    sig = signals(f, config)
    ranking = sig['ranking']
    if name.startswith('family_'):
        ranking = ranking.where(sig['flags'][name.removeprefix('family_')])
    elif name == 'rank_momentum':
        ranking = sig['momentum'].where(ranking.notna())
    elif name == 'rank_membership_count':
        count = sum(v.astype(int) for v in sig['flags'].values())
        ranking = (count * 100 + sig['momentum']).where(ranking.notna())
    elif name == 'first_signal_10':
        recent = ranking.notna().shift(1, fill_value=False).rolling(10, min_periods=1).sum().gt(0)
        ranking = ranking.where(~recent)
    return f, dict(sig, ranking=ranking)


def select_training(rows):
    """Only training fields enter selection; absence of a viable arm means cash."""
    good = [r for r in rows if r['status'] == 'completed'
            and r['summary']['completed_trades'] >= 20 and r['summary']['cagr'] > 0]
    if not good:
        return None
    return sorted(good, key=lambda r: (-(r['summary']['cagr']
        - .5 * abs(r['summary']['max_drawdown'])), r['arm']))[0]['arm']


def trade_attribution(trades, calendar, market_close, slippage=.001):
    active, result = {}, []
    locations = {str(d.date()): i for i, d in enumerate(calendar)}
    mean200 = market_close.dropna().rolling(200, min_periods=200).mean().reindex(calendar)
    for row in trades:
        sid = row['stock_id']
        if row['side'] == 'BUY':
            if sid in active:
                raise ValueError('Duplicate open position in attribution')
            active[sid] = row
            continue
        buy = active.pop(sid)
        if not np.isclose(buy['adjusted_units'], row['adjusted_units']):
            raise ValueError('Partial positions require a different attribution ledger')
        units = buy['adjusted_units']
        raw_entry = buy['adjusted_price'] / (1 + slippage)
        raw_exit = row['adjusted_price'] / (1 - slippage)
        gross = units * (raw_exit - raw_entry)
        fee_tax = buy['fee'] + row['fee'] + row['tax']
        slip_cost = units * slippage * (raw_entry + raw_exit)
        net = gross - fee_tax - slip_cost
        if not np.isclose(net, row['net_pnl'], atol=1e-6):
            raise ValueError('Trade PnL fails exact reconciliation')
        signal_date = pd.Timestamp(buy['signal_date'])
        above = market_close.loc[signal_date] > mean200.loc[signal_date]
        result.append(dict(stock_id=sid, entry_date=buy['date'], exit_date=row['date'],
            signal_date=buy['signal_date'], reason=row['reason'],
            holding_sessions=locations[row['date']] - locations[buy['date']],
            net_pnl=net, same_units_gross_pnl=gross, fee_tax=fee_tax, slippage=slip_cost,
            net_return=net/(buy['notional']+buy['fee']),
            entry_market_regime='above_ma200' if above else 'not_above_ma200'))
    return pd.DataFrame(result)


def overlap_diagnostics(sig, start, end):
    eligible = sig['ranking'].loc[start:end].notna()
    flags = {k: v.loc[start:end] & eligible for k, v in sig['flags'].items()}
    count = sum(v.astype(int) for v in flags.values())
    rows = []
    for name, values in flags.items():
        rows.append(dict(family=name, eligible_stock_days=int(values.to_numpy().sum()),
            unique_stock_days=int((values & count.eq(1)).to_numpy().sum()),
            removal_changes_union=bool((values & count.eq(1)).to_numpy().any())))
    pairs = pd.DataFrame(index=FAMILIES, columns=FAMILIES, dtype=int)
    for a in FAMILIES:
        for b in FAMILIES:
            pairs.loc[a, b] = int((flags[a] & flags[b]).to_numpy().sum())
    return rows, pairs.astype(int), dict(eligible_stock_days=int(eligible.to_numpy().sum()),
        membership_distribution={str(n): int((count.eq(n) & eligible).to_numpy().sum()) for n in range(1, 8)})


def forward_signal_study(f, sig, start, end, horizons=(5, 20)):
    """Complete-case signal study, not a portfolio, probability model or causal test."""
    result, censor = [], []
    eligible = sig['ranking'].notna()
    count = sum(v.astype(int) for v in sig['flags'].values())
    score = sig['momentum']
    groups = {'all': eligible, 'count1': count.eq(1), 'count2': count.eq(2),
        'count3': count.eq(3), 'count4plus': count.ge(4), 'M_below70': score.lt(70),
        'M_70_80': score.ge(70) & score.lt(80), 'M_80_90': score.ge(80) & score.lt(90),
        'M_90plus': score.ge(90)}
    opening = f['open'].shift(-1)
    for horizon in horizons:
        endpoint = f['close'].shift(-horizon)
        marks_ok = f['mark_close'].gt(0).rolling(horizon).sum().shift(-horizon).eq(horizon)
        no_conflicts = f['conflict'].astype(int).rolling(horizon).sum().shift(-horizon).eq(0)
        entry_ok = f['trade'].shift(-1, fill_value=False)
        exit_ok = f['trade'].shift(-horizon, fill_value=False)
        valid = marks_ok & no_conflicts & entry_ok & exit_ok & opening.gt(0) & endpoint.gt(0)
        benchmark_valid = valid['0050'].copy()
        valid &= opening.div(f['close']).sub(1).abs().le(.03)
        net = endpoint * .999 * (1 - .001425 - .003) / (opening * 1.001 * (1 + .001425)) - 1
        benchmark = endpoint['0050'] * .999 * (1-.001425-.001) / (opening['0050'] * 1.001 * (1+.001425)) - 1
        valid = valid.mul(benchmark_valid, axis=0).astype(bool)
        excess = net.sub(benchmark, axis=0)
        censor.append(dict(horizon=horizon, eligible=int(eligible.loc[start:end].to_numpy().sum()),
            retained=int((eligible & valid).loc[start:end].to_numpy().sum())))
        for name, group in groups.items():
            mask = (eligible & group & valid).loc[start:end]
            values = net.loc[start:end].where(mask)
            ex = excess.loc[start:end].where(mask)
            flat = values.to_numpy(); flat = flat[np.isfinite(flat)]
            if not len(flat):
                continue
            # Every trading date gets equal weight in this excess statistic.
            daily = ex.mean(axis=1).dropna()
            result.append(dict(horizon=horizon, group=name, stock_days=len(flat),
                signal_dates=len(daily), mean_net_return=float(flat.mean()),
                positive_fraction=float((flat>0).mean()),
                equal_date_mean_excess=float(daily.mean()),
                median_net_return=float(np.median(flat))))
    return result, censor


def paired_block_interval(left, right, *, seed=20261028, repeats=1000, block=20):
    a = left.set_index('date').equity.pct_change().dropna()
    b = right.set_index('date').equity.pct_change().dropna()
    if not a.index.equals(b.index):
        raise ValueError('Paired inference requires identical dates')
    values = (a-b).to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(repeats, int(np.ceil(len(values)/block))))
    indices = (starts[..., None] + np.arange(block)) % len(values)
    samples = values[indices.reshape(repeats, -1)[:, :len(values)]].mean(axis=1)*252
    return dict(annualized_mean_difference=float(values.mean()*252),
        low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)),
        block_sessions=block, replicates=repeats, multiple_testing_adjusted=False)
