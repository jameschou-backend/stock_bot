"""Exploratory monthly rules: known-at-close signals, next-session execution.

No forecasting model, API calls, production writes or parameter search. Adjusted
units track portfolio ratios; this is not a broker-accurate share/cash simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

RULE_NAMES = {'momentum': '中期動能', 'risk_momentum': '波動調整動能', 'near_high': '接近一年新高'}


def scores_for(close: pd.DataFrame, turnover: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every rolling window ends at the signal date; never backfill missing prices."""
    ma120 = close.rolling(120, min_periods=120).mean()
    mom = close.shift(21) / close.shift(126) - 1
    vol = close.pct_change(fill_method=None).rolling(126, min_periods=126).std() * np.sqrt(252)
    liquid = turnover.rolling(20, min_periods=20).mean().ge(50_000_000)
    trend = close.gt(ma120) & mom.gt(0) & liquid
    near = (close.ge(close.rolling(252, min_periods=252).max() * .95)
            & close.rolling(60, min_periods=60).mean().gt(ma120) & liquid)
    ret63 = close / close.shift(63) - 1
    return {'momentum': mom.where(trend),
            'risk_momentum': (mom / vol.where(vol > 0)).where(trend),
            'near_high': ret63.where(near & ret63.gt(0))}


def executable(raw_close, raw_volume, raw_high, raw_low):
    # Full-day single-price bars may be locked-limit bars. Conservatively do not fill.
    return (raw_close.gt(0) & raw_volume.gt(0) & raw_high.gt(raw_low)
            & raw_high.ge(raw_close) & raw_low.le(raw_close))


def metrics(curve: pd.DataFrame, start=None, end=None) -> dict:
    nav = curve.set_index('date')['equity'].sort_index()
    changes = nav.pct_change(fill_method=None).dropna()
    mask = pd.Series(True, index=changes.index)
    if start is not None: mask &= changes.index >= pd.Timestamp(start)
    if end is not None: mask &= changes.index <= pd.Timestamp(end)
    r = changes[mask]
    if r.empty: return {}
    path = np.r_[1., (1+r).cumprod().to_numpy()]
    years = len(r) / 252
    total = path[-1]-1
    return {'start': str(r.index[0].date()), 'end': str(r.index[-1].date()),
            'total_return': float(total), 'annualized_return': float(path[-1]**(1/years)-1),
            'max_drawdown': float((path / np.maximum.accumulate(path)-1).min()),
            'sharpe': float((r.mean()-.015/252)/r.std()*np.sqrt(252)) if r.std()>0 else None}


@dataclass
class Simulation:
    summary: dict
    curve: pd.DataFrame
    trades: list
    decisions: list


def simulate(close: pd.DataFrame, can_trade: pd.DataFrame, scores: pd.DataFrame | None,
             *, start='2018-01-01', topn=10, slippage=.003, benchmark=None) -> Simulation:
    if topn < 1 or not 0 <= slippage < 1: raise ValueError('Invalid simulation configuration')
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError('Dates must be sorted and unique')
    if not close.index.equals(can_trade.index) or not close.columns.equals(can_trade.columns):
        raise ValueError('Trade flags and adjusted prices must align')
    if scores is not None and (not scores.index.equals(close.index) or not scores.columns.equals(close.columns)):
        raise ValueError('Scores and prices must align')
    first = int(close.index.searchsorted(pd.Timestamp(start)))
    if first == 0 or first >= len(close)-1: raise ValueError('Need pre-start data and at least two evaluation sessions')
    days, ids = close.index, close.columns.to_numpy()
    px = close.to_numpy(float)
    tradable = can_trade.to_numpy(bool) & np.isfinite(px) & (px > 0)
    values = scores.to_numpy(float) if scores is not None else None
    if benchmark is not None and benchmark not in close.columns: raise ValueError('Missing benchmark')
    if benchmark is not None and not tradable[first,close.columns.get_loc(benchmark)]:
        raise ValueError('Benchmark cannot execute on the common start date; never substitute cash')
    count = len(ids)
    units, last_px = np.zeros(count), np.zeros(count)
    valid_before = close.iloc[:first].ffill().iloc[-1].to_numpy(float)
    last_px[np.isfinite(valid_before)] = valid_before[np.isfinite(valid_before)]
    cash, costs, traded_value, turns = 1., 0., 0., 0
    trades, decisions = [], []
    curve = [{'date': days[first-1], 'equity': 1., 'cash': 1., 'positions': 0}]
    pending = set()
    missing_hold_days, blocked_orders = 0, 0
    buy_cost, sell_cost = .001425+slippage, .001425+slippage+(.001 if benchmark else .003)

    def sell(i, j, amount):
        nonlocal cash, costs, traded_value, turns
        amount = min(amount, units[j]*px[i,j])
        if amount <= 1e-12: return
        units[j] -= amount/px[i,j]
        cash += amount*(1-sell_cost)
        costs += amount*sell_cost
        traded_value += amount
        turns += 1
        trades.append({'date': str(days[i].date()), 'stock_id': str(ids[j]), 'side': 'sell',
                       'notional_initial_equity': amount, 'cost_initial_equity': amount*sell_cost})

    for i in range(first, len(days)):
        observed = np.isfinite(px[i]) & (px[i]>0)
        missing_hold_days += int(((units>1e-12) & ~observed).sum())
        last_px[observed] = px[i,observed]
        equity = cash+float(np.dot(units,last_px))
        rebalance = i == first or (benchmark is None and days[i].to_period('M') != days[i-1].to_period('M'))
        if rebalance:
            if benchmark:
                chosen = [int(close.columns.get_loc(benchmark))]
            else:
                # Only yesterday's scores choose securities; execution-day bars cannot change ranking.
                eligible = np.flatnonzero(np.isfinite(values[i-1]))
                chosen = sorted(eligible,key=lambda j:(-values[i-1,j],str(ids[j])))[:topn]
            target = np.zeros(count)
            target[chosen] = equity/(1 if benchmark else topn)
            pending = {int(j) for j in np.flatnonzero((units>1e-12)&(target==0))}
            decisions.append({'signal_date':str(days[i-1].date()),'execution_date':str(days[i].date()),
                              'stocks':[str(ids[j]) for j in chosen]})
            for j in np.flatnonzero(units>1e-12):
                excess = units[j]*last_px[j]-target[j]
                if excess > 1e-12:
                    if tradable[i,j]: sell(i,j,excess)
                    else: blocked_orders += 1
            for j in chosen:
                wanted = max(0.,target[j]-units[j]*last_px[j])
                if wanted<=1e-12: continue
                # A blocked exit still occupies a slot; it cannot fund an eleventh holding.
                if units[j]<=1e-12 and int((units>1e-12).sum()) >= (1 if benchmark else topn):
                    blocked_orders += 1
                    continue
                if not tradable[i,j]:
                    blocked_orders += 1
                    continue
                amount = min(wanted, max(0,cash)/(1+buy_cost))
                units[j] += amount/px[i,j]
                cash -= amount*(1+buy_cost)
                costs += amount*buy_cost
                traded_value += amount
                turns += 1
                trades.append({'date':str(days[i].date()),'stock_id':str(ids[j]),'side':'buy',
                               'notional_initial_equity':amount,'cost_initial_equity':amount*buy_cost})
        else:
            for j in list(pending):
                if units[j]>1e-12 and tradable[i,j]: sell(i,j,units[j]*px[i,j])
        pending = {j for j in pending if units[j]>1e-12}
        # Report terminal realized liquidation only where execution is possible.
        if i==len(days)-1:
            for j in np.flatnonzero(units>1e-12):
                if tradable[i,j]: sell(i,j,units[j]*px[i,j])
        curve.append({'date':days[i],'equity':cash+float(np.dot(units,last_px)),
                      'cash':cash,'positions':int((units>1e-12).sum())})
        if cash < -1e-10: raise AssertionError('Simulation cannot borrow cash')

    frame = pd.DataFrame(curve)
    summary = metrics(frame)
    summary.update(trades=turns, traded_notional_initial_equity=traded_value,
                   fees_initial_equity=costs, average_positions=float(frame.positions.iloc[1:].mean()),
                   blocked_orders=blocked_orders, missing_hold_days=missing_hold_days,
                   unliquidated_positions=int((units>1e-12).sum()),
                   annual_returns={str(y):metrics(frame,f'{y}-01-01',f'{y}-12-31')['total_return']
                                   for y in sorted(set(days[first:].year))})
    return Simulation(summary,frame,trades,decisions)
