"""Causal, course-derived long-only signals and a daily adjusted-unit NAV model.

The score is our disclosed hypothesis, not the author's proprietary indicator.
Corporate cash timing, integer shares and queue priority are not modeled.
"""
from dataclasses import dataclass, asdict
from itertools import product
import numpy as np
import pandas as pd

WEIGHTS = ((.4, .4, .2), (.6, .2, .2), (.3, .3, .4))
FAMILIES = ('volume_momentum', 'trust_near_breakout', 'trust_consecutive',
            'foreign_first', 'foreign_trust', 'price_volume_400', 'near_breakout')


@dataclass(frozen=True)
class Config:
    weights: int = 0
    threshold: int = 70
    exit_ma: int = 10
    entries: str = 'both'
    baseline: bool = False

    @property
    def name(self):
        return f'w{self.weights}_m{self.threshold}_ma{self.exit_ma}_{self.entries}' + ('_momentum_only' if self.baseline else '')


def configurations():
    return [Config(*values) for values in product(range(3), (70, 80), (5, 10), ('breakout', 'both'))]


def macd_hist(close):
    dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    return dif - dif.ewm(span=9, adjust=False, min_periods=9).mean()


def prepare_features(bars, institutional, turnover, *, halts=()):
    """Missing inputs remain missing. Every observation is dated at its close."""
    bars = bars.copy()
    bars['date'] = pd.to_datetime(bars.date)
    if bars.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate stock/date bars')
    def matrix(col):
        return bars.pivot(index='date', columns='stock_id', values=col).sort_index()
    raw = {key: matrix(key).astype(float) for key in ('open', 'high', 'low', 'close', 'volume', 'adjusted_close')}
    days, ids = raw['close'].index, raw['close'].columns
    if '0050' not in ids:
        raise ValueError('0050 market proxy and benchmark required')
    eligible = matrix('eligible').fillna(False).astype(bool)
    conflict = matrix('source_disagreement').fillna(False).astype(bool)
    usable = matrix('quality').fillna(False).astype(bool) & ~conflict
    known_source = matrix('source_return_known').fillna(False).astype(bool)
    signal_quality = usable & known_source & eligible
    close = raw['adjusted_close'].where(signal_quality)
    factor = raw['adjusted_close'] / raw['close'].where(raw['close'] > 0)
    adj = {key: raw[key] * factor for key in ('open', 'high', 'low')}
    volume = raw['volume'].where(signal_quality)

    def align(frame, names):
        frame = frame.copy()
        frame['date'] = pd.to_datetime(frame.date)
        if frame.duplicated(['date', 'stock_id']).any():
            raise ValueError('Duplicate auxiliary input')
        return {name: frame.pivot(index='date', columns='stock_id', values=name).reindex(index=days, columns=ids).astype(float) for name in names}
    inst = align(institutional, ('trust_net', 'foreign_net'))
    money = align(turnover, ('Trading_money', 'Trading_Volume', 'close'))
    # A revised or differently scaled quote cannot silently supply our turnover.
    matching = np.isclose(money['close'], raw['close'], rtol=0, atol=1e-5, equal_nan=False)
    matching &= money['Trading_Volume'].eq(raw['volume'])
    amount = money['Trading_money'].where(matching & money['Trading_money'].ge(0))
    trust, foreign = inst['trust_net'], inst['foreign_net']
    institutional_known = trust.notna().rolling(3, min_periods=3).sum().eq(3) & foreign.notna().rolling(3, min_periods=3).sum().eq(3)
    ma = {n: close.rolling(n, min_periods=n).mean() for n in (5, 10, 20)}
    ret20 = close / close.shift(20) - 1
    bias = close / ma[20] - 1
    money_ratio = amount / amount.shift(1).rolling(20, min_periods=20).mean().where(lambda x: x > 0)
    liquid = amount.shift(1).gt(500_000_000)
    high40 = adj['high'].where(signal_quality).shift(1).rolling(40, min_periods=40).max()
    high400 = adj['high'].where(signal_quality).shift(1).rolling(400, min_periods=400).max()
    vol20 = volume.shift(1).rolling(20, min_periods=20).mean()
    vol9 = volume.shift(1).rolling(9, min_periods=9).max()
    common = (signal_quality & institutional_known & liquid & ret20.notna()
              & bias.notna() & money_ratio.notna() & high40.notna() & vol20.notna())
    common.loc[:, '0050'] = False
    ranks = [x.where(common).rank(axis=1, pct=True, method='average') * 100 for x in (ret20, bias, money_ratio)]
    hist = macd_hist(close)
    market_close = raw['adjusted_close']['0050'].where(usable['0050'])
    market_hist = macd_hist(market_close)
    market = market_close.gt(market_close.rolling(5, min_periods=5).mean()) & market_hist.gt(market_hist.shift(1))
    trend = common & ret20.gt(0) & close.gt(ma[20])
    red = close.gt(adj['open'])
    volume_confirm = volume.ge(vol20 * 1.2)
    breakout = close.gt(high40) & red & volume_confirm
    pullback = adj['low'].le(ma[10] * 1.02) & close.ge(ma[10]) & close.le(ma[10] * 1.03)
    pullback &= red & close.gt(close.shift(1)) & hist.gt(hist.shift(1))
    distance40 = close / high40 - 1
    near40 = distance40.ge(-.02) & distance40.le(.02)
    valid_trade = (raw['volume'].gt(0) & raw['high'].gt(raw['low']) & raw['open'].gt(0)
                   & raw['high'].ge(raw['open']) & raw['low'].le(raw['open']))
    halt_mask = pd.DataFrame(False, index=days, columns=ids)
    for row in halts:
        if row['stock_id'] in ids:
            halt_mask.loc[(days>=pd.Timestamp(row['start'])) & (days<pd.Timestamp(row['end'])), row['stock_id']] = True
    halt_conflict = (raw['volume'].gt(0) | raw['close'].gt(0)) & halt_mask
    # An overlong/mismatched historical suspension interval is not usable
    # execution evidence. Quarantine those coordinates, and halt if held.
    conflict |= halt_conflict
    common &= ~halt_conflict
    trend &= ~halt_conflict
    # Providers encode a verified halt either as a zero bar or an absent bar.
    verified_zero_halt = halt_mask & raw['volume'].fillna(0).eq(0) & raw['close'].fillna(0).eq(0)
    # Carry a valuation mark only on independently documented full-session halts.
    # Signal histories and raw execution bars remain missing/zero.
    mark_close = raw['adjusted_close'].where(~verified_zero_halt, raw['adjusted_close'].ffill())
    valid_trade &= ~halt_mask
    audit = dict(universe_stocks=len(ids) - 1, calendar_start=str(days[0].date()), calendar_end=str(days[-1].date()),
        price_observations=int(usable.to_numpy().sum()),
        actual_turnover_observations=int(amount.notna().to_numpy().sum()),
        turnover_quote_mismatch=int((money['Trading_money'].notna() & ~matching).to_numpy().sum()),
        institutional_known_observations=int(institutional_known.to_numpy().sum()),
        liquid_known_observations=int(common.to_numpy().sum()),
        missing_institutional_liquid=int((liquid & signal_quality & ~institutional_known).to_numpy().sum()))
    audit['historical_halt_quote_conflicts'] = int(halt_conflict.to_numpy().sum())
    coverage = pd.DataFrame({'date': days,
        'eligible': eligible.drop(columns='0050').sum(axis=1).to_numpy(),
        'usable_prices': usable.drop(columns='0050').sum(axis=1).to_numpy(),
        'actual_turnover': amount.drop(columns='0050').notna().sum(axis=1).to_numpy(),
        'institutional_known': institutional_known.drop(columns='0050').sum(axis=1).to_numpy(),
        'common_liquid': common.sum(axis=1).to_numpy()})
    return dict(days=days, ids=ids, close=raw['adjusted_close'], mark_close=mark_close,
        halt=verified_zero_halt, raw_close=raw['close'],
        open=adj['open'], high=adj['high'], low=adj['low'], volume=raw['volume'],
        raw_open=raw['open'], trade=valid_trade, conflict=conflict, ma=ma,
        ranks=ranks, trend=trend, common=common, market=market, breakout=breakout,
        pullback=pullback, near40=near40, high400=high400, vol9=vol9,
        volume_confirm=volume_confirm, trust=trust, foreign=foreign,
        audit=audit, coverage=coverage)


def signals(features, config):
    if config.weights not in range(3) or config.exit_ma not in (5, 10) or config.entries not in ('breakout', 'both'):
        raise ValueError('Unregistered course configuration')
    f = features
    momentum = sum(w * r for w, r in zip(WEIGHTS[config.weights], f['ranks']))
    trust, foreign = f['trust'], f['foreign']
    high = momentum.ge(config.threshold)
    flags = {
        'volume_momentum': high,
        'trust_near_breakout': f['near40'] & trust.gt(0).rolling(3, min_periods=3).sum().ge(1),
        'trust_consecutive': high & trust.gt(0) & trust.shift(1).gt(0),
        'foreign_first': high & foreign.gt(0) & foreign.shift(1).le(0),
        'foreign_trust': momentum.ge(config.threshold-10) & (foreign*f['raw_close']).ge(50_000_000) & (trust*f['raw_close']).ge(20_000_000),
        'price_volume_400': f['close'].gt(f['high400']) & f['volume'].ge(f['vol9']),
        'near_breakout': f['near40'],
    }
    membership = high if config.baseline else np.logical_or.reduce(list(flags.values()))
    if not isinstance(membership, pd.DataFrame):
        membership = pd.DataFrame(membership, index=f['days'], columns=f['ids'])
    entry = f['breakout'] if config.entries == 'breakout' else f['breakout'] | f['pullback']
    eligible = membership & f['trend'] & entry
    eligible = eligible.mul(f['market'], axis=0).astype(bool)
    ranking = momentum if config.baseline else .7*momentum + 10*trust.gt(0) + 10*foreign.gt(0) + 10*f['volume_confirm']
    ranking = ranking.where(eligible)
    family_counts = {k: int((v & eligible).to_numpy().sum()) for k, v in flags.items()}
    return dict(ranking=ranking, momentum=momentum, flags=flags, family_counts=family_counts)


def summarize(curve, trades, costs, turnover, counters, start, end):
    nav = curve.equity.to_numpy(float)
    returns = pd.Series(nav).pct_change(fill_method=None).dropna()
    years = (pd.Timestamp(end)-pd.Timestamp(start)).days/365.25
    total = nav[-1]/nav[0]-1
    completed = [t for t in trades if t['side'] == 'SELL']
    return dict(start=start, end=end, total_return=float(total),
        cagr=float((1+total)**(1/years)-1) if years > 0 else None,
        max_drawdown=float((nav/np.maximum.accumulate(nav)-1).min()),
        sharpe=float(returns.mean()/returns.std()*np.sqrt(252)) if returns.std()>0 else None,
        completed_trades=len(completed), win_rate=float(np.mean([t['net_pnl'] > 0 for t in completed])) if completed else None,
        initial_nav=float(nav[0]), final_net_nav=float(nav[-1]),
        commissions_and_taxes=float(costs), traded_notional=float(turnover),
        annual_one_way_turnover=float(turnover/2/np.mean(nav)/years) if years>0 else None,
        average_positions=float(curve.positions.iloc[1:].mean()), **counters)


def simulate(features, signal, config, start, end, *, slippage=.001, benchmark=False, initial_cash=1_000_000., slots=5):
    """Next-open execution, immediate daily-OHLC stops, no future-based allocation."""
    if not 0 <= slippage < .1 or initial_cash <= 0 or slots < 1:
        raise ValueError('Invalid research account settings')
    f = features
    days, ids = f['days'], f['ids']
    first, last = days.get_loc(pd.Timestamp(start)), days.get_loc(pd.Timestamp(end))
    if first < 1 or last <= first:
        raise ValueError('Need previous signal date and multiple evaluation sessions')
    arrays = {key: f[key].to_numpy() for key in ('open', 'low', 'close', 'mark_close', 'halt', 'trade', 'conflict', 'raw_open', 'volume')}
    rank = signal['ranking'].to_numpy() if signal is not None else None
    ma = f['ma'][config.exit_ma].to_numpy()
    cash, cost_total, notional_total = float(initial_cash), 0., 0.
    holdings, pending, pending_stop = {}, set(), set()
    trades, decisions = [], []
    curve = [dict(date=days[first-1], equity=cash, gross_equity=cash, cash=cash, positions=0)]
    counters = dict(blocked_orders=0, gap_rejections=0, stop_exits=0, verified_halt_hold_days=0,
        terminal_positions=0, terminal_liquidation_cost_estimate=0.)
    previous_nav = cash

    def sell(j, i, reference, reason):
        nonlocal cash, cost_total, notional_total
        pos = holdings.pop(j)
        price = reference*(1-slippage)
        gross = pos['units']*price
        fee = max(20., gross*.001425)
        tax = gross*(.001 if benchmark else .003)
        proceeds = gross-fee-tax
        cash += proceeds
        cost_total += fee+tax
        notional_total += gross
        trades.append(dict(date=str(days[i].date()), stock_id=str(ids[j]), side='SELL',
            signal_date=pos['signal_date'], entry_date=pos['entry_date'],
            adjusted_price=float(price), adjusted_units=pos['units'], notional=gross,
            fee=fee, tax=tax, net_pnl=proceeds-pos['entry_outlay'], reason=reason))
        pending.discard(j)
        pending_stop.discard(j)

    for i in range(first, last+1):
        stamp = str(days[i].date())
        for j in holdings:
            counters['verified_halt_hold_days'] += int(arrays['halt'][i,j])
            if not np.isfinite(arrays['mark_close'][i,j]) or arrays['mark_close'][i,j] <= 0 or arrays['conflict'][i,j]:
                raise ValueError(f'Held-position price/adjustment gap: {stamp} {ids[j]}')
        sold_open = set()
        for j in sorted(pending | pending_stop):
            if arrays['trade'][i,j] and np.isfinite(arrays['open'][i,j]):
                was_stop = j in pending_stop
                sell(j, i, arrays['open'][i,j], 'delayed_hard_stop' if was_stop else 'previous_close_below_ma')
                if was_stop:
                    counters['stop_exits'] += 1
                sold_open.add(j)
            else:
                counters['blocked_orders'] += 1
        if benchmark:
            candidates = [ids.get_loc('0050')] if i == first else []
        else:
            candidates = sorted(np.flatnonzero(np.isfinite(rank[i-1])), key=lambda j: (-rank[i-1,j], str(ids[j])))
        for j in candidates:
            if j in holdings or j in sold_open or len(holdings) >= (1 if benchmark else slots):
                continue
            opening, previous = arrays['open'][i,j], arrays['close'][i-1,j]
            if not arrays['trade'][i,j] or not np.isfinite(opening) or opening<=0 or arrays['conflict'][i,j]:
                counters['blocked_orders'] += 1
                if benchmark:
                    raise ValueError('Benchmark cannot execute at common start')
                continue
            gap = opening/previous-1
            if not benchmark and abs(gap) > .03:
                counters['gap_rejections'] += 1
                continue
            # Target allocation uses NAV known before this session's open.
            target = min(previous_nav/(1 if benchmark else slots), cash)
            gross = max(0., min(target/(1+.001425), target-20))
            if gross < 1000:
                continue
            fee = max(20., gross*.001425)
            price = opening*(1+slippage)
            units = gross/price
            cash -= gross+fee
            cost_total += fee
            notional_total += gross
            holdings[j] = dict(units=units, entry_outlay=gross+fee, stop=price*.93,
                entry_date=stamp, signal_date=str(days[i-1].date()))
            trades.append(dict(date=stamp, stock_id=str(ids[j]), side='BUY',
                signal_date=str(days[i-1].date()), adjusted_price=float(price),
                adjusted_units=float(units), notional=float(gross), fee=float(fee), tax=0.,
                net_pnl=None, reason='benchmark' if benchmark else 'course_entry'))
            decisions.append(dict(signal_date=str(days[i-1].date()), execution_date=stamp,
                stock_id=str(ids[j]), rank_score=None if benchmark else float(rank[i-1,j]),
                previous_nav=float(previous_nav), target_budget=float(target)))
        # Stop proceeds are unavailable to the already-completed entry stage.
        if not benchmark:
            for j in list(holdings):
                pos = holdings[j]
                if not arrays['halt'][i,j] and arrays['low'][i,j] <= pos['stop']:
                    if arrays['trade'][i,j]:
                        reference = min(arrays['open'][i,j], pos['stop'])
                        sell(j, i, reference, 'hard_stop_gap' if arrays['open'][i,j] <= pos['stop'] else 'hard_stop')
                        counters['stop_exits'] += 1
                    else:
                        pending_stop.add(j)
                        counters['blocked_orders'] += 1
            for j in holdings:
                if np.isfinite(ma[i,j]) and arrays['close'][i,j] < ma[i,j]:
                    pending.add(j)
        gross_nav = cash + sum(pos['units']*arrays['mark_close'][i,j] for j,pos in holdings.items())
        nav = gross_nav
        if i == last:
            terminal = sum(max(20., pos['units']*arrays['mark_close'][i,j]*(1-slippage)*.001425)
                + pos['units']*arrays['mark_close'][i,j]*(slippage+(1-slippage)*(.001 if benchmark else .003))
                for j,pos in holdings.items())
            counters['terminal_liquidation_cost_estimate'] = float(terminal)
            counters['terminal_positions'] = len(holdings)
            nav -= terminal
        if cash < -1e-7 or not np.isfinite(nav) or nav<=0:
            raise AssertionError('Invalid account value or cash borrowing')
        curve.append(dict(date=days[i], equity=float(nav), gross_equity=float(gross_nav),
            cash=float(cash), positions=len(holdings)))
        previous_nav = gross_nav
    curve = pd.DataFrame(curve)
    summary = summarize(curve, trades, cost_total, notional_total, counters, start, end)
    return dict(config=asdict(config), summary=summary, curve=curve, trades=trades,
        decisions=decisions, holdings=[dict(stock_id=str(ids[j]), **p) for j,p in holdings.items()])
