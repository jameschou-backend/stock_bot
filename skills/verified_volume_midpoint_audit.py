"""Independently rebuild HL2 orders with session-specific capacity evidence.

The ordinary matrix is separate from total daily shares used to size positions.
Missing evidence means an unfilled diagnostic order, never a zero-volume fact.
"""
from copy import deepcopy
import math

import pandas as pd

from skills.execution_stress import StressOrder
from skills.midpoint_exit_audit import legacy_resource_plans
from skills.mixed_odd_audit import audit_mixed_execution


def _ordinary_details(matrices, resolver, calendar, day, sid):
    """Independent scalar rebuild; do not call the execution engine's helper."""
    position = calendar.get_loc(day)
    window = calendar[max(0, position-20):position+1]
    observations, gaps, markets = [], [], []
    for stamp in window:
        market = resolver(stamp, sid)
        if isinstance(market, dict):
            market = market.get('market') if market.get('status') in ('identified', 'official_trading_suspension') else None
        market = market.upper() if isinstance(market, str) and market.upper() in ('TWSE', 'TPEX') else None
        markets.append(market)
        matrix = matrices.get(market)
        value = (matrix.at[stamp, sid] if matrix is not None
                 and stamp in matrix.index and sid in matrix.columns else None)
        if value is None or pd.isna(value):
            observations.append(None)
            gaps.append(dict(date=str(stamp.date()), market=market, stock_id=sid,
                             reason='dated_market_unknown' if market is None else 'ordinary_volume_missing'))
        else:
            value = float(value)
            if not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError('Invalid ordinary-session audit evidence')
            observations.append(value)
    current = observations[-1]
    prior = observations[:-1]
    prior_complete = len(prior) == 20 and all(v is not None for v in prior)
    if len(prior) != 20:
        gaps.append(dict(date=str(day.date()), market=markets[-1], stock_id=sid,
                         reason='ordinary_history_less_than_20_market_days'))
    complete = prior_complete and current is not None
    return current, sum(prior)/20 if prior_complete else None, complete, gaps


def ordinary_inputs(matrices, resolver, calendar, day, sid):
    return _ordinary_details(matrices, resolver, calendar, day, sid)[:3]


def audit_verified_volume_midpoint(account, ticks, odds, markets, quotes, calendar,
                                   corporate, feeds, ordinary_volumes, resolver, *, verified_halts=()):
    settings = account['settings']
    if (settings.get('price_formula') != '(high+low)/2'
            or settings.get('actual_fill_verified') is not False
            or settings.get('opening_auction_inferred') is not False
            or settings.get('live_qualified') is not False
            or settings.get('volume_policy') != 'strict'
            or settings.get('odd_execution_evidence') != 'daily_high_low_midpoint_proxy'
            or settings['participation'] != .01 or settings['slippage'] != .0045):
        raise ValueError('Session-specific midpoint assumptions changed')
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('Canonical market calendar required')
    plans = (legacy_resource_plans(account, feeds)
             if settings.get('sell_limit_policy') == 'official_daily_lower_limit'
             else deepcopy(account['tick_plans']))
    for plan in plans:
        if plan['expires_at'] != '13:30:00':
            raise ValueError('Midpoint order window changed')
        plan['expires_at'] = '13:25:00'
    skeleton = dict(account, tick_plans=plans, orders=[], trades=[], settings=dict(
        settings, odd_execution_evidence='daily_envelope_estimate'))
    audit_mixed_execution(skeleton, ticks, odds, markets, quotes, calendar, corporate, feeds)
    indexed = quotes.set_index(['date', 'stock_id'])
    volumes = quotes.pivot(index='date', columns='stock_id', values='volume').reindex(calendar)
    closes = quotes.pivot(index='date', columns='stock_id', values='close').reindex(calendar)
    adv = volumes.rolling(20, min_periods=20).mean().shift(1)
    amount = (volumes*closes).rolling(20, min_periods=20).mean().shift(1)
    frozen = {(p['date'], p['event_id'], p['side']): p for p in account['tick_plans']}
    if len(frozen) != len(account['tick_plans']):
        raise ValueError('Duplicate frozen midpoint plan')
    # Nonzero child plans are the denominator, including failed orders. A
    # deleted zero-fill attempt must not turn incomplete evidence into success.
    required = {(*key, channel) for key, plan in frozen.items()
                for channel in ('board', 'odd') if plan[channel+'_qty'] > 0}
    halted = set()
    for row in account['orders']:
        if row['channel'] != 'event':
            continue
        key = (row['date'], row['event_id'], row['side'])
        plan = frozen.get(key)
        if not plan or not plan['planned_qty']:
            continue  # Resource rejections of zero-sized plans create no child.
        matching_halts = []
        for halt in verified_halts:
            source = halt.get('source_row', [])
            full_day = (halt.get('kind') == 'trading_suspension'
                        or (halt.get('kind') == 'information_halt' and len(source) == 7
                            and source[4] in ('8:00', '08:00') and source[6] in ('8:00', '08:00')))
            if (full_day and halt['stock_id'] == plan['stock_id'] and halt.get('start') and halt.get('end')
                    and halt['start'] <= row['date'] < halt['end']):
                matching_halts.append(halt)
        if (key in halted or row.get('failure') != 'official_full_session_halt'
                or row.get('filled_qty') != 0 or row.get('requested_qty') != plan['planned_qty']
                or row.get('stock_id') != plan['stock_id'] or row.get('signal_date') != plan['signal_date']
                or not matching_halts):
            raise ValueError('Nonzero frozen child lacks an independently evidenced full-session halt')
        day, sid = pd.Timestamp(row['date']), row['stock_id']
        identity = resolver(day, sid)
        market = identity.get('market') if isinstance(identity, dict) else identity
        if not market or not any(h['market'].upper() == market.upper() for h in matching_halts):
            raise ValueError('Full-session halt dated market differs')
        if (day, sid) in indexed.index:
            source = indexed.loc[(day, sid)]
            if any(pd.notna(source[k]) and float(source[k]) != 0 for k in ('open', 'high', 'low', 'close', 'volume')):
                raise ValueError('Full-session halt conflicts with observed activity')
        matrix = ordinary_volumes.get(market.upper())
        if (matrix is not None and day in matrix.index and sid in matrix.columns
                and pd.notna(matrix.at[day, sid]) and matrix.at[day, sid] != 0):
            raise ValueError('Full-session halt conflicts with ordinary activity')
        halted.add(key)
    required = {child for child in required if child[:3] not in halted}
    orders, fills, stockdays, cash_by_day = {}, {}, set(), {}
    for i, daily in enumerate(account['daily']):
        cash = account['daily'][i-1]['cash'] if i else settings['initial_cash']
        cash += sum(r['cash_change'] for r in account['cash_ledger']
                    if r['date'] == daily['date'] and r['kind'] not in ('initial_deposit', 'buy', 'sell'))
        cash_by_day[daily['date']] = cash
    cost = StressOrder()
    cost.stress_slippage = .0045
    blocked, expected_blocks = 0, []
    for row in account['orders']:
        if row['channel'] not in ('board', 'odd') or not row['requested_qty']:
            continue
        key = (row['date'], row['event_id'], row['side'])
        child = (*key, row['channel'])
        plan = frozen[key]
        sid, day, channel = row['stock_id'], pd.Timestamp(row['date']), row['channel']
        stockday = (row['date'], sid, channel)
        if child in orders or stockday in stockdays:
            raise ValueError('Ordinary/odd capacity reused')
        stockdays.add(stockday)
        limit = plan['limit_price'] if channel == 'board' else plan['odd_limit']
        if (row['requested_qty'] != plan[channel+'_qty'] or row['limit_price'] != limit
                or row['signal_date'] != plan['signal_date'] or row['signal_date'] >= row['date']
                or sid != plan['stock_id'] or row['expires_at'] != '13:30:00'
                or row['execution_evidence'] != 'daily_high_low_midpoint_proxy'
                or row['prior_avg_volume20'] != adv.at[day, sid]
                or row['prior_avg_amount20'] != amount.at[day, sid]):
            raise ValueError('Order differs from prior-only position sizing')
        limits = feeds.get_limits(sid)[row['date']]
        complete = True
        if channel == 'board':
            source = indexed.loc[(day, sid)]
            high, low = (float(source[k]) if pd.notna(source[k]) else None for k in ('high', 'low'))
            volume, capacity_adv, complete, gaps = _ordinary_details(ordinary_volumes, resolver, calendar, day, sid)
            if (row.get('volume_scope') != 'ordinary_session'
                    or row.get('capacity_prior_ordinary_volume20') != capacity_adv
                    or row.get('source_total_volume') != (float(source['volume']) if pd.notna(source['volume']) else None)
                    or row.get('ordinary_capacity_verified') is not complete
                    or row.get('volume_policy') != 'strict'):
                raise ValueError('Ordinary capacity input or scope differs')
        else:
            source = odds.get_odd(row['date'], sid, markets.get((sid, row['date']), markets.get(sid)))
            high, low, volume = (source[k] for k in ('odd_high', 'odd_low', 'odd_shares'))
            capacity_adv = adv.at[day, sid]
            if (row.get('volume_policy') != 'strict' or row.get('volume_scope') != 'independent_odd_session'
                    or row.get('source_total_volume') is not None
                    or row.get('capacity_prior_ordinary_volume20') is not None
                    or row.get('ordinary_capacity_verified') is not False):
                raise ValueError('Odd session evidence mislabeled as ordinary volume')
        if (row['source_high'], row['source_low'], row['source_volume']) != (high, low, volume):
            raise ValueError('Session price/volume evidence differs')
        if not complete:
            blocked += 1
            if (row['filled_qty'] != 0 or row['capacity_qty'] != 0
                    or row['reference_price'] is not None
                    or row.get('failure') != 'ordinary_volume_evidence_missing'
                    or row.get('ordinary_volume_gaps') != gaps):
                raise ValueError('Missing ordinary evidence generated a simulated fill')
            expected_blocks.append(dict(date=row['date'], stock_id=sid, side=row['side'], event_id=row['event_id'],
                requested_qty=row['requested_qty'], missing=gaps, reason='ordinary_volume_evidence_missing'))
            orders[child] = row
            continue
        if channel == 'board' and capacity_adv == 0:
            if (row['filled_qty'] != 0 or row['capacity_qty'] != 0
                    or row['reference_price'] is not None
                    or row.get('failure') != 'official_ordinary_history_zero'):
                raise ValueError('Zero observed ordinary history generated capacity')
            orders[child] = row
            continue
        price = (high+low)/2 if volume else None
        if volume and not limits['lower']-1e-8 <= low <= high <= limits['upper']+1e-8:
            raise ValueError('HL2 source outside legal price range')
        eligible = volume > 0 and (price < min(limit, limits['upper'])-1e-8 if row['side'] == 'buy'
                                  else price > max(limit, limits['lower'])+1e-8)
        capacity = math.floor((min(volume, capacity_adv) if channel == 'board' else volume)*.01) if eligible else 0
        if channel == 'board':
            capacity = capacity//1000*1000
        fill = min(row['requested_qty'], capacity)
        if fill:
            change = cost._costs(price, fill, row['side'], sid)['cash_change']
            if round(cash_by_day[row['date']]+change, 2) < 0:
                if row.get('failure') != 'proceeds_below_costs_insufficient_cash':
                    raise ValueError('Midpoint order overspent cash')
                fill = 0
            else:
                cash_by_day[row['date']] += change
        if row['reference_price'] != price or row['capacity_qty'] != capacity or row['filled_qty'] != fill:
            raise ValueError('Session-specific midpoint arithmetic differs')
        orders[child] = row
    if set(orders) != required:
        raise ValueError('Nonzero frozen child plans and execution orders differ')
    for sequence, trade in enumerate(account['trades'], start=1):
        key = (trade['date'], trade['event_id'], trade['side'], trade['channel'])
        row = orders.get(key)
        if not row or trade['reference_price'] != row['reference_price']:
            raise ValueError('Midpoint trade lacks a source-priced order')
        if any(k not in trade or trade[k] != value for k,value in row.items() if k not in ('filled_qty','failure')):
            raise ValueError('Trade copied different identity, price or capacity evidence from its order')
        if (type(trade['qty']) is not int or trade['qty'] <= 0 or trade.get('sequence') != sequence
                or (trade['channel'] == 'board' and trade['qty'] % 1000)):
            raise ValueError('Trade quantity or sequence is invalid')
        expected = cost._costs(row['reference_price'], trade['qty'], trade['side'], trade['stock_id'])
        if any(trade[k] != value for k, value in expected.items()):
            raise ValueError('Per-channel transaction costs differ')
        fills[key] = fills.get(key, 0)+trade['qty']
    if any(fills.get(key, 0) != row['filled_qty'] for key, row in orders.items()):
        raise ValueError('Audited orders and trade quantities differ')
    if any(abs(cash_by_day[row['date']]-row['cash']) > .02 for row in account['daily']):
        raise ValueError('Independent daily cash reconciliation failed')
    board_count = sum(child[3] == 'board' for child in required)
    evidence = account['ordinary_volume_evidence']
    missing = {(g['market'],g['date'],b['stock_id']) for b in expected_blocks for g in b['missing']}
    missing_rows = [dict(market=m,date=d,stock_id=s) for m,d,s in sorted(missing,key=str)]
    if (evidence['blocked_board_children'] != blocked
            or evidence['requested_board_children'] != board_count
            or evidence.get('verified_board_children') != board_count-blocked
            or evidence.get('blocks') != expected_blocks
            or evidence.get('missing_stock_days') != missing_rows
            or evidence.get('policy') != 'strict'
            or evidence.get('actual_fill_verified') is not False
            or evidence.get('live_qualified') is not False
            or evidence['all_requested_board_capacity_observed'] is not (bool(board_count) and blocked == 0)):
        raise ValueError('Ordinary-volume coverage claim differs from rebuilt orders')
    last_day = account['daily'][-1]['date'] if account['daily'] else None
    final_holdings = [r for r in account['holdings'] if r['date'] == last_day and r['qty'] > 0]
    if len({r['stock_id'] for r in final_holdings}) != len(final_holdings):
        raise ValueError('Duplicate final holding')
    expected_inventory = dict(valuation='mark_to_market', positions=final_holdings, automatically_liquidated=False,
                              remaining_positions=len(final_holdings), stale_positions=sum(bool(r.get('stale')) for r in final_holdings))
    if (account.get('ending_inventory') != expected_inventory
            or settings.get('ending_inventory_valuation') != 'mark_to_market'
            or settings.get('forced_end_liquidation') is not False):
        raise ValueError('Ending inventory must match marked holdings without invented liquidation')
    return dict(prior_only_plans_rebuilt=True, midpoint_prices_rebuilt=True,
                daily_capacity_rebuilt=True, separate_channel_costs_rebuilt=True,
                nonzero_planned_children_reconciled=True, full_session_halt_plans=len(halted),
                ordinary_evidence_blocked_orders=blocked,
                all_attempted_ordinary_capacity_verified=bool(board_count) and blocked == 0,
                diagnostic_skips_present=blocked > 0, price_proxy_only=True,
                actual_fill_verified=False, live_qualified=False)
