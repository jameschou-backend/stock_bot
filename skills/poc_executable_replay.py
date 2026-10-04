"""Preplanned limits with chronological print estimates and a separate odd auction.

These are observable-price execution models, not authenticated broker fills.
No daily high/low midpoint participates in matching, price selection or sizing.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
import math

import numpy as np
import pandas as pd

from skills.million_replay import money, costs
from skills.mixed_odd_replay import sized_quantity
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.poc_executable_odd import match_after_hours

OPEN, END = '09:01:00', '13:25:00'
ODD_OPEN, ODD_END = '13:40:00', '14:30:00'
PARTICIPATION = .01


def finite(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number)) and math.isfinite(value)


def validate_tape(tape):
    if not isinstance(tape, pd.DataFrame) or tape.empty or not {'time', 'price', 'shares'} <= set(tape):
        raise ReplayDataUnavailable('Missing normalized execution tape')
    if not pd.api.types.is_timedelta64_dtype(tape.time):
        raise ReplayDataUnavailable('Execution tape must preserve normalized exchange clocks')
    if tape.time.isna().any() or not tape.time.is_monotonic_increasing:
        raise ReplayDataUnavailable('Execution tape clock missing or out of order')
    if (tape.time.lt(pd.Timedelta(0)).any() or tape.time.ge(pd.Timedelta(days=1)).any()
            or pd.api.types.is_bool_dtype(tape.price) or pd.api.types.is_bool_dtype(tape.shares)):
        raise ReplayDataUnavailable('Invalid execution tape types')
    for p, q in zip(tape.price, tape.shares):
        if not finite(p) or p <= 0 or not finite(q) or q < 0 or q != int(q) or q > 2**53-1:
            raise ReplayDataUnavailable('Invalid execution tape price or share unit')


def match_board_prints(tape, side, limit_price, quantity, prior_adv, participation=PARTICIPATION):
    """Credit each lot at the actual print when enough prior capacity has accrued."""
    if side not in ('buy', 'sell') or type(quantity) is not int or quantity <= 0 or quantity % 1000:
        raise ValueError('Expected positive whole-lot order')
    if not all(finite(v) and v > 0 for v in (limit_price, prior_adv, participation)) or participation != PARTICIPATION:
        raise ValueError('Unregistered limit/ADV/participation')
    validate_tape(tape)
    rate = Decimal(str(participation))
    adv_cap = int((Decimal(str(prior_adv))*rate/1000).to_integral_value(rounding=ROUND_FLOOR))*1000
    cumulative, filled, gross, allocations = 0, 0, Decimal(0), []
    start, end = pd.Timedelta(OPEN), pd.Timedelta(END)
    for index, row in enumerate(tape.itertuples(index=False)):
        eligible = start < row.time < end and (row.price < limit_price if side == 'buy' else row.price > limit_price)
        if not eligible:
            continue
        cumulative += int(row.shares)
        cap = min(adv_cap, int((Decimal(cumulative)*rate/1000).to_integral_value(rounding=ROUND_FLOOR))*1000)
        qty = min(quantity-filled, max(0, cap-filled))
        if not qty:
            continue
        price = float(row.price)
        allocations.append(dict(source_index=index, time=str(row.time).split('days ')[-1],
            price=price, qty=qty, eligible_cumulative_shares=cumulative))
        filled += qty
        gross += Decimal(str(price))*qty
    capacity = min(adv_cap, int((Decimal(cumulative)*rate/1000).to_integral_value(rounding=ROUND_FLOOR))*1000)
    return dict(filled_qty=filled, capacity_qty=capacity, eligible_shares=cumulative,
        reference_price=float(gross/filled) if filled else None, allocations=allocations,
        last_fill_time=allocations[-1]['time'] if allocations else None,
        participation_limit=participation, prior_adv_capacity_qty=adv_cap,
        failure=None if filled == quantity else 'partial_trade_through_capacity' if filled else 'no_trade_through_capacity',
        execution_evidence='chronological_trade_through_print_estimate', actual_fill_verified=False)


class ExecutableOrders:
    def _plan(self, day, *args, **kwargs):
        super()._plan(day, *args, **kwargs)
        plan = self.tick_plans[-1]
        plan.update(order_time=OPEN, expires_at=END, odd_order_time=ODD_OPEN, odd_expires_at=ODD_END)
        if plan['side'] == 'sell' and plan['planned_qty']:
            bounds = self.feeds.get_limits(plan['stock_id']).get(plan['date'])
            if not bounds:
                raise ReplayDataUnavailable('Sell plan lacks dated official legal limits')
            plan['limit_price'] = plan['odd_limit'] = bounds['lower']
        self.day_plans[(plan['event_id'], plan['side'])] = deepcopy(plan)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if side == 'sell' and reason == 'scheduled_exit' and hasattr(self, 'exit_states'):
            state = self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:
                raise ValueError('Exit needs a previously latched signal')
            reason, signal_date = state['trigger_reason'], state['signal_date']
        plan = self.day_plans[(event_id, side)]
        stamp = str(day.date())
        if (plan['stock_id'] != sid or plan['signal_date'] != signal_date
                or not signal_date < stamp or sid in self.tick_attempts):
            raise ValueError('Order identity/timing changed or stock-day reused')
        self.tick_attempts.add(sid)
        if hasattr(self, 'identity'):
            item = self.identity(day, sid)
            if item['status'] not in ('identified', 'official_trading_suspension'):
                raise ReplayDataUnavailable('Execution requires identified dated security')
            if item['status'] == 'identified' and item['category'] != ('ETF' if sid == '0050' else '股票'):
                raise ReplayDataUnavailable('Execution security category differs')
            self.markets[sid] = item['market'].upper()
        if self.official_halt(day, sid):
            self.orders.append(dict(date=stamp, stock_id=sid, side=side, event_id=event_id,
                signal_date=signal_date, reason=reason, channel='event', requested_qty=plan['planned_qty'],
                filled_qty=0, failure='official_full_session_halt'))
            return 0
        if qty < plan['planned_qty']:
            raise ValueError('Post-plan quantity shrank')
        if plan['planned_qty']:
            self.require_prior_inputs(day, sid)
        filled_total, buy_spend = 0, 0.
        for channel, requested in (('board', plan['board_qty']), ('odd', plan['odd_qty'])):
            if not requested:
                continue
            limit = plan['limit_price'] if channel == 'board' else plan['odd_limit']
            bounds = self.feeds.get_limits(sid).get(stamp)
            if not bounds or not finite(limit) or not 0 < bounds['lower'] <= limit <= bounds['upper']:
                raise ReplayDataUnavailable('Order lacks a valid preknown legal limit')
            adv, amount = float(self.volume20.at[day, sid]), float(self.amount20.at[day, sid])
            row = dict(date=stamp, stock_id=sid, name=self.names.get(sid, sid), side=side,
                event_id=event_id, signal_date=signal_date, reason=reason, channel=channel,
                requested_qty=requested, limit_price=limit, prior_avg_volume20=adv,
                prior_avg_amount20=amount, order_time=OPEN if channel == 'board' else ODD_OPEN,
                expires_at=END if channel == 'board' else ODD_END)
            if self.used[(sid, channel)]:
                raise ValueError('Channel execution capacity already consumed')
            if channel == 'board':
                tape, digest = self.ticks.get(sid, stamp, self.markets[sid])
                quote = {k: self.raw(day, sid, k) for k in ('open', 'high', 'low', 'close', 'volume')}
                evidence = self.ticks.audit_day(sid, stamp, self.markets[sid], tape, quote)
                if tape.price.lt(bounds['lower']-1e-8).any() or tape.price.gt(bounds['upper']+1e-8).any():
                    raise ReplayDataUnavailable('Tape prices outside official legal range')
                row.update(ticks_sha256=digest, source_volume=int(tape.shares.sum()), tape_evidence=evidence,
                    **match_board_prints(tape, side, limit, requested, adv))
            else:
                odd = self.odd_feeds.get_odd(stamp, sid, self.markets[sid])
                if odd is None:
                    raise ReplayDataUnavailable('Missing after-hours auction evidence')
                row.update(match_after_hours(odd, side, limit, requested))
                price = row['auction_price']
                if price is not None and not bounds['lower']-1e-8 <= price <= bounds['upper']+1e-8:
                    raise ReplayDataUnavailable('Odd auction price outside official legal range')
            filled, price = row['filled_qty'], row['reference_price']
            if filled:
                paid = self._costs(price, filled, side, sid)
                if side == 'buy':
                    buy_spend = money(buy_spend-paid['cash_change'])
                    if buy_spend > plan['reserved_cash']+.005:
                        raise ValueError('Combined channel fills exceed reserved budget')
                if money(self.cash+paid['cash_change']) < 0:
                    raise ValueError('Execution would overdraw account')
                self.cash_move(day, side, paid['cash_change'], stock_id=sid, event_id=event_id, channel=channel)
                self.holdings[sid]['qty'] += filled if side == 'buy' else -filled
                if self.holdings[sid]['qty'] < 0:
                    raise ValueError('Execution oversold holdings')
                mark = self.raw(day, sid)
                self.marks[sid] = dict(price=mark, date=stamp)
                self.used[(sid, channel)] += filled
                self.day_cost += paid['total_cost']
                self.day_basis += filled*(mark-price)*(1 if side == 'buy' else -1)
                trade = dict(row, **paid, qty=filled, cash_after=self.cash,
                    remaining_shares=self.holdings[sid]['qty'],
                    day_participation=filled/row['source_volume'], sequence=len(self.trades)+1)
                trade.pop('filled_qty'); trade.pop('failure', None)
                self.trades.append(trade)
            self.orders.append(row)
            filled_total += filled
        return filled_total

    def run(self):
        result = super().run()
        settings = result['settings']
        for name in ('price_formula', 'historical_odd_regime'):
            settings.pop(name, None)
        settings.update(execution='preplanned_limits_chronological_prints_afterhours_odd_v1',
            buy_limit_policy='official_daily_upper_limit', sell_limit_policy='official_daily_lower_limit',
            sell_price_policy='chronological_print_estimate_or_single_afterhours_auction',
            board_capacity='post_order_strict_trade_through_1pct_capped_by_prior_total_adv20',
            volume_policy='observed_provider_prints_not_verified_complete_order_book',
            odd_execution_evidence='after_hours_single_auction_price_and_volume',
            missing_ordinary_policy='stop_on_missing_or_known_conflict', opening_auction_inferred=False,
            actual_fill_verified=False, odd_tick_verified=False, live_qualified=False, unseen_validation=False)
        board = [r for r in result['orders'] if r['channel'] == 'board']
        result['ordinary_volume_evidence'] = dict(requested_board_children=len(board),
            all_requested_board_capacity_observed=bool(board) and all('ticks_sha256' in r for r in board),
            complete_exchange_tape_verified=False, actual_fill_verified=False,
            policy='observed_provider_prints', source_hashes_must_be_bound_by_caller=True)
        return result


def _near(a, b, label, tolerance=.011):
    if not finite(a) or not finite(b) or not math.isclose(a, b, abs_tol=tolerance, rel_tol=0):
        raise ValueError('Execution audit differs: '+label)


def audit_cash_phases(account):
    """Reconcile session order, without treating symbol-loop order as a clock.

    Within a channel all purchases debit before sales: this stronger check does
    not borrow the proceeds of a sale whose individual print time is later.
    """
    for index, day in enumerate(account['daily']):
        cash = account['daily'][index-1]['cash'] if index else account['settings']['initial_cash']
        cash = money(cash+sum(r['cash_change'] for r in account['cash_ledger']
            if r['date'] == day['date'] and r['kind'] not in ('initial_deposit', 'buy', 'sell')))
        rows = [t for t in account['trades'] if t['date'] == day['date']]
        for trade in sorted(rows, key=lambda t:(t['channel'] == 'odd', t['side'] == 'sell')):
            cash = money(cash+trade['cash_change'])
            if cash < 0:
                raise ValueError('Later sale proceeds financed an earlier session purchase')
        _near(cash, day['cash'], 'independent session cash')
    return dict(session_cash_verified=True, ledger_sequence_is_execution_clock=False)


def audit_executable(account, ticks, odds, routes, quotes, days, corp, feeds, *, verified_halts=()):
    """Independently reconstruct committed limits, chronological allocation and cash."""
    calendar = pd.DatetimeIndex(days)
    prior = {str(calendar[i].date()): str(calendar[i-1].date()) for i in range(1, len(calendar))}
    source = quotes.copy(); source['date'] = pd.to_datetime(source.date)
    if source.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate audit quote')
    closes = source.pivot(index='date', columns='stock_id', values='close').reindex(calendar)
    volumes = source.pivot(index='date', columns='stock_id', values='volume').reindex(calendar)
    adv = volumes.rolling(20, min_periods=20).mean().shift(1)
    amounts = (closes*volumes).rolling(20, min_periods=20).mean().shift(1)
    opening_cash = {r['date']: account['daily'][i-1]['cash'] if i else account['settings']['initial_cash']
                    for i, r in enumerate(account['daily'])}
    plans, budgets, spent = {}, {}, {}
    for p in account['tick_plans']:
        key = p['date'], p['event_id'], p['side']
        if key in plans or prior.get(p['date']) != p['reference_date'] or p['signal_date'] > p['reference_date']:
            raise ValueError('Noncausal or duplicate execution plan')
        if p['side'] == 'buy' and p['signal_date'] != p['reference_date']:
            raise ValueError('Buy delayed past next session')
        if (p['order_time'], p['expires_at'], p['odd_order_time'], p['odd_expires_at']) != (OPEN, END, ODD_OPEN, ODD_END):
            raise ValueError('Execution plan window differs')
        if (any(type(p[k]) is not int or p[k] < 0 for k in ('planned_qty', 'board_qty', 'odd_qty'))
                or p['board_qty'] % 1000 or p['odd_qty'] >= 1000
                or p['planned_qty'] != p['board_qty']+p['odd_qty']):
            raise ValueError('Execution plan quantity differs')
        sid, day = p['stock_id'], pd.Timestamp(p['date'])
        raw = float(closes.at[pd.Timestamp(p['reference_date']), sid])
        if math.isfinite(raw) and raw > 0:
            reference = corp.reference_price(sid, p['date'], raw)
            _near(p['prior_reference'], reference, 'prior reference', 1e-8)
        elif p['planned_qty']:
            raise ValueError('Order sized without previous-session reference')
        if p['planned_qty']:
            limits = feeds.get_limits(sid).get(p['date'])
            if not limits:
                raise ValueError('Plan has no legal limits')
            bound = limits['upper' if p['side'] == 'buy' else 'lower']
            if p['limit_price'] != bound or p['odd_limit'] != bound:
                raise ValueError('Plan was not fixed at legal marketable limit')
            if p['side'] == 'buy':
                maximum = max(0, math.floor((p['sizing_budget']-40)/(p['prior_reference']*(1+.001425+.0045))))
                class Cost:
                    _costs = staticmethod(costs)
                if p['planned_qty'] != sized_quantity(maximum, bound, p['sizing_budget'], Cost(), sid):
                    raise ValueError('Plan size differs from opening budget')
        plans[key] = p
        budgets[p['date']] = budgets.get(p['date'], 0.)+p['reserved_cash']
        if budgets[p['date']] > opening_cash[p['date']]+.011:
            raise ValueError('Plans use unconfirmed same-day proceeds')
    required = {(key, channel) for key, p in plans.items()
                for channel in ('board', 'odd') if p[channel+'_qty'] > 0}
    verified, seen, halted = {}, set(), set()
    for row in account['orders']:
        if row['channel'] == 'event':
            key = row['date'], row['event_id'], row['side']
            p = plans.get(key)
            if not p or row['filled_qty'] or row['stock_id'] != p['stock_id'] or row['signal_date'] != p['signal_date']:
                raise ValueError('Unexpected non-tradable event fill')
            if not p['planned_qty']:
                if not row.get('failure'):
                    raise ValueError('Rejected zero plan lacks a reason')
                continue
            matching = []
            for h in verified_halts:
                s = h.get('source_row', [])
                full = (h.get('kind') == 'trading_suspension' or
                    h.get('kind') == 'information_halt' and len(s) == 7 and s[4] in ('8:00','08:00') and s[6] in ('8:00','08:00'))
                market = routes.get((p['stock_id'], row['date']), routes.get(p['stock_id']))
                if (full and h['stock_id'] == p['stock_id'] and h.get('start') and h.get('end')
                    and h['start'] <= row['date'] < h['end'] and h['market'].upper() == market.upper()):
                    matching.append(h)
            if (key in halted or row['failure'] != 'official_full_session_halt'
                or row['requested_qty'] != p['planned_qty'] or not matching):
                raise ValueError('Nonzero plan lacks independently verified halt evidence')
            halted.add(key)
            continue
        day, sid = row['date'], row['stock_id']
        key = day, row['event_id'], row['side']
        identity = day, sid, row['channel']
        if identity in seen:
            raise ValueError('Execution tape reused')
        seen.add(identity); p = plans[key]
        channel = row['channel']
        if channel not in ('board', 'odd') or p['stock_id'] != sid or p['signal_date'] != row['signal_date']:
            raise ValueError('Execution differs from plan identity')
        limit = p['limit_price' if channel == 'board' else 'odd_limit']
        if row['limit_price'] != limit or row['requested_qty'] != p[channel+'_qty']:
            raise ValueError('Execution differs from planned price/quantity')
        if (row['order_time'], row['expires_at']) != ((OPEN, END) if channel == 'board' else (ODD_OPEN, ODD_END)):
            raise ValueError('Execution timing differs')
        _near(row['prior_avg_volume20'], float(adv.at[pd.Timestamp(day), sid]), 'prior ADV', 1e-7)
        _near(row['prior_avg_amount20'], float(amounts.at[pd.Timestamp(day), sid]), 'prior turnover', 1e-5)
        market = routes.get((sid, day), routes.get(sid))
        if channel == 'board':
            tape, digest = ticks.get(sid, day, market)
            if digest != row['ticks_sha256']:
                raise ValueError('Execution tick hash differs')
            # Separate loop from matcher: one integer credit bucket per print.
            cumulative, shares, total, expected = 0, 0, Decimal(0), []
            cap_adv = int(Decimal(str(row['prior_avg_volume20']))/100000)*1000
            for j, t in enumerate(tape.itertuples(index=False)):
                if not pd.Timedelta(OPEN) < t.time < pd.Timedelta(END):
                    continue
                if not (t.price < limit if row['side'] == 'buy' else t.price > limit):
                    continue
                cumulative += int(t.shares)
                capacity = min(cumulative//100000*1000, cap_adv)
                addition = min(p['board_qty']-shares, max(0, capacity-shares))
                if addition:
                    expected.append(dict(source_index=j, time=str(t.time).split('days ')[-1], price=float(t.price),
                        qty=addition, eligible_cumulative_shares=cumulative))
                    shares += addition; total += Decimal(str(t.price))*addition
            if row['allocations'] != expected or row['eligible_shares'] != cumulative:
                raise ValueError('Chronological allocations differ')
            capacity = min(cumulative//100000*1000, cap_adv)
            price = float(total/shares) if shares else None
            last = expected[-1]['time'] if expected else None
        else:
            raw = odds.get_odd(day, sid, market)
            if not raw or not raw.get('after_hours') or raw['odd_high'] != raw['odd_low']:
                raise ValueError('Odd execution lacks single auction evidence')
            price = raw['odd_high']; volume = raw['odd_shares']
            crosses = volume > 0 and (price < limit if row['side'] == 'buy' else price > limit)
            capacity = int(volume)//100 if crosses else 0
            shares = min(p['odd_qty'], capacity)
            price = price if shares else None
            last = ODD_END if shares else None
        if row['filled_qty'] != shares or row['capacity_qty'] != capacity or row['last_fill_time'] != last:
            raise ValueError('Execution share/clock/capacity audit differs')
        if shares:
            _near(row['reference_price'], price, 'fill price', 1e-8)
        elif row['reference_price'] is not None:
            raise ValueError('Unfilled order has a fabricated fill price')
        verified[(key, channel)] = row
    halted_children = {item for item in required if item[0] in halted}
    if set(verified) & halted_children or set(verified) | halted_children != required:
        raise ValueError('Nonzero planned child order missing, unexpected, or both halted and executed')
    fills = {}
    for trade in account['trades']:
        key = trade['date'], trade['event_id'], trade['side']
        row = verified[(key, trade['channel'])]
        if (key, trade['channel']) in fills or trade['qty'] != row['filled_qty']:
            raise ValueError('Trade quantity differs from order evidence')
        fills[(key, trade['channel'])] = trade['qty']
        _near(trade['reference_price'], row['reference_price'], 'trade reference', 1e-8)
        expected = costs(row['reference_price'], trade['qty'], trade['side'], trade['stock_id'])
        for name, value in expected.items():
            _near(trade[name], value, 'trade '+name)
        if trade['side'] == 'buy':
            spent[key] = money(spent.get(key, 0.)-trade['cash_change'])
            if spent[key] > plans[key]['reserved_cash']+.011:
                raise ValueError('Combined channel spend exceeds planned budget')
    if any(fills.get(k, 0) != row['filled_qty'] for k, row in verified.items()):
        raise ValueError('Filled order omitted from trade ledger')
    return dict(**audit_cash_phases(account), preplanned_limits_verified=True, opening_cash_only=True,
        all_planned_children_reconciled=len(required), verified_halt_children=len(halted_children),
        chronological_board_allocations_rebuilt=sum(r['channel'] == 'board' for r in account['orders']),
        afterhours_auctions_rebuilt=sum(r['channel'] == 'odd' for r in account['orders']),
        fills_reconciled=len(fills), actual_fill_verified=False, live_qualified=False)
