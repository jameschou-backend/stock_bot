"""Daily range-price research for ordinary and independent intraday odd shares.

Daily ranges and totals are hindsight execution proxies. They never enter the
precommitted quantity, limit, signal or cash budget and prove no fill timestamp.
The original source quality gates and whole-order gap rollback remain intact.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR

import pandas as pd

from skills.million_replay import money
from skills.poc_executable_replay import PARTICIPATION, ExecutableOrders, finite, validate_tape
from skills.poc_gap_execution import DataGapOrders
from skills.poc_intraday_execution import validate_intraday_odd
from skills.replay_market_feeds import ReplayDataUnavailable

MODEL = 'ordinary_and_intraday_odd_daily_range_fraction_proxy_v1'
SCOPE = 'intraday_odd_session'
BOARD_EVIDENCE = 'ordinary_daily_range_fraction_proxy'
ODD_EVIDENCE = 'intraday_odd_daily_range_fraction_proxy'
OPEN, END = '09:00:00', '13:30:00'
ODD_OPEN, ODD_END = OPEN, END


def validate_fractions(buy_fraction, sell_fraction):
    if (not finite(buy_fraction) or not finite(sell_fraction)
            or (buy_fraction, sell_fraction) not in ((.5, .5), (.7, .3))):
        raise ValueError('Only registered midpoint or buy70/sell30 fractions are allowed')


def _validate_order(side, limit_price, quantity, fraction, participation, *, board):
    if (side not in ('buy', 'sell') or type(quantity) is not int or quantity <= 0
            or (quantity % 1000 != 0 if board else quantity >= 1000)
            or not finite(limit_price) or limit_price <= 0
            or not finite(participation) or participation != PARTICIPATION
            or not finite(fraction) or fraction not in ((.5, .7) if side == 'buy' else (.5, .3))):
        raise ValueError('Invalid preplanned range order, fraction or participation')


def _range_result(volume, high, low, side, limit_price, quantity, fraction,
                  capacity, *, board, used_shares=0):
    price = (Decimal(str(low))+Decimal(str(fraction))*(Decimal(str(high))-Decimal(str(low)))) if volume else None
    through = price is not None and (price < Decimal(str(limit_price)) if side == 'buy'
                                    else price > Decimal(str(limit_price)))
    available = max(0, capacity-used_shares) if through else 0
    filled = min(quantity, available)
    failure = (('official_zero_intraday_odd_volume' if not board else 'zero_regular_session_volume') if not volume
               else 'range_limit_not_crossed' if not through
               else 'range_capacity_zero' if not filled
               else 'partial_range_capacity' if filled < quantity else None)
    return dict(filled_qty=filled, capacity_qty=available, daily_capacity_qty=capacity,
        source_volume=volume, source_high=high, source_low=low, price_fraction=fraction,
        proxy_price=float(price) if price is not None else None,
        reference_price=float(price) if filled else None, last_fill_time=None,
        actual_fill_time=None, participation_limit=PARTICIPATION, used_shares=used_shares,
        failure=failure, volume_scope='ordinary_session' if board else SCOPE,
        evidence_status='parent_quality_checked_regular_tape' if board else 'official_intraday_daily_table',
        execution_evidence=BOARD_EVIDENCE if board else ODD_EVIDENCE,
        intraday_tick_verified=False, odd_tick_verified=False, board_tick_verified=False,
        within_window_execution_verified=False, price_level_volume_verified=False,
        source_hash_verification_required_by_caller=True,
        actual_fill_verified=False, live_qualified=False)


def match_range_board(tape, side, limit_price, quantity, prior_adv, *, fraction=.5,
                      participation=PARTICIPATION):
    _validate_order(side, limit_price, quantity, fraction, participation, board=True)
    if not finite(prior_adv) or prior_adv <= 0:
        raise ValueError('Invalid prior ADV for range capacity')
    validate_tape(tape)
    regular = tape.loc[tape.time.ge(pd.Timedelta('09:00:00'))
        & tape.time.lt(pd.Timedelta('13:34:00')) & tape.shares.gt(0)]
    volume = sum(int(n) for n in regular.shares)
    high, low = (float(regular.price.max()), float(regular.price.min())) if volume else (None, None)
    day_cap = volume//100000*1000
    adv_cap = int((Decimal(str(prior_adv))*Decimal('.01')/1000).to_integral_value(rounding=ROUND_FLOOR))*1000
    result = _range_result(volume, high, low, side, limit_price, quantity, fraction,
                          min(day_cap, adv_cap), board=True)
    result.update(regular_daily_capacity_qty=day_cap, prior_adv_capacity_qty=adv_cap,
                  allocations=[], source_window='09:00:00<=time<13:34:00_including_delayed_close')
    return result


def match_range_odd(row, side, limit_price, quantity, *, fraction=.5, used_shares=0,
                    participation=PARTICIPATION):
    _validate_order(side, limit_price, quantity, fraction, participation, board=False)
    if type(used_shares) is not int or used_shares < 0:
        raise ValueError('Invalid previously used intraday odd shares')
    volume, high, low = validate_intraday_odd(row)
    capacity = volume//100
    if used_shares > capacity:
        raise ValueError('Previously used intraday odd shares exceed daily proxy capacity')
    return _range_result(volume, high, low, side, limit_price, quantity, fraction,
                         capacity, board=False, used_shares=used_shares)


class RangeOrders(ExecutableOrders):
    def __init__(self, *args, buy_fraction=.5, sell_fraction=.5, **kwargs):
        validate_fractions(buy_fraction, sell_fraction)
        self.buy_fraction, self.sell_fraction = buy_fraction, sell_fraction
        super().__init__(*args, **kwargs)

    def _plan(self, day, *args, **kwargs):
        super()._plan(day, *args, **kwargs)
        plan = self.tick_plans[-1]
        plan.update(order_time=OPEN, expires_at=END, odd_order_time=ODD_OPEN, odd_expires_at=ODD_END)
        self.day_plans[(plan['event_id'], plan['side'])] = deepcopy(plan)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        # Preserve the sealed financial body; change only the price/capacity model.
        # Both channels are daily proxies with unknown actual execution clocks.
        self.execution_gap_stage = 'execution_context'
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
            self.execution_gap_stage = channel
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
                row.update(ticks_sha256=digest, tape_evidence=evidence,
                    **match_range_board(tape, side, limit, requested, adv,
                        fraction=self.buy_fraction if side == 'buy' else self.sell_fraction))
            else:
                odd = self.odd_feeds.get_odd(stamp, sid, self.markets[sid])
                validate_intraday_odd(odd)
                if (odd.get('source_date') != stamp
                        or not isinstance(odd.get('market'), str)
                        or odd['market'].upper() != self.markets[sid].upper()):
                    raise ReplayDataUnavailable('Intraday odd source date or market differs from the order')
                row.update(match_range_odd(odd, side, limit, requested,
                    fraction=self.buy_fraction if side == 'buy' else self.sell_fraction))
                if row['source_volume'] and not (bounds['lower']-1e-8 <= row['source_low']
                                                <= row['source_high'] <= bounds['upper']+1e-8):
                    raise ReplayDataUnavailable('Intraday odd daily range outside official legal limits')
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
        account = super().run()
        account['settings'].update(execution=MODEL,
            buy_fraction=self.buy_fraction, sell_fraction=self.sell_fraction,
            price_formula='low+fraction*(high-low)', board_execution_evidence=BOARD_EVIDENCE,
            odd_execution_evidence=ODD_EVIDENCE, odd_price_formula='low+fraction*(high-low)',
            board_capacity='min(regular_session_shares,prior_total_adv20)*0.01_floor_whole_lots',
            odd_capacity='floor(independent_intraday_odd_shares*0.01)',
            board_participation=PARTICIPATION, odd_participation=PARTICIPATION,
            board_order_time=OPEN, board_expires_at=END,
            odd_order_time=ODD_OPEN, odd_expires_at=ODD_END,
            regular_source_window='09:00:00<=time<13:34:00_including_delayed_close',
            volume_policy='verified_source_daily_range_proxy_not_price_level_volume',
            sell_price_policy='daily_range_fraction_proxy_both_channels',
            intraday_tick_verified=False, odd_tick_verified=False, board_tick_verified=False,
            actual_fill_verified=False, within_window_execution_verified=False,
            price_level_volume_verified=False, live_qualified=False, unseen_validation=False)
        evidence = account['ordinary_volume_evidence']
        evidence.update(policy='ordinary_daily_range_proxy_with_parent_tape_quality_gates',
            price_level_volume_verified=False, within_window_execution_verified=False,
            daily_range_proxy=True)
        return account


class RangeGapOrders(DataGapOrders, RangeOrders):
    """Retain the sealed rollback; adapt only its new matcher's diagnostic stage."""
    def _execute_order(self, *args, **kwargs):
        self.execution_gap_stage = 'execution_context'
        before = len(self.data_gap_exclusions)
        filled = super()._execute_order(*args, **kwargs)
        if len(self.data_gap_exclusions) != before:
            gap = self.data_gap_exclusions[-1]
            gap['failure_stage'] = self.execution_gap_stage
            self.orders[gap['order_count_before']]['failure_stage'] = self.execution_gap_stage
        return filled
