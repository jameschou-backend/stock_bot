"""Chronological board estimates with a disclosed intraday odd-lot HL2 proxy.

Only the odd-lot source/window/price model changes from the sealed executable
study. Daily high/low and volume do not prove when, or whether, our order filled.
Ordinary allocation and whole-order missing-data rollback remain unchanged.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR

from skills.million_replay import money
from skills.poc_executable_replay import (
    END, OPEN, PARTICIPATION, ExecutableOrders, finite, match_board_prints,
)
from skills.poc_gap_execution import DataGapOrders
from skills.replay_market_feeds import ReplayDataUnavailable


MODEL = 'chronological_board_intraday_odd_daily_midpoint_proxy_v1'
SCOPE = 'intraday_odd_session'
EVIDENCE = 'intraday_odd_daily_high_low_midpoint_proxy'
ODD_OPEN, ODD_END = '09:00:00', '13:30:00'


def validate_intraday_odd(row):
    """Require the independently sourced intraday session, including true zeroes."""
    if not isinstance(row, dict):
        raise ReplayDataUnavailable('Missing independent intraday odd daily evidence')
    volume, high, low = (row.get(k) for k in ('odd_shares', 'odd_high', 'odd_low'))
    if (row.get('volume_scope') != SCOPE or row.get('after_hours') is not False
            or row.get('volume_unit') != 'shares' or row.get('price_unit') != 'TWD_per_share'
            or row.get('evidence_status') != 'official_intraday_daily_table'
            or row.get('intraday_tick_verified') is not False
            or row.get('actual_fill_verified') is not False
            or row.get('auction_time') is not None
            or type(volume) is not int or not 0 <= volume <= 2**53-1
            or (volume == 0 and (high is not None or low is not None))
            or (volume > 0 and (not finite(high) or not finite(low) or not 0 < low <= high))):
        raise ReplayDataUnavailable('Missing or inconsistent intraday odd daily evidence')
    return volume, high, low


def match_intraday_odd(row, side, limit_price, quantity, *, used_shares=0, participation=PARTICIPATION):
    """HL2 price proxy and 1% intraday daily volume; no execution clock is inferred."""
    if (side not in ('buy', 'sell') or type(quantity) is not int or not 0 < quantity < 1000
            or type(used_shares) is not int or used_shares < 0
            or not finite(limit_price) or limit_price <= 0
            or not finite(participation) or participation != PARTICIPATION):
        raise ValueError('Invalid preplanned intraday odd order or participation')
    volume, high, low = validate_intraday_odd(row)
    capacity = int((Decimal(volume)*Decimal(str(participation))).to_integral_value(rounding=ROUND_FLOOR))
    if used_shares > capacity:
        raise ValueError('Previously used intraday odd shares exceed daily proxy capacity')
    price = (Decimal(str(high))+Decimal(str(low)))/2 if volume else None
    through = price is not None and (price < Decimal(str(limit_price)) if side == 'buy'
                                    else price > Decimal(str(limit_price)))
    available = max(0, capacity-used_shares) if through else 0
    filled = min(quantity, available)
    failure = ('official_zero_intraday_odd_volume' if not volume
               else 'midpoint_limit_not_crossed' if not through
               else 'intraday_midpoint_capacity_zero' if not filled
               else 'partial_intraday_midpoint_capacity' if filled < quantity else None)
    return dict(filled_qty=filled, capacity_qty=available, daily_capacity_qty=capacity,
        source_volume=volume, source_high=high, source_low=low,
        proxy_price=float(price) if price is not None else None,
        reference_price=float(price) if filled else None, last_fill_time=None,
        actual_fill_time=None, participation_limit=participation, used_shares=used_shares,
        failure=failure, volume_scope=SCOPE, evidence_status='official_intraday_daily_table',
        execution_evidence=EVIDENCE, intraday_tick_verified=False, odd_tick_verified=False,
        within_window_execution_verified=False, price_level_volume_verified=False,
        source_hash_verification_required_by_caller=True,
        actual_fill_verified=False, live_qualified=False)


class IntradayOrders(ExecutableOrders):
    def _plan(self, day, *args, **kwargs):
        super()._plan(day, *args, **kwargs)
        plan = self.tick_plans[-1]
        plan.update(odd_order_time=ODD_OPEN, odd_expires_at=ODD_END)
        self.day_plans[(plan['event_id'], plan['side'])] = deepcopy(plan)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        # This is the sealed ordinary/cash execution body. The odd data/matcher,
        # its two clocks, and the nonfinancial failure-stage marker are changed.
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
                row.update(ticks_sha256=digest, source_volume=int(tape.shares.sum()), tape_evidence=evidence,
                    **match_board_prints(tape, side, limit, requested, adv))
            else:
                odd = self.odd_feeds.get_odd(stamp, sid, self.markets[sid])
                validate_intraday_odd(odd)
                if (odd.get('source_date') != stamp
                        or not isinstance(odd.get('market'), str)
                        or odd['market'].upper() != self.markets[sid].upper()):
                    raise ReplayDataUnavailable('Intraday odd source date or market differs from the order')
                row.update(match_intraday_odd(odd, side, limit, requested))
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
            odd_execution_evidence=EVIDENCE, odd_price_formula='(intraday_odd_high+intraday_odd_low)/2',
            odd_capacity='floor(independent_intraday_odd_shares*0.01)', odd_participation=PARTICIPATION,
            odd_order_time=ODD_OPEN, odd_expires_at=ODD_END,
            sell_price_policy='chronological_board_print_estimate_or_intraday_odd_daily_HL2_proxy',
            intraday_tick_verified=False, odd_tick_verified=False,
            actual_fill_verified=False, within_window_execution_verified=False,
            price_level_volume_verified=False, live_qualified=False, unseen_validation=False)
        return account


class IntradayGapOrders(DataGapOrders, IntradayOrders):
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
