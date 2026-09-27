"""Precommitted T+1 tick execution beneath the sealed five-slot risk layers."""
from copy import deepcopy
import math

import pandas as pd

from skills.board_only_verified_replay import BoardOnlyVerifiedBenchmark
from skills.execution_stress import StressOrder
from skills.intraday_limit_replay import limit_price, match_ticks
from skills.million_replay import money
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.residual_slot_replay import ResidualSlotReplay


class TickPlanning:
    def __init__(self, *args, ticks, participation=.01, **kwargs):
        if participation not in (.01, .005):
            raise ValueError('Use a preregistered participation limit')
        super().__init__(*args, **kwargs)
        self.ticks, self.tick_participation = ticks, participation
        self.tick_plans, self.day_plans, self.tick_attempts = [], {}, set()

    def prior(self, day, sid):
        index = self.positions[day]
        if not index or not self.raw(self.days[index-1], sid):
            return None
        return super().prior(day, sid)

    def corporate_day(self, day):
        opening_cash = self.cash
        income = super().corporate_day(day)
        # No current quote or tick is read here. Only known corporate actions,
        # previous marks, dated signals and opening cash may size today's order.
        self.day_plans, self.tick_attempts = {}, set()
        prior_day = str(self.days[self.positions[day]-1].date())
        cash = min(opening_cash, self.cash)
        occupied = set() if self.benchmark else set(self.opening_members)
        for sid, holding in self.holdings.items():
            if sid == '0050' or not holding['qty'] or holding['due_index'] > self.positions[day]:
                continue
            state = self.exit_states.get(holding['event_id'])
            if not state or not state['trigger_reason']:
                raise ValueError('Sell plan requires an earlier latched exit')
            self._plan(day, sid, 'sell', holding['event_id'], state['signal_date'],
                       holding['qty']//1000*1000, 0., opening_cash, None)
        candidates = (dict(members=['0050'], event_id='benchmark', signal_date=prior_day),) if self.benchmark else self.events.get(day, [])
        for event in candidates:
            sid, eid = event['members'][0], event['event_id']
            price = self.prior(day, sid)
            amount = float(self.amount20.at[day, sid])
            allocation = min(cash, self.previous_nav/(1 if self.benchmark else self.slots))
            budget = allocation if self.benchmark else min(allocation, self.residual_budget)
            failure = None
            if not self.benchmark:
                if sid in self.holdings or sid in occupied or len(occupied) >= self.slots:
                    failure = 'opening_slots_locked'
                elif self.residual_block:
                    failure = 'residual_exposure_cap'
                elif not math.isfinite(amount) or amount < 50e6:
                    failure = 'prior_liquidity_below_50m_or_missing'
            limit = limit_price(price, sid, 'buy') if price else None
            # Keep the sealed engine's sizing cushion; reserve the entire
            # position budget so an unfilled order cannot fund a later stock.
            raw_qty = max(0, math.floor((allocation-40)/(price*(1+.001425+.0045)))) if price else 0
            attempted = failure is None and raw_qty > 0
            qty = max(0, math.floor((budget-40)/(price*(1+.001425+.0045)))) if price else 0
            qty = self._affordable(qty//1000*1000, 1000, limit, budget, sid) if limit else 0
            if failure or not qty:
                failure = failure or 'cash_below_one_lot_or_missing_prior'
                qty = 0
            self._plan(day, sid, 'buy', eid, event['signal_date'], qty,
                       allocation if attempted else 0., opening_cash, failure)
            # The sealed resource layer also locks a sub-lot attempt's budget
            # and slot. A zero board fill must not free them for later plans.
            if attempted:
                cash = money(cash-allocation)
                occupied.add(sid)
        return income

    def _plan(self, day, sid, side, eid, signal, qty, budget, opening_cash, failure):
        prior_day = str(self.days[self.positions[day]-1].date())
        if not signal or signal > prior_day or (side == 'buy' and signal != prior_day):
            raise ValueError('Buy must be T+1; sell must have an earlier signal')
        reference = self.prior(day, sid)
        row = dict(date=str(day.date()), stock_id=sid, side=side, event_id=eid,
                   signal_date=signal, reference_date=prior_day, prior_reference=reference,
                   limit_price=limit_price(reference, sid, side) if reference else None,
                   planned_qty=qty, reserved_cash=budget, opening_cash=opening_cash,
                   rejection=failure, order_time='09:01:00', expires_at='13:25:00')
        key = (eid, side)
        if key in self.day_plans:
            raise ValueError('Duplicate precommitted order')
        self.day_plans[key] = row
        self.tick_plans.append(deepcopy(row))

    def buy_etf(self, day, reason):
        if not self.benchmark or '0050' in self.tick_attempts:
            return
        plan = self.day_plans.get(('benchmark', 'buy'))
        if not plan or not plan['planned_qty']:
            return
        self.corporate.prepare('0050')
        self.holdings.setdefault('0050', dict(qty=0, event_id='benchmark', due_index=None))
        self.order(day, '0050', 'buy', plan['planned_qty'], reason, 'benchmark', plan['signal_date'])

    def run(self):
        account = super().run()
        account['tick_plans'] = self.tick_plans
        account['settings'].update(execution='five_slot_precommitted_ticks_v1',
            participation=self.tick_participation, slippage=self.stress_slippage,
            live_qualified=False, odd_tick_verified=False, unseen_validation=False)
        return account


class TickExecution(StressOrder):
    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        plan = self.day_plans.get((event_id, side))
        if plan is None:
            raise ValueError('Order has no precommitted plan')
        if sid in self.tick_attempts:
            raise ValueError('Cannot reuse stock/day tick volume')
        self.tick_attempts.add(sid)
        if plan['stock_id'] != sid or plan['signal_date'] != signal_date:
            raise ValueError('Order identity differs from precommitted plan')
        planned = plan['planned_qty']
        if qty < planned:
            raise ValueError(f'Post-plan resource sizing shrank the committed order: {day.date()} {sid} '
                             f'{side} requested={qty} planned={planned} budget={plan["reserved_cash"]} '
                             f'cash={self.cash} active_budget={self.active_budget}')
        limit = plan['limit_price']
        adv, amount = (float(frame.at[day, sid]) for frame in (self.volume20, self.amount20))
        row = dict(date=str(day.date()), stock_id=sid, name=self.names.get(sid, sid),
                   side=side, event_id=event_id, signal_date=signal_date, channel='board',
                   requested_qty=planned, filled_qty=0, reason=reason, failure=plan['rejection'],
                   prior_avg_volume20=adv if math.isfinite(adv) else None,
                   prior_avg_amount20=amount if math.isfinite(amount) else None,
                   reference_price=limit, limit_price=limit, day_volume=self.raw(day,sid,'volume'),
                   order_time=plan['order_time'], expires_at=plan['expires_at'])
        if not planned or not limit or not math.isfinite(adv) or adv <= 0:
            row['failure'] = row['failure'] or 'missing_previous_price_or_adv'
            self.orders.append(row)
            return 0
        limits = self.feeds.get_limits(sid).get(row['date'])
        if not limits:
            raise ReplayDataUnavailable(f'Missing dated price limits: {sid} {day.date()}')
        if not (0 < limits['lower'] <= limit <= limits['upper']):
            row['failure'] = 'precommitted_limit_outside_legal_range'
            self.orders.append(row)
            return 0
        tape, digest = self.ticks.get(sid, row['date'], self.markets[sid])
        high, low, volume = (self.raw(day,sid,k) for k in ('high','low','volume'))
        if (not high or not low or not volume or tape.price.max() > high+1e-6
                or tape.price.min() < low-1e-6 or int(tape.shares.sum()) > volume*1.01):
            raise ReplayDataUnavailable(f'Tick/daily price or unit conflict: {sid} {day.date()}')
        row.update(ticks_sha256=digest, tick_daily_volume_ratio=float(tape.shares.sum()/volume),
                   **match_ticks(tape,side,limit,planned,adv,self.tick_participation))
        filled = row['filled_qty']
        if filled:
            paid = self._costs(limit,filled,side,sid)
            if side == 'buy' and -paid['cash_change'] > plan['reserved_cash']+.005:
                raise ValueError('Fill exceeds precommitted cash')
            if money(self.cash+paid['cash_change']) < 0:
                raise ValueError('Tick execution overdraws account')
            self.cash_move(day,side,paid['cash_change'],stock_id=sid,event_id=event_id,channel='board')
            self.holdings[sid]['qty'] += filled if side == 'buy' else -filled
            mark = self.raw(day,sid)
            self.marks[sid] = dict(price=mark,date=row['date'])
            self.used[(sid,'board')] += filled
            self.day_cost += paid['total_cost']
            self.day_basis += filled*(mark-limit)*(1 if side=='buy' else -1)
            trade = dict(row, **paid, qty=filled, cash_after=self.cash,
                remaining_shares=self.holdings[sid]['qty'], day_participation=filled/volume,
                sequence=len(self.trades)+1)
            trade.pop('filled_qty'); trade.pop('failure')
            self.trades.append(trade)
        if filled < planned:
            row['failure'] = 'partial_trade_through_capacity' if filled else 'no_trade_through_capacity'
        self.orders.append(row)
        return filled


class ResidualTickReplay(TickPlanning, ResidualSlotReplay, TickExecution):
    pass


class ResidualTickBenchmark(TickPlanning, BoardOnlyVerifiedBenchmark, TickExecution):
    pass


def audit_tick_plans(account, ticks, markets, quotes, calendar, corporate):
    days = pd.DatetimeIndex(calendar)
    if not days.is_unique or not days.is_monotonic_increasing:
        raise ValueError('Audited calendar must be unique and increasing')
    previous = {str(days[i].date()):str(days[i-1].date()) for i in range(1,len(days))}
    source = quotes.copy()
    source['date'] = pd.to_datetime(source['date'])
    if source.duplicated(['date','stock_id']).any():
        raise ValueError('Duplicate audit price source')
    close = source.pivot(index='date',columns='stock_id',values='close').reindex(days)
    volumes = source.pivot(index='date',columns='stock_id',values='volume').reindex(days)
    adv20 = volumes.where(volumes>=0).rolling(20,min_periods=20).mean().shift(1)
    amount20 = (close*volumes).rolling(20,min_periods=20).mean().shift(1)
    plans = {}
    daily = account['daily']
    prior_cash = {r['date']: daily[i-1]['cash'] if i else account['settings']['initial_cash']
                  for i,r in enumerate(daily)}
    reserved = {}
    for plan in account['tick_plans']:
        key = (plan['date'],plan['event_id'],plan['side'])
        if (key in plans or plan['signal_date'] > plan['reference_date']
                or previous.get(plan['date']) != plan['reference_date']):
            raise ValueError('Invalid plan identity or timing')
        if (type(plan['planned_qty']) is not int or plan['planned_qty']<0 or plan['planned_qty']%1000
                or not math.isfinite(plan['reserved_cash']) or plan['reserved_cash']<0
                or plan['order_time']!='09:01:00' or plan['expires_at']!='13:25:00'):
            raise ValueError('Invalid plan quantity, cash or time window')
        if plan['side']=='buy' and plan['signal_date'] != plan['reference_date']:
            raise ValueError('Buy was not planned for next session')
        ref = plan['prior_reference']
        price = float(close.at[pd.Timestamp(plan['reference_date']),plan['stock_id']])
        expected = price if math.isfinite(price) and price>0 else None
        if expected and hasattr(corporate,'reference_price'):
            expected = corporate.reference_price(plan['stock_id'],plan['date'],expected)
        if ref != expected:
            raise ValueError('Plan reference differs from dated price source')
        if plan['limit_price'] != (limit_price(ref,plan['stock_id'],plan['side']) if ref else None):
            raise ValueError('Plan limit differs from prior reference')
        plans[key] = plan
        reserved[plan['date']] = reserved.get(plan['date'],0.)+plan['reserved_cash']
        if reserved[plan['date']] > prior_cash[plan['date']]+.01:
            raise ValueError('Plans use same-day cash')
    seen = set()
    matched = {}
    for row in account['orders']:
        if 'ticks_sha256' not in row:
            continue
        key = (row['date'],row['event_id'],row['side'])
        plan = plans[key]
        stockday = (row['stock_id'],row['date'])
        if stockday in seen or row['requested_qty'] != plan['planned_qty'] or row['limit_price'] != plan['limit_price']:
            raise ValueError('Execution differs from frozen plan or reuses ticks')
        seen.add(stockday)
        tape,digest = ticks.get(row['stock_id'],row['date'],markets.get(stockday,markets.get(row['stock_id'])))
        day,sid = pd.Timestamp(row['date']),row['stock_id']
        if (row['prior_avg_volume20'] != adv20.at[day,sid]
                or row['prior_avg_amount20'] != amount20.at[day,sid]):
            raise ValueError('Execution liquidity differs from prior source window')
        expected = match_ticks(tape,row['side'],plan['limit_price'],plan['planned_qty'],
                               row['prior_avg_volume20'],account['settings']['participation'])
        if digest != row['ticks_sha256'] or any(row[k] != v for k,v in expected.items()):
            raise ValueError('Independent tick replay differs')
        matched[key] = row
    fills = {}
    for trade in account['trades']:
        key = (trade['date'],trade['event_id'],trade['side'])
        row = matched.get(key)
        if not row or trade['reference_price'] != plans[key]['limit_price']:
            raise ValueError('Fill lacks verified precommitted tick order')
        fills[key] = fills.get(key,0)+trade['qty']
    if any(fills.get(k,0) != r['filled_qty'] for k,r in matched.items()):
        raise ValueError('Tick quantities disagree with trade ledger')
    return dict(precommitted_limits=True, opening_cash_reserved=True, tick_fills_rebuilt=True)
