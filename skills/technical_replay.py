"""Preregistered support, position-risk, add and pattern account experiments.

The control calls the sealed loss12 engine unchanged. Experimental variants
copy its explicit daily ordering while reusing all execution/corporate/journal
primitives. Signal quantities and risk budgets use only the prior session.
"""
from __future__ import annotations

from collections import defaultdict
import math

import pandas as pd

from skills.exit_policy import decide_exit
from skills.million_replay import (COMMISSION, MIN_AMOUNT20, MIN_FEE,
    ODD_PARTICIPATION, PARTICIPATION, SLIPPAGE, Replay, costs, money)
from skills.scenario_exit_replay import ScenarioExitReplay


MODES = ('control', 'support20', 'risk2', 'support_risk2',
         'support_risk2_add', 'support_risk2_pattern')
SUPPORT_MODES = frozenset(('support20', 'support_risk2',
                          'support_risk2_add', 'support_risk2_pattern'))
RISK_MODES = frozenset(('risk2', 'support_risk2',
                       'support_risk2_add', 'support_risk2_pattern'))
RISK_BUDGET = .02
ADD_BUDGET = .10


def transaction_cash(price, qty, side, sid):
    """Projected original fees for one board and one odd-lot order, if used."""
    if type(qty) is not int or qty < 0:
        raise ValueError('Projected quantity must be a nonnegative integer')
    return money(sum(costs(price, part, side, sid)['cash_change']
                     for part in (qty//1000*1000, qty%1000) if part))


def projected_loss(price, stop, new_qty, sid, existing_qty=0):
    """Prior-price mark to planned liquidation, plus incremental trade fees.

    Existing acquisition costs are sunk. Pending whole shares participate in
    existing_qty even though their actual sale must wait for delivery. This is
    an estimate at the planned stop, not a bound on gaps or blocked execution.
    """
    if (not math.isfinite(price) or not math.isfinite(stop)
            or price <= 0 or stop <= 0 or type(existing_qty) is not int
            or existing_qty < 0):
        raise ValueError('Projected loss needs positive finite prices and integer shares')
    return max(0., money(money(existing_qty*price)
        - transaction_cash(price, new_qty, 'buy', sid)
        - transaction_cash(stop, existing_qty+new_qty, 'sell', sid)))


def risk_quantity(max_qty, price, stop, risk_cap, sid, existing_qty=0):
    """Largest affordable risk quantity despite per-channel minimum-fee jumps.

    Board/odd fees can drop at thousand-share boundaries. Within a channel
    interval, rounded principal cash can still oscillate by a cent when the
    corporate reference has fractional cents. A monotone lower bound locates
    the last possible candidate; exact descending checks select the maximum.
    """
    if type(max_qty) is not int or max_qty < 0 or not math.isfinite(risk_cap) or risk_cap < 0:
        raise ValueError('Invalid risk quantity cap')
    if stop >= price:
        # Ratcheted stops at/above the prior close must already have latched;
        # this helper is only for positive-distance sizing.
        return 0
    best = 0
    # Only the final few board intervals can be near the risk bound. A gross
    # loss lower bound safely rules out all quantities above this threshold.
    gross_bound = max(0, math.floor((risk_cap+.03)/(price-stop))-existing_qty)
    def lower_bound(quantity):
        fees = sum(costs(value, part, side, sid)['total_cost']
            for value, count, side in ((price, quantity, 'buy'),
                                      (stop, existing_qty+quantity, 'sell'))
            for part in (count//1000*1000, count%1000) if part)
        # Existing mark and each board/odd buy/sell gross can independently
        # round by half a cent: five terms total. Three cents cover that
        # 2.5-cent envelope plus floating-point representation noise.
        return max(0., (existing_qty+quantity)*(price-stop)+fees-.03)
    upper = min(max_qty, gross_bound)
    for board in range(upper//1000, -1, -1):
        left, right = board*1000, min(upper, board*1000+999)
        if right <= best:
            break
        # Sell-fee discontinuity also occurs when existing+new is a full lot.
        boundaries = sorted({left, right+1, *[q for q in
            (math.ceil((existing_qty+left)/1000)*1000-existing_qty,)
            if left < q <= right]})
        for start, end in zip(boundaries, boundaries[1:]):
            low, high = start, end-1
            if lower_bound(low) > risk_cap+1e-9:
                continue
            while low < high:
                mid = (low+high+1)//2
                if lower_bound(mid) <= risk_cap+1e-9:
                    low = mid
                else:
                    high = mid-1
            for candidate in range(low, start-1, -1):
                if projected_loss(price, stop, candidate, sid, existing_qty) <= risk_cap+1e-9:
                    best = max(best, candidate)
                    break
        if best >= left:
            break
    return best


class TechnicalReplay(ScenarioExitReplay):
    def __init__(self, *args, technical_signals, mode='control', **kwargs):
        if mode not in MODES:
            raise ValueError('Unknown technical research mode: '+str(mode))
        super().__init__(*args, exit_signals=technical_signals, mode='loss12', **kwargs)
        if mode != 'control' and not callable(getattr(technical_signals, 'technical_context', None)):
            raise ValueError('Experimental modes require lagged TechnicalSignals')
        self.technical_mode = mode
        self.technical_signals = technical_signals
        self.sizing_decisions, self.add_decisions, self.pattern_decisions = [], [], []
        self.entry_decisions = self.sizing_decisions
        self.entry_plans, self.add_states = {}, {}

    def _technical(self, index, sid):
        context = self.technical_signals.technical_context(index, sid)
        expected = str(self.days[index-1].date()) if index else None
        if context.get('signal_date') != expected or context.get('signal_index') != index-1:
            raise ValueError('Technical signals must use exactly the previous market row')
        return context

    def corporate_day(self, day):
        if self.technical_mode not in SUPPORT_MODES:
            return super().corporate_day(day)
        index = self.positions[day]
        active = {h['event_id'] for sid,h in self.holdings.items() if sid != '0050'}
        active |= {r['event_id'] for r in self.receivables if r.get('qty', 0) > 0}
        for cohort in self.cohorts:
            identity, sid = cohort['event_id'], cohort['stock_id']
            if identity not in active:
                continue
            if identity not in self.exit_states:
                entry_index = self.positions[pd.Timestamp(cohort['entry_date'])]
                anchor = self.exit_signals.price(entry_index, sid)
                self.exit_states[identity] = dict(stock_id=sid, event_id=identity,
                    entry_index=entry_index, entry_price=anchor, peak_price=anchor,
                    trigger_reason=None, signal_date=None, target_date=None, target_index=None,
                    support_floor=self.entry_plans[identity]['support20'])
            state = self.exit_states[identity]
            context = self.exit_signals.context(index, sid, state)
            technical = self._technical(index, sid)
            support = technical['support20']
            if technical['support_available'] and support is not None:
                state['support_floor'] = max(state['support_floor'], support)
            signal_close = self.exit_signals.price(index-1, sid)
            broken = signal_close is not None and signal_close < state['support_floor']
            if state['trigger_reason']:
                decision = dict(exit=True, reason=state['trigger_reason'], phase='exiting', extend=False)
            else:
                decision = decide_exit(context, 'loss12')
                if broken and decision['reason'] != 'loss12':
                    decision = dict(exit=True, reason='support20', phase='exiting', extend=False)
                if decision['exit']:
                    state.update(trigger_reason=decision['reason'], signal_date=str(self.days[index-1].date()),
                                 target_date=str(day.date()), target_index=index)
            self.exit_decisions.append(dict(date=str(day.date()), signal_date=str(self.days[index-1].date()),
                stock_id=sid, event_id=identity, mode=self.technical_mode, **context, **decision,
                signal_close=signal_close, support20=support, support_floor=state['support_floor'],
                support_available=technical['support_available'], support_failure=broken,
                technical_diagnostics=technical['diagnostics'], entry_price=state['entry_price'],
                peak_price=state['peak_price'], first_signal_date=state['signal_date'],
                target_date=state['target_date']))
            if sid in self.holdings and self.holdings[sid]['event_id'] == identity:
                if state['target_index'] is not None:
                    self.holdings[sid]['due_index'] = state['target_index']
        # Bypass ScenarioExitReplay's policy, retaining its wrapped corporate provider.
        income = Replay.corporate_day(self, day)
        for sid, holding in self.holdings.items():
            state = self.exit_states.get(holding['event_id'])
            if state and state['target_index'] is not None:
                holding['due_index'] = state['target_index']
        return income

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        """Allow an otherwise executable residual sale to pay more fee than gross.

        All original price, limit, liquidity and channel checks run first. A
        fee larger than the residual market value does not justify trapping a
        cohort indefinitely when the account can pay the shortfall. Only that
        one original rejection is reconsidered, with no fee waiver or funding.
        """
        first = len(self.orders)
        filled_total = super().order(day, sid, side, qty, reason, event_id, signal_date)
        if side != 'sell':
            return filled_total
        for row in self.orders[first:]:
            if row.get('failure') != 'proceeds_below_costs':
                continue
            channel, reference = row['channel'], row['reference_price']
            step = 1000 if channel == 'board' else 1
            capacity = max(0, row['capacity_qty']-self.used[(sid, channel)])
            remaining = self.holdings.get(sid, {}).get('qty', 0)
            filled = min(row['requested_qty'], capacity, remaining)//step*step
            if not filled:
                continue
            paid = costs(reference, int(filled), 'sell', sid)
            row['negative_proceeds_cash_required'] = max(0., -paid['cash_change'])
            row['negative_proceeds_cash_available'] = self.cash
            if money(self.cash+paid['cash_change']) < 0:
                row['failure'] = 'proceeds_below_costs_insufficient_cash'
                continue
            price = self.raw(day, sid)
            self.cash_move(day, 'sell', paid['cash_change'], stock_id=sid,
                           event_id=row['event_id'], channel=channel)
            self.holdings[sid]['qty'] -= filled
            self.marks[sid] = dict(price=price, date=str(day.date()))
            self.used[(sid, channel)] += filled
            self.day_cost += paid['total_cost']
            self.day_basis -= filled*(price-reference)
            row['negative_proceeds_settlement'] = True
            trade = dict(row, **paid, qty=int(filled), reference_price=reference,
                cash_after=self.cash, remaining_shares=self.holdings[sid]['qty'],
                day_participation=filled/(row['day_volume'] if channel == 'board' else row['odd_volume']),
                sequence=len(self.trades)+1)
            trade.pop('filled_qty')
            trade.pop('failure')
            self.trades.append(trade)
            row['filled_qty'] = int(filled)
            row['failure'] = 'partial_capacity_or_cash' if filled < row['requested_qty'] else None
            filled_total += filled
        return filled_total

    def _entry_plan(self, day, event, opening_nav, previous_price, budget):
        sid = event['members'][0]
        technical = self._technical(self.positions[day], sid)
        signal = technical['adjusted_close']
        support = technical['support20']
        row = dict(date=str(day.date()), signal_date=event['signal_date'], stock_id=sid,
            event_id=event['event_id'], mode=self.technical_mode, prior_nav=opening_nav,
            previous_price=previous_price, adjusted_close=signal, support20=support,
            capital_cap=budget, risk_cap=opening_nav*RISK_BUDGET,
            risk_applied=self.technical_mode in RISK_MODES, planned_stop=None,
            raw_planned_stop=None, planned_risk=None, requested_qty=0, filled_qty=0,
            failure=None, diagnostics=technical['diagnostics'])
        self.sizing_decisions.append(row)
        if signal is None:
            row['failure'] = 'missing_prior_adjusted_close'
        elif self.technical_mode in SUPPORT_MODES and (not technical['support_available'] or support is None or support >= signal):
            row['failure'] = 'initial_support_missing_or_not_below_price'
        if self.technical_mode == 'support_risk2_pattern':
            passed = technical['pattern_pass']
            self.pattern_decisions.append(dict(date=row['date'], signal_date=row['signal_date'],
                stock_id=sid, event_id=event['event_id'], pattern_available=technical['pattern_available'],
                pattern_pass=passed, breakout20=technical['breakout20'],
                contraction10=technical['contraction10'], volume_expansion=technical['volume_expansion'],
                diagnostics=technical['diagnostics']))
            if passed is not True:
                row['failure'] = row['failure'] or ('pattern_not_confirmed' if passed is False else 'pattern_data_missing')
        if row['failure']:
            return 0, budget, row
        stop = max(signal*.88, support) if self.technical_mode in SUPPORT_MODES else signal*.88
        raw_stop = previous_price*stop/signal
        qty = max(0, math.floor((budget-2*MIN_FEE)/(previous_price*(1+COMMISSION+SLIPPAGE))))
        if self.technical_mode in RISK_MODES:
            qty = risk_quantity(qty, previous_price, raw_stop, row['risk_cap'], sid)
            budget = min(budget, -transaction_cash(previous_price, qty, 'buy', sid))
        row.update(planned_stop=stop, raw_planned_stop=raw_stop, requested_qty=qty,
                   planned_risk=projected_loss(previous_price, raw_stop, qty, sid), funding_budget=budget)
        if not qty:
            row['failure'] = 'risk_or_capital_budget_zero'
        return qty, budget, row

    def _fund(self, day, sid, budget, event_id, signal_date, reason='fund_stock'):
        if self.cash < budget and self.holdings.get('0050', {}).get('qty', 0):
            etf_prior = self.prior(day, '0050')
            funding_qty = min(self.holdings['0050']['qty'], math.ceil(
                (budget-self.cash+2*MIN_FEE)/(etf_prior*(1-COMMISSION-SLIPPAGE-.001))))
            self.order(day, '0050', 'sell', int(funding_qty), reason, event_id, signal_date)

    def _add_positions(self, day, opening_nav):
        if self.technical_mode != 'support_risk2_add':
            return
        index = self.positions[day]
        for sid, holding in sorted(list(self.holdings.items()), key=lambda pair: pair[1]['event_id']):
            identity = holding['event_id']
            if sid == '0050' or not holding['qty'] or self.add_states.get(identity):
                continue
            state = self.exit_states.get(identity)
            # Newly entered cohorts have no state until the next market day.
            if not state or state['entry_index'] >= index or state['trigger_reason']:
                continue
            technical = self._technical(index, sid)
            signal, anchor = technical['adjusted_close'], state['entry_price']
            why = None
            if signal is None or anchor is None:
                why = 'add_missing_price_anchor'
            elif signal < 1.10*anchor-1e-12:
                why = 'add_gain_below_10pct'
            elif technical['breakout20'] is not True:
                why = 'add_breakout_not_confirmed'
            elif not math.isfinite(float(self.amount20.at[day, sid])) or self.amount20.at[day, sid] < MIN_AMOUNT20:
                why = 'prior_liquidity_below_50m_or_missing'
            previous = self.prior(day, sid)
            if not previous:
                why = why or 'no_prior_price'
            row = dict(date=str(day.date()), signal_date=technical['signal_date'], stock_id=sid,
                event_id=identity, prior_nav=opening_nav, adjusted_close=signal, entry_price=anchor,
                previous_price=previous, support_floor=state['support_floor'],
                breakout20=technical['breakout20'], requested_qty=0, filled_qty=0,
                failure=why, risk_cap=opening_nav*RISK_BUDGET, planned_risk=None,
                planned_stop=None, raw_planned_stop=None, total_cost=0.)
            self.add_decisions.append(row)
            if why:
                continue
            pending = sum(r.get('qty', 0) for r in self.receivables
                          if r.get('event_id') == identity and r.get('kind') == 'shares')
            existing = holding['qty']+pending
            capital = min(opening_nav*ADD_BUDGET, max(0., opening_nav/self.slots-existing*previous),
                self.cash+self.holdings.get('0050', {}).get('qty', 0)*(self.prior(day, '0050') or 0))
            stop = max(anchor*.88, state['support_floor'])
            raw_stop = previous*stop/signal
            qty_cap = max(0, math.floor((capital-2*MIN_FEE)/(previous*(1+COMMISSION+SLIPPAGE))))
            qty = risk_quantity(qty_cap, previous, raw_stop, row['risk_cap'], sid, existing)
            budget = min(capital, -transaction_cash(previous, qty, 'buy', sid))
            row.update(existing_qty=holding['qty'], pending_whole_qty=pending,
                capital_cap=capital, planned_stop=stop, raw_planned_stop=raw_stop,
                requested_qty=qty, planned_risk=projected_loss(previous, raw_stop, qty, sid, existing),
                funding_budget=budget)
            if not qty:
                row['failure'] = 'add_risk_or_capital_budget_zero'
                continue
            self._fund(day, sid, budget, identity, technical['signal_date'], 'fund_stock_add')
            trade_start = len(self.trades)
            filled = self.order(day, sid, 'buy', qty, 'pyramid_add', identity, technical['signal_date'])
            row.update(filled_qty=filled, total_cost=sum(t['total_cost'] for t in self.trades[trade_start:]),
                       failure=None if filled == qty else 'partial_or_unfilled_execution')
            if filled:
                self.add_states[identity] = dict(date=str(day.date()), signal_date=technical['signal_date'], qty=filled)

    def run(self):
        if self.technical_mode == 'control':
            return super().run()
        return self._run_experiment()

    def _run_experiment(self):
        peak = self.initial_cash
        for day in self.days[(self.days>=self.start)&(self.days<=self.end)]:
            self.used = defaultdict(int)
            self.day_cost = self.day_basis = 0.
            opening_nav = self.previous_nav
            old_assets = sum(h['qty']*self.marks[sid]['price'] for sid,h in self.holdings.items() if h['qty'])+self.receivable_value()
            income = self.corporate_day(day)
            # Marks are valuation only; missing current quotes remain visibly stale.
            for sid in {s for s,h in self.holdings.items() if h['qty']}|{r['stock_id'] for r in self.receivables}:
                price = self.raw(day,sid)
                if price:
                    self.marks[sid] = dict(price=price,date=str(day.date()))
                elif sid not in self.marks:
                    raise ValueError('No valuation price for held asset '+sid)
            marked_assets = sum(h['qty']*self.marks[sid]['price'] for sid,h in self.holdings.items() if h['qty'])+self.receivable_value()
            # Paid dividends migrate receivable to cash and are not market P&L.
            paid_today = sum(r['cash_change'] for r in self.cash_ledger if r['date']==str(day.date()) and r['kind'] in ('dividend_payment','fractional_share_payment'))
            market_pnl = marked_assets-old_assets-income+paid_today
            if day == self.start:
                self.buy_etf(day,'initial_allocation')
            for sid,holding in list(self.holdings.items()):
                if sid!='0050' and holding['qty'] and self.positions[day] >= holding['due_index']:
                    self.order(day,sid,'sell',holding['qty'],'scheduled_exit',holding['event_id'])
            # Zero positions can close only after outstanding stock rights deliver.
            for sid,h in list(self.holdings.items()):
                if sid!='0050' and h['qty']==0 and not any(r.get('qty',0)>0 and r['event_id']==h['event_id'] for r in self.receivables):
                    cohort = next(c for c in self.cohorts if c['event_id']==h['event_id'])
                    cohort['exit_date'] = str(day.date())
                    del self.holdings[sid]
            for event in self.events.get(day,[]):
                sid = event['members'][0]
                why = None
                if sid in self.holdings:
                    why = 'overlapping_member'
                elif len([h for s,h in self.holdings.items() if s!='0050']) >= self.slots:
                    why = 'slots_full'
                elif not math.isfinite(float(self.amount20.at[day,sid])) or self.amount20.at[day,sid]<MIN_AMOUNT20:
                    why = 'prior_liquidity_below_50m_or_missing'
                previous_price = self.prior(day,sid)
                if why or not previous_price:
                    self.orders.append(dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),
                        event_id=event['event_id'],signal_date=event['signal_date'],side='buy',channel='event',
                        requested_qty=0,filled_qty=0,reason='leader_entry',failure=why or 'no_prior_price'))
                    continue
                budget = min(opening_nav/self.slots, self.cash+self.holdings.get('0050',{}).get('qty',0)*(self.prior(day,'0050') or 0))
                qty, budget, plan = self._entry_plan(day, event, opening_nav, previous_price, budget)
                if not qty:
                    self.orders.append(dict(date=str(day.date()), stock_id=sid, name=self.names.get(sid,sid),
                        event_id=event['event_id'], signal_date=event['signal_date'], side='buy', channel='event',
                        requested_qty=0, filled_qty=0, reason='leader_entry', failure=plan['failure']))
                    continue
                self.corporate.prepare(sid)
                self._fund(day, sid, budget, event['event_id'], event['signal_date'])
                due = self.positions[day]+self.horizon
                self.holdings[sid] = dict(qty=0,event_id=event['event_id'],due_index=due)
                if self.raw(day,sid):
                    self.marks[sid] = dict(price=self.raw(day,sid),date=str(day.date()))
                filled = self.order(day,sid,'buy',qty,'leader_entry',event['event_id'],event['signal_date'])
                plan['filled_qty'] = filled
                plan['failure'] = None if filled == qty else 'partial_or_unfilled_execution'
                if filled:
                    self.entry_plans[event['event_id']] = dict(plan)
                    self.cohorts.append(dict(event,stock_id=sid,name=self.names.get(sid,sid),bought_qty=filled,
                        due_index=due,due_date=str(self.days[due].date()) if due<len(self.days) else None,exit_date=None))
                else:
                    del self.holdings[sid]
            self._add_positions(day, opening_nav)
            self.buy_etf(day,'idle_cash')
            assets = 0.
            for sid,h in self.holdings.items():
                if not h['qty']:
                    continue
                if sid not in self.marks:
                    self.marks[sid] = dict(price=self.raw(day,sid),date=str(day.date()))
                mark = self.marks[sid]
                value = h['qty']*mark['price']; assets+=value
                self.holding_rows.append(dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),
                    qty=h['qty'],price=mark['price'],market_value=value,mark_date=mark['date'],
                    event_id=h['event_id'],stale=mark['date']!=str(day.date())))
            receivable = self.receivable_value()
            nav = self.cash+assets+receivable
            expected = opening_nav+market_pnl+income+self.day_basis-self.day_cost
            if not math.isclose(nav,expected,abs_tol=.06,rel_tol=1e-10):
                raise ValueError(f'Daily account does not reconcile {day.date()}: {nav} vs {expected}')
            peak = max(peak,nav)
            self.daily.append(dict(date=str(day.date()),opening_nav=opening_nav,market_pnl=market_pnl,
                dividend_entitlement=income,execution_basis_pnl=self.day_basis,cost=self.day_cost,
                cash=self.cash,market_value=assets,receivable=receivable,nav=nav,
                daily_return=nav/opening_nav-1,total_return=nav/self.initial_cash-1,
                drawdown=nav/peak-1,holdings=len([s for s,h in self.holdings.items() if s!='0050' and h['qty']]),
                stale_holdings=sum(r['stale'] for r in self.holding_rows if r['date']==str(day.date()))))
            self.previous_nav = nav
        return dict(daily=self.daily,trades=self.trades,orders=self.orders,corporate_actions=self.actions,
            cash_ledger=self.cash_ledger,holdings=self.holding_rows,cohorts=self.cohorts,
            receivables=self.receivables,settings=dict(initial_cash=self.initial_cash,slots=self.slots,
                horizon=self.horizon,slippage=SLIPPAGE,commission=COMMISSION,minimum_fee=MIN_FEE,
                participation=PARTICIPATION,odd_participation=ODD_PARTICIPATION,benchmark=self.benchmark))
