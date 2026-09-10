"""Integer-share historical account with explicit liquidity and cash constraints.

Prices are unadjusted. A market-data provider supplies independently dated odd
lot observations and price limits; a corporate provider supplies real rights.
No network or database is imported by this accounting engine.
"""
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math

import numpy as np
import pandas as pd


INITIAL_CASH = 1_000_000.
SLIPPAGE = .0045
COMMISSION = .001425
MIN_FEE = 20.
MIN_AMOUNT20 = 50_000_000.
PARTICIPATION = .01
ODD_PARTICIPATION = .05
MIN_IDLE_BUY = 5000.


def money(value):
    return float(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def costs(price, qty, side, sid):
    if type(qty) is not int or qty <= 0 or not math.isfinite(price) or price <= 0:
        raise ValueError('A trade requires positive reference price and integer shares')
    gross = Decimal(str(price)) * qty
    fee = max(Decimal('20'), (gross * Decimal(str(COMMISSION))).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    tax = (gross * Decimal('.001' if sid == '0050' else '.003')).quantize(Decimal('1'), rounding=ROUND_FLOOR) if side == 'sell' else Decimal(0)
    slip = (gross * Decimal(str(SLIPPAGE))).quantize(Decimal('1'), rounding=ROUND_CEILING)
    amount = money(gross)
    expense = float(fee + tax + slip)
    return dict(gross=amount, commission=float(fee), tax=float(tax), slippage=float(slip),
                total_cost=expense, cash_change=money((amount if side == 'sell' else -amount)-expense))


def affordable(qty, step, price, cash, sid):
    """Reduce a pre-existing order limit; never enlarge it using today's price."""
    low, high = 0, qty // step
    while low < high:
        middle = (low + high + 1) // 2
        needed = -costs(price, int(middle * step), 'buy', sid)['cash_change']
        if needed <= cash + 1e-7:
            low = middle
        else:
            high = middle - 1
    return int(low * step)


class UnresolvedAction(ValueError):
    pass


class Replay:
    def __init__(self, quotes, companies, calendar, events, feeds, corporate, *,
                 start='2022-01-03', end='2026-09-09', initial_cash=INITIAL_CASH,
                 slots=3, horizon=63, benchmark=False):
        if initial_cash <= 0 or slots < 1 or horizon < 1:
            raise ValueError('Invalid account settings')
        self.days = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
        if not self.days.is_unique or self.days.tz is not None:
            raise ValueError('Unique naive market dates required')
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        if self.start not in self.days or self.end not in self.days:
            raise ValueError('Requested account endpoints must exist in the audited calendar')
        self.positions = {d: i for i, d in enumerate(self.days)}
        q = quotes.copy()
        q['date'] = pd.to_datetime(q['date'])
        if q.duplicated(['stock_id', 'date']).any():
            raise ValueError('Duplicate raw prices')
        self.fields = {key: q.pivot(index='date', columns='stock_id', values=key).reindex(self.days)
                       for key in ('open', 'high', 'low', 'close', 'volume')}
        self.amount20 = (self.fields['close'] * self.fields['volume']).rolling(20, min_periods=20).mean().shift(1)
        # An observed zero is a zero-volume session, not missing evidence.
        self.volume20 = self.fields['volume'].where(self.fields['volume'] >= 0).rolling(20, min_periods=20).mean().shift(1)
        self.names = dict(zip(companies.stock_id, companies.name))
        self.markets = dict(zip(companies.stock_id, companies.market))
        self.names['0050'], self.markets['0050'] = '元大台灣50', 'TWSE'
        self.feeds, self.corporate = feeds, corporate
        self.slots, self.horizon, self.benchmark = slots, horizon, benchmark
        self.cash = float(initial_cash)
        self.initial_cash, self.previous_nav = float(initial_cash), float(initial_cash)
        self.holdings, self.marks, self.receivables, self.cohorts = {}, {}, [], []
        self.trades, self.orders, self.actions, self.daily, self.holding_rows = [], [], [], [], []
        self.cash_ledger = [dict(date=str(self.start.date()), kind='initial_deposit', cash_change=self.cash, cash_after=self.cash)]
        self.events = defaultdict(list)
        identities = set()
        for e in events:
            if e['event_id'] in identities or len(e['members']) != 1:
                raise ValueError('Unique single-stock events required')
            identities.add(e['event_id'])
            signal, entry = pd.Timestamp(e['signal_date']), pd.Timestamp(e['entry_date'])
            if signal not in self.positions or entry not in self.positions or self.positions[entry] != self.positions[signal]+1:
                raise ValueError('Entries must be the first audited market day after signals')
            if self.start <= entry <= self.end and not benchmark:
                self.events[entry].append(deepcopy(e))
        for d in self.events:
            self.events[d].sort(key=lambda e: (-e['priority'], e['event_id']))

    def raw(self, day, sid, key='close'):
        try:
            value = float(self.fields[key].at[day, sid])
            return value if math.isfinite(value) and value > 0 else None
        except (KeyError, TypeError, ValueError):
            return None

    def prior(self, day, sid):
        i = self.positions[day]
        if i == 0:
            return None
        # A last known mark may plan a quantity, but cannot authorize execution.
        values = self.fields['close'][sid].iloc[:i]
        good = values[np.isfinite(values) & (values > 0)]
        if not len(good):
            return None
        value = float(good.iloc[-1])
        if hasattr(self.corporate,'reference_price'):
            value = self.corporate.reference_price(sid,str(day.date()),value)
        return value

    def cash_move(self, day, kind, change, **extra):
        self.cash = money(self.cash + change)
        if self.cash < -.005:
            raise ValueError('Account overdraft')
        self.cash_ledger.append(dict(date=str(day.date()), kind=kind, cash_change=money(change), cash_after=self.cash, **extra))

    def corporate_day(self, day):
        income = 0.
        for sid, holding in list(self.holdings.items()):
            if not holding['qty']:
                continue
            for action in self.corporate.on_date(sid, str(day.date())):
                qty = holding['qty']
                kind = action['kind']
                if kind in ('cash_dividend','stock_dividend') and self.raw(day,sid) is None:
                    raise UnresolvedAction(f'Ex-date valuation price missing: {sid} {day.date()}')
                row = dict(action, date=str(day.date()), entitled_qty=qty, event_id=holding['event_id'])
                if kind == 'cash_dividend':
                    amount = money(qty * action['cash_per_share'])
                    if action.get('cash_rounding') == 'floor_ntd':
                        amount = float((Decimal(qty)*Decimal(str(action['cash_per_share']))).to_integral_value(rounding=ROUND_FLOOR))
                    self.receivables.append(dict(stock_id=sid, amount=amount, kind='cash',
                        pay_date=action.get('pay_date'), action_id=action['action_id'],
                        event_id=holding['event_id'], ex_date=str(day.date())))
                    row['entitlement_value'] = amount
                    income += amount
                elif kind == 'split':
                    quantity = Decimal(qty) * Decimal(str(action['multiplier']))
                    if quantity != quantity.to_integral_value():
                        raise UnresolvedAction(f'Fractional split settlement missing: {sid} {day.date()}')
                    holding['qty'] = int(quantity)
                    if sid in self.marks:
                        self.marks[sid]['price'] /= action['multiplier']
                    row['qty_after'] = holding['qty']
                elif kind == 'stock_dividend':
                    quantity = Decimal(qty) * Decimal(str(action['shares_per_share']))
                    whole = int(quantity.to_integral_value(rounding=ROUND_FLOOR))
                    fraction = float(quantity - whole)
                    if not action.get('pay_date'):
                        raise UnresolvedAction(f'Stock delivery date missing: {sid} {day.date()}')
                    if fraction and action.get('fractional_cash_per_share') is None:
                        raise UnresolvedAction(f'Fractional share settlement missing: {sid} {day.date()}')
                    self.receivables.append(dict(stock_id=sid, qty=whole, fraction=fraction,
                        kind='shares', pay_date=action['pay_date'], action_id=action['action_id'],
                        event_id=holding['event_id'], ex_date=str(day.date()),
                        fractional_cash_per_share=action.get('fractional_cash_per_share')))
                    row.update(whole_new_shares=whole, fractional_right=fraction)
                    # Shares receivable offset the ex-rights price drop in NAV.
                elif kind == 'waive_subscription':
                    row['note'] = '不認購現金增資，未認列或變現認股權價值'
                else:
                    raise UnresolvedAction(f'Unresolved corporate action: {sid} {day.date()} {kind}')
                self.actions.append(row)
        pending = []
        for entitlement in self.receivables:
            pay = entitlement.get('pay_date')
            if pay and pd.Timestamp(pay) <= day:
                if entitlement['kind'] == 'cash':
                    self.cash_move(day, 'dividend_payment', entitlement['amount'],
                                   stock_id=entitlement['stock_id'], action_id=entitlement['action_id'])
                else:
                    sid = entitlement['stock_id']
                    if sid not in self.holdings:
                        # The original position can have sold before stock delivery.
                        old = next(c for c in self.cohorts if c['event_id'] == entitlement['event_id'])
                        self.holdings[sid] = dict(qty=0, event_id=old['event_id'], due_index=old['due_index'])
                    self.holdings[sid]['qty'] += entitlement['qty']
                    if entitlement['fraction']:
                        rate = entitlement.get('fractional_cash_per_share')
                        if rate is None:
                            # A right below one share remains an explicitly estimated receivable.
                            residual = dict(entitlement, qty=0, pay_date=None, kind='fraction')
                            pending.append(residual)
                        else:
                            self.cash_move(day, 'fractional_share_payment', money(entitlement['fraction']*rate), stock_id=sid)
                self.actions.append(dict(date=str(day.date()), stock_id=entitlement['stock_id'],
                    kind='payment' if entitlement['kind']=='cash' else 'share_delivery', **{
                        k:v for k,v in entitlement.items() if k not in ('stock_id','kind')}))
            else:
                pending.append(entitlement)
        self.receivables = pending
        return income

    def receivable_value(self):
        amount = 0.
        for item in self.receivables:
            if item['kind'] == 'cash':
                amount += item['amount']
            else:
                amount += item.get('qty', 0) * self.marks[item['stock_id']]['price']
                rate = item.get('fractional_cash_per_share')
                if item['fraction'] and rate is None:
                    raise UnresolvedAction('Fractional entitlement has no settlement terms')
                amount += item['fraction'] * (rate or 0)
        return amount

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        """A pre-sized order may fill partly. No failed remainder is hidden."""
        if type(qty) is not int or qty < 0:
            raise ValueError('Order shares must be integer and nonnegative')
        if qty == 0:
            return 0
        if side == 'sell':
            qty = min(qty, self.holdings.get(sid, {}).get('qty', 0))
        total_filled = 0
        day_text = str(day.date())
        price, volume = self.raw(day,sid), self.raw(day,sid,'volume')
        high, low = self.raw(day,sid,'high'), self.raw(day,sid,'low')
        adv = float(self.volume20.at[day,sid])
        amount = float(self.amount20.at[day,sid])
        # Limits are dated observations; provider failure is a preparation error.
        limits = self.feeds.get_limits(sid).get(day_text)
        for channel, request, step in [('board', qty//1000*1000, 1000), ('odd', qty%1000, 1)]:
            if request == 0:
                continue
            row = dict(date=day_text, stock_id=sid, name=self.names.get(sid,sid), side=side,
                channel=channel, requested_qty=request, filled_qty=0, reason=reason,
                event_id=event_id, signal_date=signal_date, failure=None,
                prior_avg_volume20=adv if math.isfinite(adv) else None,
                prior_avg_amount20=amount if math.isfinite(amount) else None,
                day_volume=volume, price_limit_source='FinMind' if limits else 'unavailable')
            ref, cap = price, 0
            if not price or not volume or not high or not low:
                row['failure'] = 'missing_or_zero_quote_volume'
            elif high == low:
                row['failure'] = 'single_price_session'
            elif not limits:
                row['failure'] = 'missing_price_limits'
            elif limits['upper'] == limits['lower'] == 0:
                row['failure'] = 'unsupported_no_price_limit_session'
            elif side == 'buy' and price >= limits['upper']-1e-8:
                row['failure'] = 'at_upper_limit'
            elif side == 'sell' and price <= limits['lower']+1e-8:
                row['failure'] = 'at_lower_limit'
            elif channel == 'board':
                if not math.isfinite(adv) or adv <= 0:
                    row['failure'] = 'missing_adv20'
                else:
                    capacity = int(min(volume,adv)*PARTICIPATION)//1000*1000
                    used = self.used[(sid,channel)]
                    cap = max(0,capacity-used)
                    row.update(capacity_qty=capacity, participation_limit=PARTICIPATION)
            else:
                odd = self.feeds.get_odd(day_text,sid,self.markets[sid])
                if not odd or not odd.get('odd_shares') or not odd.get('odd_last'):
                    row['failure'] = 'no_odd_lot_trade'
                elif not odd.get('odd_bid') or not odd.get('odd_ask') or odd['odd_bid'] > odd['odd_ask']:
                    row['failure'] = 'invalid_odd_lot_quote'
                elif odd.get('ask_qty' if side == 'buy' else 'bid_qty') == 0:
                    row['failure'] = 'no_odd_lot_opposing_quote_quantity'
                else:
                    ref = odd['odd_last']
                    row.update(odd_volume=odd['odd_shares'], odd_bid=odd['odd_bid'], odd_ask=odd['odd_ask'],
                               odd_bid_qty=odd.get('bid_qty'), odd_ask_qty=odd.get('ask_qty'))
                    if (side == 'buy' and ref >= limits['upper']-1e-8) or (side == 'sell' and ref <= limits['lower']+1e-8):
                        row['failure'] = 'odd_lot_at_price_limit'
                    else:
                        capacity = math.floor(odd['odd_shares']*ODD_PARTICIPATION)
                        cap = max(0,capacity-self.used[(sid,channel)])
                        row.update(capacity_qty=capacity, participation_limit=ODD_PARTICIPATION)
            if not row['failure']:
                fill = min(request,cap)//step*step
                if side == 'buy':
                    fill = affordable(fill,step,ref,self.cash,sid)
                if fill:
                    paid = costs(ref,fill,side,sid)
                    if side == 'sell' and paid['cash_change'] <= 0:
                        row['failure'] = 'proceeds_below_costs'
                        fill = 0
                    else:
                        self.cash_move(day,side,paid['cash_change'],stock_id=sid,event_id=event_id,channel=channel)
                        self.holdings[sid]['qty'] += fill if side == 'buy' else -fill
                        # A zero holding can retain yesterday's mark. A new
                        # fill must always be valued on this session's basis.
                        self.marks[sid] = dict(price=price,date=day_text)
                        self.used[(sid,channel)] += fill
                        self.day_cost += paid['total_cost']
                        self.day_basis += fill * (price-ref) * (1 if side=='buy' else -1)
                        trade = dict(row, **paid, qty=fill, reference_price=ref,
                            cash_after=self.cash, remaining_shares=self.holdings[sid]['qty'],
                            day_participation=fill/(volume if channel=='board' else odd['odd_shares']),
                            sequence=len(self.trades)+1)
                        trade.pop('filled_qty');trade.pop('failure')
                        self.trades.append(trade)
                        row['filled_qty'] = fill
                        total_filled += fill
                if fill < request and not row['failure']:
                    row['failure'] = 'partial_capacity_or_cash' if fill else 'capacity_or_cash_zero'
            row['reference_price'] = ref
            self.orders.append(row)
        return total_filled

    def buy_etf(self, day, reason):
        if self.cash < MIN_IDLE_BUY:
            return
        prior = self.prior(day,'0050')
        if not prior:
            return
        max_qty = max(0, math.floor((self.cash-2*MIN_FEE)/(prior*(1+COMMISSION+SLIPPAGE))))
        self.holdings.setdefault('0050',dict(qty=0,event_id='benchmark',due_index=None))
        self.corporate.prepare('0050')
        self.order(day,'0050','buy',max_qty,reason,'benchmark')

    def run(self):
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
                qty = max(0,math.floor((budget-2*MIN_FEE)/(previous_price*(1+COMMISSION+SLIPPAGE))))
                if not qty:
                    continue
                self.corporate.prepare(sid)
                if self.cash<budget and self.holdings.get('0050',{}).get('qty',0):
                    etf_prior = self.prior(day,'0050')
                    funding_qty = min(self.holdings['0050']['qty'],math.ceil((budget-self.cash+2*MIN_FEE)/(etf_prior*(1-COMMISSION-SLIPPAGE-.001))))
                    self.order(day,'0050','sell',int(funding_qty),'fund_stock',event['event_id'],event['signal_date'])
                due = self.positions[day]+self.horizon
                self.holdings[sid] = dict(qty=0,event_id=event['event_id'],due_index=due)
                if self.raw(day,sid):
                    self.marks[sid] = dict(price=self.raw(day,sid),date=str(day.date()))
                filled = self.order(day,sid,'buy',qty,'leader_entry',event['event_id'],event['signal_date'])
                if filled:
                    self.cohorts.append(dict(event,stock_id=sid,name=self.names.get(sid,sid),bought_qty=filled,
                        due_index=due,due_date=str(self.days[due].date()) if due<len(self.days) else None,exit_date=None))
                else:
                    del self.holdings[sid]
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
