"""Isolated execution stress on sealed accounting primitives; no global patches.

Order and independent audit are explicit derivatives of the sealed replay.
Only price, depth and realized slippage differ; neutral mode must match exactly.
"""
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
import math
import pandas as pd
from skills.million_replay import Replay, costs, money, PARTICIPATION, ODD_PARTICIPATION
from skills.technical_replay import TechnicalReplay

MODES = ('control','depth','quote','slip90','entry_delay','exit_delay','combined')

class StressOrder:
    def _setup_stress(self, mode):
        if mode not in MODES:
            raise ValueError('Unknown execution stress mode')
        self.stress_mode = mode
        self.stress_slippage = .009 if mode in ('slip90','combined') else .0045
        self.stress_depth = mode in ('depth','combined')
        self.stress_quote = mode in ('quote','combined')

    def _costs(self, price, qty, side, sid):
        paid = costs(price,qty,side,sid)
        slip = float((Decimal(str(price))*qty*Decimal(str(self.stress_slippage))).quantize(Decimal(1),rounding=ROUND_CEILING))
        extra = slip-paid['slippage']
        paid.update(slippage=slip,total_cost=paid['total_cost']+extra,cash_change=money(paid['cash_change']-extra))
        return paid

    def _affordable(self, qty, step, price, cash, sid):
        lo, hi = 0, qty//step
        while lo < hi:
            mid = (lo+hi+1)//2
            if -self._costs(price,int(mid*step),'buy',sid)['cash_change'] <= cash+1e-7:
                lo=mid
            else:
                hi=mid-1
        return int(lo*step)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        """A pre-sized order may fill partly. No failed remainder is hidden."""
        if type(qty) is not int or qty < 0:
            raise ValueError('Order shares must be integer and nonnegative')
        if qty == 0:
            return 0
        if side == 'sell':
            qty = min(qty, self.holdings.get(sid, {}).get('qty', 0))
        if sid != '0050' and side == 'sell' and reason == 'scheduled_exit' and hasattr(self,'exit_states'):
            state=self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:
                raise ValueError('Stock sale requires a latched exit')
            reason,signal_date=state['trigger_reason'],state['signal_date']
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
                    ref = odd['odd_ask' if side == 'buy' else 'odd_bid'] if self.stress_quote else odd['odd_last']
                    row.update(odd_volume=odd['odd_shares'], odd_bid=odd['odd_bid'], odd_ask=odd['odd_ask'],
                               odd_bid_qty=odd.get('bid_qty'), odd_ask_qty=odd.get('ask_qty'))
                    if (side == 'buy' and ref >= limits['upper']-1e-8) or (side == 'sell' and ref <= limits['lower']+1e-8):
                        row['failure'] = 'odd_lot_at_price_limit'
                    else:
                        capacity = math.floor(odd['odd_shares']*ODD_PARTICIPATION)
                        if self.stress_depth:
                            opposing = odd.get('ask_qty' if side == 'buy' else 'bid_qty')
                            if opposing is None or not math.isfinite(opposing) or opposing < 0:
                                capacity = 0
                                row['failure'] = 'missing_opposing_depth'
                            else:
                                capacity = min(capacity,math.floor(opposing))
                        cap = max(0,capacity-self.used[(sid,channel)])
                        row.update(capacity_qty=capacity, participation_limit=ODD_PARTICIPATION)
            if not row['failure']:
                fill = min(request,cap)//step*step
                if side == 'buy':
                    fill = self._affordable(fill,step,ref,self.cash,sid)
                if fill:
                    paid = self._costs(ref,fill,side,sid)
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

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        """Allow an otherwise executable residual sale to pay more fee than gross.

        All original price, limit, liquidity and channel checks run first. A
        fee larger than the residual market value does not justify trapping a
        cohort indefinitely when the account can pay the shortfall. Only that
        one original rejection is reconsidered, with no fee waiver or funding.
        """
        first = len(self.orders)
        filled_total = self._execute_order(day, sid, side, qty, reason, event_id, signal_date)
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
            paid = self._costs(reference, int(filled), 'sell', sid)
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

    def run(self):
        account=super().run()
        account['settings']['slippage']=self.stress_slippage
        return account


class StressReplay(StressOrder, TechnicalReplay):
    def __init__(self,*args,stress_mode='control',**kwargs):
        super().__init__(*args,mode='control',**kwargs)
        self._setup_stress(stress_mode)
        self.delayed_exits=set()
        if stress_mode in ('entry_delay','combined'):
            shifted=defaultdict(list)
            for day,entries in self.events.items():
                index=self.positions[day]+1
                if index >= len(self.days):
                    raise ValueError('No market day for delayed entry')
                execution=self.days[index]
                for original in entries:
                    event=deepcopy(original)
                    event['original_entry_date']=event['entry_date']
                    event['entry_date']=str(execution.date())
                    shifted[execution].append(event)
            self.events=shifted

    def corporate_day(self,day):
        income=super().corporate_day(day)
        if self.stress_mode in ('exit_delay','combined'):
            for identity,state in self.exit_states.items():
                if state['target_index'] is not None and identity not in self.delayed_exits:
                    state['original_target_date']=state['target_date']
                    state['target_index']+=1
                    i=state['target_index']
                    state['target_date']=str(self.days[i].date()) if i<len(self.days) else None
                    self.delayed_exits.add(identity)
                holding=self.holdings.get(state['stock_id'])
                if holding and holding['event_id']==identity and state['target_index'] is not None:
                    holding['due_index']=state['target_index']
        return income


class StressBenchmark(StressOrder, Replay):
    def __init__(self,*args,stress_mode='control',**kwargs):
        super().__init__(*args,benchmark=True,**kwargs)
        self._setup_stress(stress_mode)


def audit_stress(account):
    """Independently rebuild fees, daily cash and integer units from journal rows."""
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Non-finite or invalid account number')
        return value

    def same(a, b, label, tolerance=.011):
        if not math.isclose(number(a), number(b), abs_tol=tolerance, rel_tol=0):
            raise ValueError(label)

    def integer(value, label, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError(label)
        return value

    daily = account['daily']
    dates = [row['date'] for row in daily]
    if not dates or dates != sorted(set(dates)):
        raise ValueError('Daily dates must be unique and increasing')
    allowed = set(dates)
    for day in dates:
        if pd.Timestamp(day).strftime('%Y-%m-%d') != day:
            raise ValueError('Daily dates must be ISO dates')
    initial = number(account['settings']['initial_cash'])
    if initial <= 0:
        raise ValueError('Initial account cash must be positive')
    journals = {}
    for name in ('cash_ledger', 'trades', 'holdings', 'corporate_actions'):
        rows = account[name]
        if [row['date'] for row in rows] != sorted(row['date'] for row in rows):
            raise ValueError('Journal is not chronological: '+name)
        grouped = defaultdict(list)
        for row in rows:
            if row['date'] not in allowed:
                raise ValueError('Journal date outside account: '+name)
            grouped[row['date']].append(row)
        journals[name] = grouped
    cash, previous, peak = 0., initial, initial
    units = defaultdict(int)
    pending_shares = {}
    trade_sequence = 0
    initial_seen = False
    for row in daily:
        day = row['date']
        same(row['opening_nav'], previous, 'Opening NAV is not previous closing NAV')
        ledger_trades = []
        for movement in journals['cash_ledger'][day]:
            if movement['kind'] == 'initial_deposit':
                if initial_seen or day != dates[0] or cash != 0:
                    raise ValueError('Invalid initial cash deposit')
                same(movement['cash_change'], initial, 'Initial deposit differs from settings')
                initial_seen = True
            cash = float((Decimal(str(cash))+Decimal(str(number(movement['cash_change'])))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
            same(cash, movement['cash_after'], 'Cash ledger does not reconcile')
            if cash < 0:
                raise ValueError('Cash ledger overdraft')
            if movement['kind'] in ('buy', 'sell'):
                ledger_trades.append(movement)
        if not initial_seen:
            raise ValueError('Initial deposit missing')
        same(cash, row['cash'], 'Daily cash differs from ledger')
        # Entitlements and deliveries occur before this day's trading.
        for action in journals['corporate_actions'][day]:
            sid, kind = action['stock_id'], action['kind']
            if 'entitled_qty' in action:
                integer(action['entitled_qty'], 'Invalid entitled shares', 1)
                if action['entitled_qty'] != units[sid]:
                    raise ValueError('Corporate entitlement differs from opening shares')
            if kind == 'split':
                multiplier = number(action['multiplier'])
                transformed = Decimal(units[sid])*Decimal(str(multiplier))
                if multiplier <= 0 or transformed != transformed.to_integral_value():
                    raise ValueError('Split does not produce exact integer shares')
                units[sid] = int(transformed)
                if integer(action['qty_after'], 'Invalid split shares') != units[sid]:
                    raise ValueError('Split share journal disagrees')
            elif kind == 'stock_dividend':
                key = (sid, action['action_id'])
                if key in pending_shares:
                    raise ValueError('Duplicate stock entitlement')
                quantity = Decimal(units[sid])*Decimal(str(number(action['shares_per_share'])))
                whole = int(quantity.to_integral_value(rounding=ROUND_FLOOR))
                if integer(action['whole_new_shares'], 'Invalid stock entitlement') != whole:
                    raise ValueError('Stock entitlement quantity disagrees')
                same(action['fractional_right'], float(quantity-whole), 'Fractional entitlement disagrees', 1e-10)
                pending_shares[key] = (whole, action['event_id'])
            elif kind == 'share_delivery':
                key = (sid, action['action_id'])
                expected = pending_shares.pop(key, None)
                if expected != (integer(action['qty'], 'Invalid delivered shares'), action['event_id']):
                    raise ValueError('Share delivery differs from locked entitlement')
                units[sid] += action['qty']
        trades = journals['trades'][day]
        if len(trades) != len(ledger_trades):
            raise ValueError('Trade and cash journals have different lengths')
        participation, daily_cost = defaultdict(int), 0.
        for trade, movement in zip(trades, ledger_trades):
            sid, side, channel = trade['stock_id'], trade['side'], trade['channel']
            if side not in ('buy', 'sell') or channel not in ('board', 'odd'):
                raise ValueError('Invalid trade side/channel')
            qty = integer(trade['qty'], 'Trade must contain positive integer shares', 1)
            if channel == 'board' and qty % 1000:
                raise ValueError('Board trade is not a full lot')
            if channel == 'odd' and qty >= 1000:
                raise ValueError('Odd trade exceeds one lot')
            trade_sequence += 1
            if trade['sequence'] != trade_sequence:
                raise ValueError('Trade sequence is inconsistent')
            price = number(trade['reference_price'])
            if price <= 0:
                raise ValueError('Trade price must be positive')
            raw_gross = Decimal(str(price))*qty
            gross = raw_gross.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
            fee = max(Decimal(20), (raw_gross*Decimal('.001425')).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            tax = ((raw_gross*Decimal('.001' if sid == '0050' else '.003')).quantize(Decimal(1), rounding=ROUND_FLOOR) if side == 'sell' else Decimal(0))
            slip = (raw_gross*Decimal(str(account['settings']['slippage']))).quantize(Decimal(1), rounding=ROUND_CEILING)
            expected_cost = float(fee+tax+slip)
            expected_cash = float((gross if side == 'sell' else -gross)-fee-tax-slip)
            for field, expected in [('gross',float(gross)),('commission',float(fee)),('tax',float(tax)),('slippage',float(slip)),('total_cost',expected_cost),('cash_change',expected_cash)]:
                same(trade[field], expected, 'Trade fee/cash calculation differs: '+field)
            if any(trade[key] != movement[key] for key in ('stock_id','event_id','channel')) or movement['kind'] != side:
                raise ValueError('Trade and cash journal identity disagree')
            same(expected_cash, movement['cash_change'], 'Trade cash movement differs')
            same(trade['cash_after'], movement['cash_after'], 'Trade cash balance differs')
            participation[(sid, channel)] += qty
            if qty > integer(trade['requested_qty'], 'Invalid requested shares', 1) or participation[(sid,channel)] > integer(trade['capacity_qty'], 'Invalid capacity', 1):
                raise ValueError('Trade exceeds order/capacity')
            if side == 'buy' and sid != '0050' and number(trade['prior_avg_amount20']) < 50e6:
                raise ValueError('Entry violates liquidity filter')
            units[sid] += qty if side == 'buy' else -qty
            if units[sid] < 0 or units[sid] != integer(trade['remaining_shares'], 'Invalid remaining shares'):
                raise ValueError('Trade shares do not reconcile')
            daily_cost += expected_cost
        actual, assets = {}, 0.
        for holding in journals['holdings'][day]:
            sid = holding['stock_id']
            if sid in actual:
                raise ValueError('Duplicate daily holding')
            actual[sid] = integer(holding['qty'], 'Holding must contain positive integer shares', 1)
            price = number(holding['price'])
            if price <= 0 or holding['mark_date'] > day:
                raise ValueError('Invalid holding valuation')
            value = actual[sid]*price
            same(value, holding['market_value'], 'Holding value differs', .06)
            assets += value
        if actual != {sid:qty for sid,qty in units.items() if qty}:
            raise ValueError('Daily holdings differ from trades/splits/deliveries')
        same(daily_cost, row['cost'], 'Daily cost differs from trades')
        same(assets, row['market_value'], 'Daily market value differs', .06)
        same(row['nav'], cash+assets+number(row['receivable']), 'Daily balance sheet does not reconcile', .06)
        same(row['nav'], row['opening_nav']+row['market_pnl']+row['dividend_entitlement']+row['execution_basis_pnl']-row['cost'], 'Daily P&L does not reconcile', .06)
        same(row['daily_return'], row['nav']/previous-1, 'Daily return differs', 1e-10)
        same(row['total_return'], row['nav']/initial-1, 'Total return differs', 1e-10)
        peak = max(peak,row['nav'])
        same(row['drawdown'], row['nav']/peak-1, 'Drawdown differs', 1e-10)
        previous = row['nav']
    return dict(cash_ledger_reconciled=True, all_daily_nav_reconciled=True,
        integer_shares=True, no_overdraft=True, capacity_and_order_limits=True,
        fees_recomputed=True, trade_cash_reconciled=True, daily_shares_reconstructed=True,
        opening_nav_continuity=True, trading_days=len(daily), cash_movements=len(account['cash_ledger']))
