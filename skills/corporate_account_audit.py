"""Corporate-capable journal audit v2.

The 2026-09-27 stress auditor is sealed into past publications. Its cash, trade,
fee and daily-unit reconstruction is retained here verbatim except for the new
capital-reduction branch and independent cash entitlement verification. The old
file and old reports remain reproducible. This module is separately hash-bound.
"""
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math
import pandas as pd


def audit_capital_cash(account):
    for action in account['corporate_actions']:
        if action['kind'] != 'capital_reduction':
            continue
        for suffix, amount, pay in (
                ('-capital-cash', action['capital_cash_amount'], action['pay_date']),
                ('-fractional-cash', action['fractional_gross_amount'], None)):
            key = action['action_id'] + suffix
            payments = [r for r in account['cash_ledger'] if r.get('action_id') == key]
            pending = [r for r in account['receivables'] if r.get('action_id') == key]
            due = next((r['date'] for r in account['daily'] if pay and r['date'] >= pay), None)
            if amount <= 0:
                if payments or pending:
                    raise ValueError('Zero capital entitlement unexpectedly paid')
            elif due:
                if (len(payments) != 1 or pending or payments[0]['date'] != due
                        or payments[0]['stock_id'] != action['stock_id']
                        or payments[0]['cash_change'] != amount):
                    raise ValueError('Capital cash payment differs from entitlement')
            elif (payments or len(pending) != 1 or pending[0]['amount'] != amount
                    or pending[0]['pay_date'] != pay or pending[0]['stock_id'] != action['stock_id']):
                raise ValueError('Unpaid capital entitlement is missing or paid early')


def audit_corporate_account(account):
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
            if kind == 'capital_reduction':
                ratio = number(action['multiplier'])
                if not 0 < ratio < 1:
                    raise ValueError('Invalid capital reduction ratio')
                transformed = Decimal(units[sid]) * Decimal(str(ratio))
                whole = int(transformed.to_integral_value(rounding=ROUND_FLOOR))
                fraction = transformed - whole
                rate = number(action['cash_per_share'])
                if rate < 0:
                    raise ValueError('Invalid capital return rate')
                refund = float((Decimal(units[sid])*Decimal(str(rate))).to_integral_value(rounding=ROUND_FLOOR))
                same(action['capital_cash_amount'], refund, 'Capital return cash disagrees', 0)
                same(action['fractional_right'], float(fraction), 'Capital fraction disagrees', 1e-10)
                price = number(action['fractional_reference_price'])
                if price <= 0 or not action['fractional_reference_date'] < day:
                    raise ValueError('Invalid capital fraction reference')
                fraction_cash = float((fraction*Decimal(str(price))).to_integral_value(rounding=ROUND_FLOOR))
                same(action['fractional_gross_amount'], fraction_cash, 'Capital fraction cash disagrees', 0)
                if integer(action['qty_after'], 'Invalid exchanged shares') != whole:
                    raise ValueError('Capital exchange share journal disagrees')
                units[sid] = whole
            elif kind == 'split':
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
    audit_capital_cash(account)
    return dict(capital_reductions_reconciled=True, cash_ledger_reconciled=True, all_daily_nav_reconciled=True,
        integer_shares=True, no_overdraft=True, capacity_and_order_limits=True,
        fees_recomputed=True, trade_cash_reconciled=True, daily_shares_reconstructed=True,
        opening_nav_continuity=True, trading_days=len(daily), cash_movements=len(account['cash_ledger']))
