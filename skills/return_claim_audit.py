"""Reconcile a fixed HL2 stock account without importing its replay engine.

This verifies recorded simulations, not actual execution or an unseen edge.
Decimal HL2 costs are compared separately with the historical binary-float
price convention; precision differences remain visible and are never waived.
"""
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math


D = Decimal
CENT = D('.01')
ZERO = D(0)


def number(value):
    result = D(str(value))
    if not result.is_finite():
        raise ValueError('Non-finite audit input')
    return result


def money(value):
    return number(value).quantize(CENT, rounding=ROUND_HALF_UP)


def same(actual, expected, label, tolerance=D('.00001')):
    if abs(number(actual) - number(expected)) > tolerance:
        raise ValueError(f'{label}: recorded={actual}, rebuilt={expected}')


def require(condition, label):
    if not condition:
        raise ValueError(label)


def trade_costs(high, low, qty, side, *, legacy_float=False):
    require(type(qty) is int and qty > 0, 'Positive integer quantity required')
    require(side in ('buy', 'sell'), 'Unknown trade side')
    hi, lo = number(high), number(low)
    require(hi >= lo > 0, 'Invalid source price envelope')
    price = (hi + lo) / 2
    if legacy_float:
        price = number((float(high) + float(low)) / 2)
    gross = price * qty
    fee = max(D(20), (gross * D('.001425')).quantize(D(1), rounding=ROUND_HALF_UP))
    slip = (gross * D('.0045')).quantize(D(1), rounding=ROUND_CEILING)
    tax = (gross * D('.003')).quantize(D(1), rounding=ROUND_FLOOR) if side == 'sell' else ZERO
    cost = fee + slip + tax
    gross = money(gross)
    return dict(gross=gross, commission=fee, slippage=slip, tax=tax,
                total_cost=cost, cash_change=(gross if side == 'sell' else -gross) - cost)


def audit_account(case, marks, execution_quotes, *, fractional_rounding):
    """Rebuild cash, units, rights and every NAV from independent price inputs.

    marks[(date, sid)] is (raw close, actual quote date). execution_quotes uses
    (date, sid, channel) -> (high, low, volume, prior20 volume). Unsupported
    corporate events fail explicitly. No cached NAV/holding value is an input
    to the reconstruction. Recorded cash is checked in chronological order.
    """
    a, summary = case['account'], case['summary']
    settings = a['settings']
    require(settings['price_formula'] == '(high+low)/2', 'Only sealed HL2 accounts supported')
    require(not settings['benchmark'], 'This audit is for individual stock accounts')
    initial = number(settings['initial_cash'])
    daily = a['daily']
    dates = [r['date'] for r in daily]
    require(dates == sorted(set(dates)) and dates, 'Unique chronological trading days required')
    day_set = set(dates)
    groups = {}
    for key in ('trades', 'corporate_actions', 'holdings'):
        group = defaultdict(list)
        for r in a[key]:
            require(r['date'] in day_set, 'Record outside account dates: ' + key)
            group[r['date']].append(r)
        groups[key] = group
    require([t['sequence'] for t in a['trades']] == list(range(1, len(a['trades'])+1)),
            'Trade sequence duplicated, missing or unordered')
    require([t['date'] for t in a['trades']] == sorted(t['date'] for t in a['trades']),
            'Trade dates out of order')
    cohorts = {r['event_id']: r for r in a['cohorts']}
    require(len(cohorts) == len(a['cohorts']), 'Duplicate cohort')
    ledger = a['cash_ledger']
    cursor, cash, precise_delta = 0, ZERO, ZERO
    units, rights, attribution = {}, {}, defaultdict(lambda: ZERO)
    consumed, unique_fills = set(), set()
    fee_variances, daily_evidence, corporate_evidence = [], [], []
    costs = {k: ZERO for k in ('commission', 'slippage', 'tax', 'total_cost')}
    min_cash, maximum_precision_delta = initial, ZERO
    previous, peak, mdd, compounded = initial, initial, ZERO, D(1)
    annual, year_start = {}, initial
    last_year = None
    holding_count = stale_count = 0

    def cash_move(day, kind, value, **identity):
        nonlocal cursor, cash, min_cash
        require(cursor < len(ledger), 'Missing cash movement')
        r = ledger[cursor]
        require(r['date'] == day and r['kind'] == kind, 'Cash chronology or type differs')
        for k, v in identity.items():
            require(r.get(k) == v, 'Cash identity differs: ' + k)
        same(r['cash_change'], value, 'Cash movement')
        cash = money(cash + value)
        same(r['cash_after'], cash, 'Cash running balance')
        require(cash >= 0, 'Account overdraft')
        min_cash = min(min_cash, cash)
        cursor += 1

    def mark(day, sid):
        price, mark_day = marks[(day, sid)]
        require(mark_day <= day and number(price) > 0, 'Future or invalid raw mark')
        return number(price), mark_day

    cash_move(dates[0], 'initial_deposit', initial)
    for row in daily:
        day = row['date']
        for r in groups['corporate_actions'][day]:
            kind, sid, eid, aid = (r[k] for k in ('kind', 'stock_id', 'event_id', 'action_id'))
            require(eid in cohorts and cohorts[eid]['stock_id'] == sid, 'Corporate cohort differs')
            qty = units.get(eid, 0)
            if kind in ('cash_dividend', 'stock_dividend', 'split', 'waive_subscription'):
                require((kind, aid) not in consumed, 'Duplicate corporate entitlement')
                consumed.add((kind, aid))
                require(qty == r['entitled_qty'] and qty > 0, 'Corporate opening shares differ')
            if kind in ('cash_dividend', 'stock_dividend'):
                require(aid not in rights and r['pay_date'] >= day, 'Duplicate or backward payable')
                require(r.get('announcement_date', day) <= day, 'Future dividend announcement')
                right = dict(stock_id=sid, event_id=eid, pay_date=r['pay_date'], kind=kind)
                if kind == 'cash_dividend':
                    amount = number(qty) * number(r['cash_per_share'])
                    rounding = r.get('cash_rounding', 'half_up_cents')
                    require(rounding in ('floor_ntd', 'half_up_cents'), 'Unknown cash rounding')
                    amount = amount.quantize(D(1), rounding=ROUND_FLOOR) if rounding == 'floor_ntd' else money(amount)
                    same(r['entitlement_value'], amount, 'Dividend entitlement')
                    right['amount'] = amount
                else:
                    total = number(qty) * number(r['shares_per_share'])
                    whole = int(total.to_integral_value(rounding=ROUND_FLOOR))
                    fraction = total - whole
                    require(whole == r['whole_new_shares'], 'Stock dividend shares differ')
                    same(r['fractional_right'], fraction, 'Fractional shares')
                    fractional = fraction * number(r['fractional_cash_per_share'])
                    policy = fractional_rounding[aid]
                    require(policy in ('floor_ntd', 'half_up_cents'), 'Unknown fractional rounding')
                    if policy == 'floor_ntd':
                        fractional = fractional.to_integral_value(rounding=ROUND_FLOOR)
                    right.update(qty=whole, fraction=fraction, fractional=fractional)
                rights[aid] = right
            elif kind in ('payment', 'share_delivery'):
                require(aid in rights, 'Payment without an unpaid entitlement')
                right = rights.pop(aid)
                require(right['stock_id'] == sid and right['event_id'] == eid, 'Payee differs')
                due = next((d for d in dates if d >= right['pay_date']), None)
                require(day == due, 'Payment is not on first session on or after due date')
                if kind == 'payment':
                    require(right['kind'] == 'cash_dividend', 'Wrong payment type')
                    amount = right['amount']
                    same(r['amount'], amount, 'Dividend payment')
                    cash_move(day, 'dividend_payment', amount, stock_id=sid, action_id=aid)
                else:
                    require(right['kind'] == 'stock_dividend' and r['qty'] == right['qty'],
                            'Share delivery differs')
                    same(r['fraction'], right['fraction'], 'Delivered fraction')
                    if right['qty']:
                        units[eid] = qty + right['qty']
                    amount = money(right['fractional'])
                    cash_move(day, 'fractional_share_payment', amount, stock_id=sid)
                attribution[eid] += amount
            elif kind == 'split':
                new_qty = number(qty) * number(r['multiplier'])
                require(new_qty == int(new_qty) and new_qty == r['qty_after'], 'Split shares differ')
                units[eid] = int(new_qty)
                corporate_evidence.append(dict(date=day, stock_id=sid, old_qty=qty,
                    new_qty=int(new_qty), multiplier=r['multiplier']))
            elif kind != 'waive_subscription':
                raise ValueError('Unsupported corporate action: ' + kind)

        day_cost = ZERO
        for t in groups['trades'][day]:
            sid, eid, side, channel, qty = (t[k] for k in ('stock_id', 'event_id', 'side', 'channel', 'qty'))
            require(eid in cohorts and cohorts[eid]['stock_id'] == sid, 'Trade cohort differs')
            require(len(sid) == 4 and sid.isdigit() and not sid.startswith('0'), 'Individual stocks only')
            require(t['signal_date'] < day, 'Trade uses same-day or future signal')
            key = day, sid, channel
            require(key not in unique_fills, 'Daily channel capacity reused')
            unique_fills.add(key)
            hi, lo, volume, prior_volume = execution_quotes[key]
            for name, value in (('source_high', hi), ('source_low', lo), ('source_volume', volume)):
                same(t[name], value, name)
            same(t['reference_price'], (number(hi)+number(lo))/2, 'HL2 reference')
            require(number(t['participation_limit']) == D('.01'), 'Participation setting differs')
            cap = int((min(number(volume), number(prior_volume)) if channel == 'board'
                       else number(volume)) * D('.01'))
            if channel == 'board':
                cap = cap//1000*1000
                require(qty % 1000 == 0, 'Board fill not whole lots')
            else:
                require(channel == 'odd' and qty < 1000, 'Invalid odd-lot fill')
            require(0 < qty <= cap and cap == t['capacity_qty'], 'Source capacity differs')
            legacy = trade_costs(hi, lo, qty, side, legacy_float=True)
            exact = trade_costs(hi, lo, qty, side)
            for name, value in legacy.items():
                same(t[name], value, 'Trade '+name)
            for name in ('gross', 'commission', 'slippage', 'tax'):
                if exact[name] != legacy[name]:
                    fee_variances.append(dict(sequence=t['sequence'], date=day, stock_id=sid,
                        field=name, recorded=float(legacy[name]), exact=float(exact[name])))
            precise_delta += exact['cash_change'] - legacy['cash_change']
            maximum_precision_delta = max(maximum_precision_delta, abs(precise_delta))
            cash_move(day, side, legacy['cash_change'], stock_id=sid, event_id=eid, channel=channel)
            same(t['cash_after'], cash, 'Trade cash after')
            units[eid] = units.get(eid, 0) + (qty if side == 'buy' else -qty)
            require(units[eid] >= 0 and units[eid] == t['remaining_shares'], 'Oversold or wrong shares')
            attribution[eid] += legacy['cash_change']
            for name in costs:
                costs[name] += legacy[name]
            day_cost += legacy['total_cost']

        holdings = {h['event_id']: h for h in groups['holdings'][day]}
        require(len(holdings) == len(groups['holdings'][day]), 'Duplicate daily holding')
        require(set(holdings) == {eid for eid, qty in units.items() if qty > 0}, 'Daily holdings differ')
        mv, receivable, day_stale = ZERO, ZERO, 0
        for eid, h in holdings.items():
            sid = cohorts[eid]['stock_id']
            price, mark_day = mark(day, sid)
            value = price * units[eid]
            require(h['stock_id'] == sid and h['qty'] == units[eid], 'Holding identity/quantity')
            require(h['mark_date'] == mark_day and h['stale'] == (mark_day < day), 'Mark date/staleness')
            same(h['price'], price, 'Raw holding price')
            same(h['market_value'], value, 'Raw holding value')
            mv += value
            day_stale += int(mark_day < day)
            holding_count += 1
        for right in rights.values():
            if right['kind'] == 'cash_dividend':
                receivable += right['amount']
            else:
                receivable += right['qty'] * mark(day, right['stock_id'])[0] + right['fractional']
        nav = money(cash + mv + receivable)
        ret = nav / previous - 1
        peak = max(peak, nav)
        dd = nav / peak - 1
        mdd = min(mdd, dd)
        for name, value in (('cash', cash), ('market_value', mv), ('receivable', money(receivable)),
                            ('nav', nav), ('opening_nav', previous), ('cost', day_cost)):
            same(row[name], value, 'Daily '+day+' '+name)
        for name, value in (('daily_return', ret), ('total_return', nav/initial-1), ('drawdown', dd)):
            same(row[name], value, 'Daily '+name, D('1e-10'))
        require(row['stale_holdings'] == day_stale and row['holdings'] == len(holdings), 'Daily holding count')
        stale_count += day_stale
        compounded *= 1 + ret
        year = day[:4]
        if year != last_year:
            year_start, last_year = previous, year
        annual[year] = dict(year=year, start_nav=float(year_start), end_nav=float(nav),
                            total_return=float(nav/year_start-1))
        daily_evidence.append(dict(date=day, cash=float(cash), market_value=float(mv),
            receivable=float(money(receivable)), nav=float(nav),
            exact_decimal_fixed_fills_nav=float(nav+precise_delta), precision_delta=float(precise_delta)))
        previous = nav
    require(cursor == len(ledger), 'Unexplained cash movement or external deposit')
    require(not rights and not a['receivables'], 'Pending rights require additional final attribution')
    same(summary['final_nav'], previous, 'Final NAV')
    same(summary['total_return'], previous/initial-1, 'Total return', D('1e-10'))
    same(summary['max_drawdown'], mdd, 'Maximum drawdown', D('1e-10'))
    same(compounded, previous/initial, 'Compounded return identity', D('1e-10'))
    for name, value in costs.items():
        same(summary['costs'][name], value, 'Summary cost '+name)
    published_annual = {r['year']: r for r in summary['annual']}
    for year, r in annual.items():
        for field in ('start_nav', 'end_nav', 'total_return'):
            same(published_annual[year][field], r[field], 'Annual '+field)
    annual_product = math.prod(1+r['total_return'] for r in annual.values())
    same(annual_product, previous/initial, 'Annual compound identity', D('1e-10'))
    for eid, qty in units.items():
        if qty:
            attribution[eid] += qty * mark(dates[-1], cohorts[eid]['stock_id'])[0]
    same(sum(attribution.values()), previous-initial, 'Sum of cohort net PnL')
    ranked = sorted(attribution, key=attribution.get, reverse=True)
    top = [dict(event_id=eid, stock_id=cohorts[eid]['stock_id'], name=cohorts[eid]['name'],
                entry_date=cohorts[eid]['entry_date'], net_pnl=float(attribution[eid])) for eid in ranked[:5]]
    return dict(ledger_reconciled=True, strict_decimal_costs_match=not fee_variances,
        initial_cash=float(initial), start=dates[0], end=dates[-1], trading_days=len(dates),
        trade_count=len(a['trades']), cash_movements=len(ledger), holdings_checked=holding_count,
        corporate_actions_checked=len(a['corporate_actions']), stale_holding_days=stale_count,
        final_cash=float(cash), final_market_value=float(mv), final_receivable=float(receivable),
        final_nav=float(previous), total_return=float(previous/initial-1), max_drawdown=float(mdd),
        costs={k:float(v) for k,v in costs.items()}, monetary_precision_findings=fee_variances,
        exact_decimal_fixed_fills_final_nav=float(previous+precise_delta),
        maximum_fixed_fills_precision_delta=float(maximum_precision_delta), minimum_cash=float(min_cash),
        annual=list(annual.values()), daily=daily_evidence, splits=corporate_evidence,
        top_five_net_profit=top, top_five_share_of_total_profit=float(sum(attribution[e] for e in ranked[:5])/(previous-initial)),
        actual_fill_verified=False, unseen_validation=False, live_qualified=False)
