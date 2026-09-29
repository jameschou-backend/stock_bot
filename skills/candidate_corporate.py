"""Explicit issuer cash-dividend supplements; never infer a missing payment."""
from copy import deepcopy
import math
import pandas as pd


def complete_cash_dividends(rows, sid, supplements):
    result = deepcopy(rows)
    for item in supplements:
        if item['stock_id'] != sid:
            continue
        ex, payment, amount = item['date'], item['pay_date'], item['cash_per_share']
        if (not math.isfinite(amount) or amount <= 0 or pd.Timestamp(payment) < pd.Timestamp(ex)
            or not item['source'] or pd.Timestamp(item['announcement_date']) > pd.Timestamp(ex)):
            raise ValueError('Invalid dated issuer cash supplement')
        existing = [r for r in result if r['date'] == ex and r['kind'] == 'cash_dividend']
        if existing:
            if len(existing) != 1 or existing[0]['cash_per_share'] != amount or existing[0]['pay_date'] != payment:
                raise ValueError('Issuer cash supplement conflicts with existing cash policy')
        else:
            unresolved = [r for r in result if r['date'] == ex and r['kind'] == 'unresolved_cash_dividend']
            if len(unresolved) != 1:
                raise ValueError('Cash supplement must match an explicit unresolved official event')
            result.append(dict(action_id=f'{sid}-cash-{ex}', kind='cash_dividend', **item))
        result = [r for r in result if not (r['date'] == ex and r['kind'] == 'unresolved_cash_dividend')]
    return sorted(result, key=lambda r: (r['date'], r['action_id']))
