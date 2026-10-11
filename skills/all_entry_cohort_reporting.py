"""Attribute explicitly valued pending shares without changing cash or settlement."""
from decimal import Decimal
import math
from pathlib import Path

import pandas as pd

from scripts.summarize_strategy_account_comparison import describe_cohorts as describe_cash_cohorts


def describe_cohorts(case, report, sources):
    rights = case['summary']['final_receivables']
    if rights != case['account']['receivables']:
        raise ValueError('Final receivable journal differs')
    shares = [r for r in rights if r['kind'] != 'cash']
    if not shares:
        return describe_cash_cohorts(case)
    if any(r['kind'] != 'shares' for r in shares):
        raise ValueError('Unsupported non-cash entitlement')
    path = str(Path(report['input_bundle'])/'raw-close.parquet')
    expected = report['source_sha256'].get(path)
    if not expected:
        raise ValueError('Pending share close source is not bound')
    quotes = pd.read_parquet(sources.bind(path, expected), columns=['date']+sorted({r['stock_id'] for r in shares}))
    end = case['summary']['end']
    quotes = quotes.loc[quotes.date.astype(str).eq(end)]
    if len(quotes) != 1:
        raise ValueError('Pending shares require one exact endpoint close')
    values, audit = {}, []
    for right in shares:
        aid, sid, eid = right['action_id'], right['stock_id'], right['event_id']
        actions = [r for r in case['account']['corporate_actions']
                   if r['action_id'] == aid and r['kind'] == 'stock_dividend']
        if len(actions) != 1 or aid in values:
            raise ValueError('Pending share entitlement is absent or duplicated')
        action = actions[0]
        qty = right['qty']; close = float(quotes.iloc[0][sid])
        if (type(qty) is not int or qty <= 0 or not math.isfinite(close) or close <= 0
                or (action['stock_id'],action['event_id'],action['date'],action['pay_date'])
                    != (sid,eid,right['ex_date'],right.get('pay_date'))
                or not right['ex_date'] <= end < (right.get('pay_date') or '')
                or qty != action['whole_new_shares']
                or qty != int(Decimal(action['entitled_qty'])*Decimal(str(action['shares_per_share'])))
                or right.get('fractional_cash_per_share') != 0
                or any(r['action_id'] == aid and r['kind'] == 'share_delivery'
                       for r in case['account']['corporate_actions'])):
            raise ValueError('Pending share units, dates or separate fractions differ')
        values[aid] = qty*close
        audit.append(dict(action_id=aid,event_id=eid,stock_id=sid,qty=qty,price=close,
            price_date=end,value=values[aid],pay_date=right['pay_date'],
            valuation_basis='ordinary_share_close_proxy',source_path=path,source_sha256=expected,
            spendable_cash=0,settled=False))
    # The cash-only attribution helper accepts monetary receivable valuations.
    # Use a private view after validating the non-cash units and endpoint quote;
    # retain the source ledger, cash balance and unsettled classification intact.
    valued = [dict(r,kind='cash',amount=values[r['action_id']]) if r['kind']=='shares' else r for r in rights]
    view = dict(case,account=dict(case['account'],receivables=valued),
                summary=dict(case['summary'],final_receivables=valued))
    result = describe_cash_cohorts(view)
    result['noncash_rights_valuation'] = audit
    return result
