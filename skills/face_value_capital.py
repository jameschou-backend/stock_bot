"""Explicit loss-reduction settlement for the five-slot follow-up only.

Unverified fractional payment stays a receivable and cannot finance purchases.
All other actions retain the sealed CapitalReturnActions behavior.
"""
from decimal import Decimal, ROUND_FLOOR
import math
from copy import deepcopy
from unittest.mock import patch
import pandas as pd
from skills.million_replay import UnresolvedAction
from skills.pending_share_entitlements import _date

class FaceValueCapitalActions:
    """Exchange shares and recognize two independent cash rights on old units.

    Fractional rights use the officially specified face value. Their gross
    value remains an unavailable receivable while personal net fees/payment are
    unknown. Ordinary capital refunds use their own evidenced payment date.
    """
    def __init__(self, provider, account):
        self.provider, self.account = provider, account
        self.processed = set()
        from skills.execution_factorial import CapitalReturnActions
        self.other_actions = CapitalReturnActions(provider)

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def on_date(self, sid, day):
        rows = self.provider.on_date(sid, day)
        capital = [r for r in rows if r['kind'] == 'capital_reduction' and r.get('fractional_policy') == 'face_value_gross_receivable']
        if not capital:
            return self.other_actions.on_date(sid, day)
        if len(rows) != 1:
            raise UnresolvedAction('Simultaneous capital actions require separate entitlement review')
        action = capital[0]
        if (action['action_id'] in self.processed or action.get('cash_rounding') != 'floor_ntd'
                or action.get('fractional_policy') != 'face_value_gross_receivable'
                or action.get('fractional_cash_rounding') != 'floor_ntd'
                or action.get('fractional_cash_pay_date') is not None
                or not action.get('evidence_files')
                or not _date(action.get('known_date')) < day <= _date(action.get('pay_date'))
                or not _date(action.get('fractional_reference_date')) < day):
            raise UnresolvedAction(f'Invalid capital settlement terms: {sid} {day}')
        for key in ('multiplier', 'cash_per_share'):
            value = action.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise UnresolvedAction(f'Invalid capital settlement amount: {sid} {day}')
        if not 0 < action['multiplier'] < 1:
            raise UnresolvedAction('Capital reduction ratio must be between zero and one')
        stamp = pd.Timestamp(day)
        price = action.get('fractional_face_value')
        if type(price) not in (int, float) or not math.isfinite(price) or price <= 0 or self.account.raw(stamp, sid) is None:
            raise UnresolvedAction(f'Capital settlement quote missing: {sid} {day}')
        holding = self.account.holdings[sid]
        old = holding['qty']
        if type(old) is not int or old <= 0:
            raise UnresolvedAction('Capital settlement requires positive integer holdings')
        new = Decimal(old)*Decimal(str(action['multiplier']))
        whole = int(new.to_integral_value(rounding=ROUND_FLOOR))
        fraction = new-whole
        refund = float((Decimal(old)*Decimal(str(action['cash_per_share']))).to_integral_value(rounding=ROUND_FLOOR))
        fractional = float((fraction*Decimal(str(price))).to_integral_value(rounding=ROUND_FLOOR))
        for suffix, amount, payment, nature in (
                ('-capital-cash', refund, action['pay_date'], 'capital_return'),
                ('-fractional-cash', fractional, None, 'fractional_capital_gross')):
            if amount:
                self.account.receivables.append(dict(stock_id=sid, amount=amount, kind='cash',
                    pay_date=payment, action_id=action['action_id']+suffix, event_id=holding['event_id'],
                    ex_date=day, cash_flow_nature=nature, net_amount_verified=payment is not None))
        self.account.actions.append(dict(action, date=day, entitled_qty=old, event_id=holding['event_id'],
            qty_after=whole, capital_cash_amount=refund, fractional_right=float(fraction),
            fractional_reference_price=price, fractional_gross_amount=fractional,
            fractional_net_amount_verified=False, fractional_net_value_range=[0., fractional]))
        holding['qty'] = whole
        # Let the account's normal zero-unit cleanup close the cohort and slot;
        # removing the holding here would strand its entry bookkeeping.
        if sid in self.account.marks:
            self.account.marks[sid]['price'] = (self.account.marks[sid]['price']-action['cash_per_share'])/action['multiplier']
        self.processed.add(action['action_id'])
        return []


def audit_face_resources(account, plans, decisions, board, snapshots, quotes):
    """Audit real capital accounting before adapting the legacy slot-only view."""
    from skills.corporate_account_audit import audit_corporate_account
    from skills.high_return_audit import audit_high_return_resources
    from skills import execution_resources
    checked = audit_corporate_account(account)
    view = dict(account, corporate_actions=deepcopy(account['corporate_actions']))
    # Only the residual slot reconstruction expects exchanges to be named split.
    # Cash, units, multiplier/fraction arithmetic have already been independently
    # audited above on the original capital_reduction journal, never this view.
    for row in view['corporate_actions']:
        if row['kind'] == 'capital_reduction':
            row['kind'] = 'split'
    def audited(original):
        if original is not view:
            raise ValueError('Unexpected resource audit account')
        return dict(checked)
    with patch.object(execution_resources, 'audit_stress', audited):
        result = audit_high_return_resources(view, plans, decisions, board, snapshots, quotes)
    result['capital_resources_independently_reconciled'] = True
    return result
