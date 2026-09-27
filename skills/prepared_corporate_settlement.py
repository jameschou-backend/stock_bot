"""Explicit corporate settlement extensions for newly prepared accounts only."""
from datetime import timedelta
from decimal import Decimal, ROUND_FLOOR
import math
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from skills.account_source_preflight import digest
from skills.million_replay import UnresolvedAction
from skills.pending_share_entitlements import validate_pending_terms as original_pending_terms, _date


def validate_delivery_terms(terms, sid, day, end, source_root=None):
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    if terms.get('pending_not_before_basis') != 'dated_agent_pending_confirmation':
        return original_pending_terms(terms, sid, day, end, root)
    confirmed = _date(terms.get('pending_confirmed_through'))
    bound = _date(terms.get('pending_delivery_not_before'))
    if (terms.get('pending_only') is not True or terms.get('pay_date') is not None
            or terms.get('ordinary_share_available_date') is not None
            or terms.get('ordinary_share_delivery_status') != 'unannounced'
            or terms.get('use_scope') != 'account_settlement_only_not_selection'
            or _date(terms.get('pending_confirmation_date')) != confirmed
            or not day <= _date(terms.get('record_date')) <= confirmed
            or _date(terms.get('entitlement_announcement_date')) > day
            or bound != str((pd.Timestamp(confirmed)+timedelta(days=1)).date())
            or str(pd.Timestamp(end).date()) > confirmed):
        raise UnresolvedAction(f'Dated pending confirmation does not cover account: {sid} {day}')
    for field in ('pending_delivery_evidence', 'pending_confirmation_index_evidence'):
        proof = terms.get(field, {})
        name = proof.get('path', '')
        path = (root/name).resolve()
        if (not name or Path(name).is_absolute() or '..' in Path(name).parts
                or not path.is_relative_to(root) or (root/name).is_symlink() or not path.is_file()
                or not proof.get('url', '').startswith('https://')
                or urlparse(proof.get('url', '')).hostname != 'www.yuanta.com.tw'
                or digest(path) != proof.get('sha256') or name not in terms.get('evidence_files', [])):
            raise UnresolvedAction(f'Dated pending confirmation source missing or changed: {sid} {day}')
    return bound


class CapitalSettlementActions:
    """Exchange shares and recognize two independent cash rights on old units.

    Fractional rights use the officially specified historical close. Their gross
    value remains an unavailable receivable while personal net fees/payment are
    unknown. Ordinary capital refunds use their own evidenced payment date.
    """
    def __init__(self, provider, account):
        self.provider, self.account = provider, account
        self.processed = set()

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def on_date(self, sid, day):
        rows = self.provider.on_date(sid, day)
        capital = [r for r in rows if r['kind'] == 'capital_reduction']
        if not capital:
            return rows
        if len(rows) != 1:
            raise UnresolvedAction('Simultaneous capital actions require separate entitlement review')
        action = capital[0]
        if (action['action_id'] in self.processed or action.get('cash_rounding') != 'floor_ntd'
                or action.get('fractional_policy') != 'historical_close_gross_receivable'
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
        price = self.account.raw(pd.Timestamp(action['fractional_reference_date']), sid)
        if price is None or self.account.raw(stamp, sid) is None:
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
