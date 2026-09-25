"""Explicit unpaid share rights before a verified earliest delivery boundary.

This adapter never invents a settlement date. The sealed account already values
undated receivables; only its creation guard needs a narrowly evidenced path.
"""
from decimal import Decimal, ROUND_FLOOR
import hashlib
import math
from numbers import Real
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from scripts.research_exit_scenarios import TrackedCorporateActions
from skills.million_replay import UnresolvedAction
from skills.replay_corporate_actions import CorporateActions


ROOT = Path(__file__).resolve().parents[1]


def _date(value):
    try:
        stamp = pd.Timestamp(value)
        if (not isinstance(value, str) or pd.isna(stamp) or stamp.tzinfo is not None
                or stamp != stamp.normalize() or str(stamp.date()) != value):
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise UnresolvedAction('Pending share evidence has an invalid date') from None


def validate_pending_terms(terms, sid, day, end, source_root=ROOT):
    """An explicit record-date lower bound is usable only strictly before it."""
    bound = _date(terms.get('pending_delivery_not_before'))
    if (terms.get('pending_only') is not True
            or terms.get('ordinary_share_delivery_status') != 'unannounced'
            or terms.get('pay_date') is not None
            or terms.get('ordinary_share_available_date') is not None
            or terms.get('pending_not_before_basis') != 'record_date'
            or _date(terms.get('record_date')) != bound
            or not _date(day) < bound
            or not str(pd.Timestamp(end).date()) < bound
            or _date(terms.get('entitlement_announcement_date')) > day):
        raise UnresolvedAction(f'Pending share delivery boundary not established: {sid} {day}')
    evidence = terms.get('pending_delivery_evidence', {})
    location = evidence.get('path')
    root = Path(source_root).resolve()
    path = root / location if isinstance(location, str) else root
    if (not isinstance(location, str) or Path(location).is_absolute() or '..' in Path(location).parts
            or not path.resolve().is_relative_to(root) or path.is_symlink()
            or not path.is_file()
            or urlparse(evidence.get('url', '')).hostname not in ('mops.twse.com.tw', 'mopsov.twse.com.tw')
            or not evidence.get('url', '').startswith('https://')
            or hashlib.sha256(path.read_bytes()).hexdigest() != evidence.get('sha256')
            or location not in terms.get('evidence_files', [])):
        raise UnresolvedAction(f'Pending share delivery boundary source missing or changed: {sid} {day}')
    return bound


class PendingOnlySource:
    """Preserve the normal provider; allow null delivery for the evidenced right."""
    def __init__(self, provider, end, source_root):
        self.provider, self.end, self.source_root = provider, end, source_root

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def on_date(self, sid, day):
        terms = self.overrides.get(f'{sid}-{day}', {})
        if terms.get('pending_only') is not True:
            return self.provider.on_date(sid, day)
        validate_pending_terms(terms, sid, day, self.end, self.source_root)
        # TrackedCorporateActions only adds its numerical/date guard to this
        # exact base loader. Invoke the same loader and prepare hook, then keep
        # all its numerical guards below while replacing the delivery guard.
        if isinstance(self.provider, TrackedCorporateActions):
            rows = CorporateActions.on_date(self.provider, sid, day)
        else:
            rows = self.provider.on_date(sid, day)
        stocks = [row for row in rows if row.get('kind') == 'stock_dividend']
        if len(stocks) != 1:
            raise UnresolvedAction(f'Pending share action is absent or ambiguous: {sid} {day}')
        row = stocks[0]
        for field in ('shares_per_share', 'fractional_cash_per_share'):
            value = row.get(field)
            if (isinstance(value, bool) or not isinstance(value, (Real, Decimal))
                    or not math.isfinite(value) or value < 0):
                raise UnresolvedAction(f'Pending share amount invalid: {sid} {day} {field}')
        if (row.get('pay_date') is not None or row.get('shares_per_share') != terms.get('shares_per_share')
                or row.get('stock_id') != sid or row.get('date') != day
                or terms.get('fractional_cash_rounding') != 'floor_ntd'
                or terms.get('fractional_cash_pay_date') is not None):
            raise UnresolvedAction(f'Pending share terms disagree with action: {sid} {day}')
        return rows


class PendingShareRights:
    """Create an explicitly undated receivable after fractional-cash separation."""
    def __init__(self, provider, account, source_root):
        self.provider, self.account, self.source_root = provider, account, source_root

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def on_date(self, sid, day):
        terms = self.overrides.get(f'{sid}-{day}', {})
        if terms.get('pending_only') is not True:
            return self.provider.on_date(sid, day)
        bound = validate_pending_terms(terms, sid, day, self.account.end, self.source_root)
        rows = self.provider.on_date(sid, day)
        if self.account.raw(pd.Timestamp(day), sid) is None:
            raise UnresolvedAction(f'Ex-date valuation price missing: {sid} {day}')
        result = []
        holding = self.account.holdings[sid]
        for row in rows:
            if row.get('kind') != 'stock_dividend':
                result.append(row)
                continue
            if row.get('fractional_cash_per_share') != 0 or 'fractional_settlement' not in row:
                raise UnresolvedAction(f'Pending shares require separate fractional cash: {sid} {day}')
            if any(item['action_id'] == row['action_id'] for item in self.account.receivables):
                raise UnresolvedAction(f'Duplicate pending share entitlement: {sid} {day}')
            quantity = Decimal(holding['qty']) * Decimal(str(row['shares_per_share']))
            whole = int(quantity.to_integral_value(rounding=ROUND_FLOOR))
            fraction = float(quantity - whole)
            pending = dict(stock_id=sid, qty=whole, fraction=fraction, kind='shares',
                pay_date=None, action_id=row['action_id'], event_id=holding['event_id'],
                ex_date=day, fractional_cash_per_share=0., delivery_status='pending_unannounced',
                delivery_not_before=bound, delivery_evidence=dict(terms['pending_delivery_evidence']),
                valuation_basis='ordinary_share_close_proxy', tradable=False)
            self.account.receivables.append(pending)
            self.account.actions.append(dict(row, date=day, entitled_qty=holding['qty'],
                event_id=holding['event_id'], whole_new_shares=whole, fractional_right=fraction,
                delivery_status=pending['delivery_status'], delivery_not_before=bound,
                delivery_evidence=pending['delivery_evidence'], tradable=False))
        return result


def install_pending_share_rights(account, source_root=ROOT):
    """No-op unless explicit pending-only evidence is among the supplied terms."""
    if not any(value.get('pending_only') is True
               for value in getattr(account.corporate, 'overrides', {}).values()):
        return
    # ScenarioExitReplay installs FractionalCashActions around execution's lazy
    # provider. Place the source adapter below both and the journal adapter above
    # both. Existing prepare methods, source tracking and cash algebra persist.
    owner = None
    provider = account.corporate
    while 'provider' in vars(provider):
        owner, provider = provider, provider.provider
    replacement = PendingOnlySource(provider, account.end, source_root)
    if owner is None:
        account.corporate = replacement
    else:
        owner.provider = replacement
    account.corporate = PendingShareRights(account.corporate, account, source_root)
