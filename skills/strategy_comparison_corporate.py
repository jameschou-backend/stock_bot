"""Source-bound settlement supplements for the fixed comparison only.

An undated gross fractional right is never broker cash or a verified net asset.
The frozen financial engine already journals it separately from whole shares.
"""
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse

TERMS = 'docs/strategy_comparison_corporate_terms_20261007.json'
EVENTS = frozenset({'3086-2024-09-24', '8932-2025-09-08'})
PRIMARY_HOSTS = frozenset({'mops.twse.com.tw', 'mopsov.twse.com.tw', 'www.twse.com.tw', 'www.tpex.org.tw'})


def _decimal(value, label):
    if isinstance(value, bool):
        raise ValueError('Invalid settlement amount: ' + label)
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError('Invalid settlement amount: ' + label) from None
    if not result.is_finite() or result <= 0:
        raise ValueError('Invalid settlement amount: ' + label)
    return result


def load_comparison_corporate_terms(root):
    """Return overrides, complete evidence hashes and valuation qualifications."""
    root = Path(root).resolve(); refs = {}

    def bind(name, expected=None):
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or Path(name).as_posix() != name):
            raise ValueError('Comparison settlement evidence path is unsafe')
        path = root / name
        if (not path.resolve().is_relative_to(root) or not path.is_file()
                or any(part.is_symlink() for part in (path, *path.parents) if part != root)):
            raise ValueError('Comparison settlement evidence path is unavailable')
        if expected is not None and (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)):
            raise ValueError('Comparison settlement evidence SHA is invalid')
        actual = sha256(path.read_bytes()).hexdigest()
        if expected is not None and actual != expected or name in refs and refs[name] != actual:
            raise ValueError('Comparison settlement evidence hash changed: ' + name)
        refs[name] = actual
        return path

    document = json.loads(bind(TERMS).read_text())
    overrides = document.get('overrides')
    if (document.get('schema') != 'strategy_comparison_corporate_completion_v1'
            or document.get('strategy_parameters_changed') is not False
            or document.get('cash_supplements') != []
            or document.get('live_qualified') is not False
            or not isinstance(overrides, dict) or not overrides or not set(overrides) <= EVENTS):
        raise ValueError('Unexpected comparison settlement scope')
    qualifications = {}
    for key, row in overrides.items():
        sid, ex = key[:4], key[5:]
        evidence = row.get('evidence_manifest', {})
        if not isinstance(evidence.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', evidence['sha256']):
            raise ValueError('Comparison settlement requires its manifest SHA')
        manifest = json.loads(bind(evidence.get('path'), evidence.get('sha256')).read_text())
        if (manifest.get('schema') != 'strategy_comparison_corporate_evidence_v1'
                or manifest.get('stock_id') != sid or manifest.get('action_id') != key
                or manifest.get('use_scope') != 'account_settlement_only_not_selection'
                or manifest.get('strategy_parameters_changed') is not False
                or manifest.get('live_qualified') is not False):
            raise ValueError('Comparison settlement manifest identity differs')
        sources = manifest.get('source_sha256')
        if not isinstance(sources, dict) or not sources:
            raise ValueError('Comparison settlement lacks original source evidence')
        for name, expected in sources.items():
            if expected is None:
                raise ValueError('Comparison settlement source lacks SHA')
            bind(name, expected)
        official = manifest.get('official_sources', [])
        if not official:
            raise ValueError('Comparison settlement lacks official announcements')
        purposes = set()
        for source in official:
            name, receipt = source.get('path'), source.get('receipt_path')
            if sources.get(name) != source.get('sha256') or sources.get(receipt) != source.get('receipt_sha256'):
                raise ValueError('Comparison official announcement is outside its evidence closure')
            meta = json.loads(bind(receipt, source['receipt_sha256']).read_text())
            url = urlparse(source.get('url', ''))
            announced = date.fromisoformat(source['announcement_date'])
            roc = f'{announced.year-1911:03d}/{announced.month:02d}/{announced.day:02d}'
            announcement = meta.get('announcement_row', [])
            if (url.scheme != 'https' or url.hostname not in PRIMARY_HOSTS
                    or source.get('http_status') != 200 or meta.get('http_status') != 200
                    or meta.get('redirected') is not False or meta.get('sha256') != sources[name]
                    or meta.get('url') != source['url'] or len(announcement) < 3
                    or announcement[0] != sid or announcement[2] != roc):
                raise ValueError('Comparison source was not the exact successful primary announcement')
            purposes.add(source['purpose'])
        if not {'issuer_confirmed_delivery', 'official_listing_confirmation'} <= purposes:
            raise ValueError('Tentative delivery alone cannot unlock share delivery')
        facts = manifest['verified_terms']
        for purpose, field in (
            ('issuer_entitlement_and_tentative_delivery', 'entitlement_announcement_date'),
            ('issuer_confirmed_delivery', 'delivery_confirmation_announcement_date'),
            ('official_listing_confirmation', 'listing_confirmation_announcement_date'),
        ):
            matching = [source for source in official if source['purpose'] == purpose]
            if len(matching) != 1 or matching[0]['announcement_date'] != facts.get(field):
                raise ValueError('Comparison settlement facts differ from the dated official notices')
        if (facts.get('ex_date') != ex or row.get('pay_date') != facts.get('stock_pay_date')
                or row.get('ordinary_share_available_date') != facts.get('ordinary_share_available_date')
                or row.get('pay_date') != row.get('ordinary_share_available_date')
                or facts.get('same_rights_as_existing_ordinary') is not True):
            raise ValueError('Comparison share delivery differs from confirmed evidence')
        if (_decimal(row.get('shares_per_share'), 'rate') != _decimal(facts.get('shares_per_share_decimal'), 'official rate')
                or _decimal(row.get('fractional_cash_per_share'), 'face') != _decimal(facts.get('par_value_ntd'), 'official face')
                or _decimal(facts.get('shares_per_1000'), 'per1000') / 1000 != _decimal(facts.get('shares_per_share_decimal'), 'rate')):
            raise ValueError('Comparison rate or historical par value differs from evidence')
        for field, fact in (
            ('entitlement_announcement_date', 'entitlement_announcement_date'),
            ('delivery_announcement_date', 'delivery_confirmation_announcement_date'),
            ('listing_announcement_date', 'listing_confirmation_announcement_date'),
        ):
            if row.get(field) != facts.get(fact):
                raise ValueError('Comparison settlement announcement date differs')
        entitlement, delivery, listing, pay = [date.fromisoformat(row[field]) for field in
            ('entitlement_announcement_date', 'delivery_announcement_date', 'listing_announcement_date', 'pay_date')]
        if not entitlement <= date.fromisoformat(ex) <= delivery <= listing <= pay:
            raise ValueError('Comparison settlement announcement chronology is invalid')
        if (row.get('use_scope') != 'account_settlement_only_not_selection'
                or row.get('fractional_cash_rounding') != 'floor_ntd'
                or facts.get('fractional_cash_rounding') != 'floor_ntd'
                or row.get('fractional_cash_pay_date') is not None
                or facts.get('fractional_cash_pay_date') is not None
                or facts.get('fractional_cash_net_amount') is not None
                or facts.get('spendable_fractional_cash_proven') is not False
                or row.get('fractional_cash_payment_date_verified') is not False
                or row.get('fractional_cash_net_amount_verified') is not False
                or row.get('fractional_cash_available_for_trading') is not False
                or row.get('fractional_cash_treatment') != 'gross_undated_receivable_upper_bound_net_unknown'
                or facts.get('fractional_cash_purpose') != {
                    '3086': 'offset_book_entry_and_dematerialized_registration_fees',
                    '8932': 'offset_book_entry_fees',
                }[sid]):
            raise ValueError('Unverified fractional net cash must remain unavailable')
        if not row.get('evidence_files') or any(name not in sources for name in row['evidence_files']):
            raise ValueError('Comparison settlement row has unbound evidence')
        qualifications[key] = dict(
            fractional_cash_available_for_trading=False, fractional_net_cash_amount=None,
            fractional_cash_payment_date=None, fees_and_net_settlement_verified=False,
            reported_nav_basis='includes_gross_undated_fractional_receivable_upper_bound',
            valuation_sensitivity='Subtract this event actual unpaid gross receivable from reported NAV for zero-asset-value sensitivity; this does not certify other fees or the account net value.',
            amount_source='actual_account_receivable_journal_not_position_example',
            evidence_manifest=deepcopy(evidence))
    return deepcopy(overrides), refs, qualifications
