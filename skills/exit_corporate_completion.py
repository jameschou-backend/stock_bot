"""Primary-source settlement supplement for the frozen exit/risk studies."""
from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from skills.backtest_case_cache import file_identities

DOCUMENT = 'docs/exit_corporate_completion_20260927.json'


def load_exit_completion(root):
    root = Path(root).resolve()
    doc = json.loads((root / DOCUMENT).read_text())
    if (doc.get('schema') != 'exit_corporate_completion_v1'
            or doc.get('secondary_sources_used_in_overrides') is not False
            or doc.get('finmind_requests') != 0):
        raise ValueError('Invalid settlement supplement provenance')
    evidence = doc['evidence_sha256']
    if not evidence or any(Path(p).is_absolute() or '..' in Path(p).parts
                           or Path(p).as_posix() != p for p in evidence):
        raise ValueError('Settlement evidence paths must be canonical and relative')
    if file_identities([root / p for p in evidence], root) != evidence:
        raise ValueError('Settlement evidence changed')
    for name in evidence:
        if name.endswith('.source.json'):
            continue
        if name + '.source.json' not in evidence:
            raise ValueError('Settlement source metadata missing')
        meta = json.loads((root / (name + '.source.json')).read_text())
        if (meta.get('http_status') != 200 or meta.get('sha256') != evidence[name]
                or urlparse(meta['url']).scheme != 'https'
                or urlparse(meta['url']).hostname not in
                ('mops.twse.com.tw', 'mopsov.twse.com.tw', 'www.hcgc.com.tw')):
            raise ValueError('Settlement source was not a successful primary response')
    overrides = doc['overrides']
    if set(overrides) != {'2374-2025-09-17', '2543-2025-09-02'}:
        raise ValueError('Unexpected settlement event scope')
    for key, row in overrides.items():
        ex = date.fromisoformat(key[5:])
        pay = date.fromisoformat(row['pay_date'])
        entitlement = date.fromisoformat(row['entitlement_announcement_date'])
        delivery = date.fromisoformat(row['delivery_announcement_date'])
        listing = date.fromisoformat(row['listing_announcement_date'])
        if not entitlement <= ex <= delivery <= listing <= pay:
            raise ValueError('Settlement announcement chronology is invalid')
        if (row.get('use_scope') != 'account_settlement_only_not_selection'
                or row.get('fractional_cash_rounding') != 'floor_ntd'
                or row.get('fractional_cash_pay_date') is not None):
            raise ValueError('Unverified fractional cash must remain unavailable')
        if not row['evidence_files'] or any(p not in evidence for p in row['evidence_files']):
            raise ValueError('Settlement event evidence is missing')
        for field in ('shares_per_share', 'fractional_cash_per_share'):
            v = row[field]
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError('Settlement amounts must be finite and nonnegative')
        if row['shares_per_share'] <= 0:
            raise ValueError('Stock distribution rate must be positive')
        if key.startswith('2374'):
            if (not ex <= date.fromisoformat(row['certificate_delivery_date']) <= pay
                    or row['ordinary_share_available_date'] != row['pay_date']
                    or row['certificate_trading_modeled'] is not False
                    or row['valuation_basis'] != 'ordinary_share_close_proxy'
                    or row['fractional_net_zero_verified'] is not True
                    or row['fractional_cash_per_share'] != 0):
                raise ValueError('Rights certificate or net fractional terms invalid')
        elif row['fractional_cash_per_share'] != 10:
            raise ValueError('Issuer gross fractional face value is NTD10')
    return deepcopy(overrides), evidence
