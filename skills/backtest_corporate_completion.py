"""Read the separate, primary-source corporate supplement without changing a seal."""
from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
import re

from skills.backtest_case_cache import file_identities


DOCUMENT = 'docs/backtest_corporate_completion_20260925.json'


def load_corporate_completion(root):
    """Return validated additional overrides; all source bytes are checked each call.

    The dated fields describe settlement, not an input to historical stock selection.
    Unverified fractional cash cannot be released on the ordinary-share pay date.
    """
    root = Path(root).resolve()
    doc = json.loads((root / DOCUMENT).read_text(encoding='utf-8'))
    if (doc.get('schema') != 1 or doc.get('secondary_sources_used_in_overrides') is not False
            or doc.get('finmind_requests') != 0):
        raise ValueError('Corporate completion provenance is invalid')
    evidence = doc.get('evidence_sha256')
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('Corporate completion requires primary evidence hashes')
    if any(not isinstance(name, str) or not isinstance(digest, str)
           or not re.fullmatch('[0-9a-f]{64}', digest) for name, digest in evidence.items()):
        raise ValueError('Corporate completion evidence inventory is invalid')
    for name in evidence:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError('Corporate evidence path must be relative and canonical')
    current = file_identities([root / name for name in evidence], root)
    if current != evidence:
        raise ValueError('Corporate completion primary evidence changed')
    overrides = doc.get('overrides')
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError('Corporate completion has no overrides')
    for key, terms in overrides.items():
        if not re.fullmatch(r'\d{4}-\d{4}-\d{2}-\d{2}', key) or not isinstance(terms, dict):
            raise ValueError('Corporate completion event key is invalid')
        ex = date.fromisoformat(key[5:])
        pay = date.fromisoformat(terms['pay_date'])
        if pay < ex:
            raise ValueError('Corporate delivery precedes ex-date')
        rate = terms.get('shares_per_share')
        face = terms.get('fractional_cash_per_share')
        if any(isinstance(x, bool) or not isinstance(x, (float, int))
               or not math.isfinite(x) or x < 0 for x in (rate, face)) or rate == 0:
            raise ValueError('Corporate entitlement must be finite and positive')
        if (terms.get('fractional_cash_rounding') != 'floor_ntd'
                or terms.get('fractional_cash_pay_date') is not None):
            raise ValueError('Unverified fractional cash must remain a rounded receivable')
        files = terms.get('evidence_files')
        if not isinstance(files, list) or not files or any(f not in evidence for f in files):
            raise ValueError('Corporate event lacks anchored primary evidence')
        if 'certificate_delivery_date' in terms:
            certificate = date.fromisoformat(terms['certificate_delivery_date'])
            if (not ex <= certificate <= pay or terms.get('certificate_trading_modeled') is not False
                    or terms.get('ordinary_share_available_date') != terms['pay_date']
                    or terms.get('valuation_basis') != 'ordinary_share_close_proxy'):
                raise ValueError('Rights certificate must remain unavailable until ordinary conversion')
    return deepcopy(overrides)
