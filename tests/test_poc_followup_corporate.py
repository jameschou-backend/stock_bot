import json
from pathlib import Path

import pytest

from skills.poc_followup_corporate import DOCUMENT, EXPECTED, ROOT, load


def test_verified_2236_entitlement_has_no_invented_fractional_cash():
    terms, sources = load()
    assert all(terms[k] == v for k, v in EXPECTED.items())
    assert terms['ordinary_share_available_date'] == '2024-10-23'
    assert terms['shares_per_share'] == .021397789
    assert terms['fractional_cash_available_for_trading'] is False
    assert any(p.endswith('minutes.pdf') for p in sources)


def test_unverified_fractional_cash_cannot_replace_verified_net_zero(tmp_path):
    doc = json.loads(DOCUMENT.read_text())
    doc['override']['fractional_cash_per_share'] = 10
    path = tmp_path/'delivery.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='scope or settlement'):
        load(root=tmp_path, document=path)
