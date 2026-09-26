from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.exit_corporate_completion import DOCUMENT, load_exit_completion
from skills.scenario_exit_replay import FractionalCashActions


@pytest.fixture
def package(tmp_path):
    doc = json.loads((Path(__file__).resolve().parents[1] / DOCUMENT).read_text())
    raw = tmp_path / 'issuer.html'
    raw.write_text('Frozen issuer terms')
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    meta = tmp_path / 'issuer.html.source.json'
    meta.write_text(json.dumps(dict(http_status=200, sha256=digest,
        url='https://mopsov.twse.com.tw/mops/web/issuer')))
    doc['evidence_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw, meta)}
    for row in doc['overrides'].values():
        row['evidence_files'] = ['issuer.html']
    def save():
        (tmp_path / DOCUMENT).parent.mkdir(exist_ok=True)
        (tmp_path / DOCUMENT).write_text(json.dumps(doc))
    save()
    return tmp_path, doc, save


def test_modified_primary_source_rejected(package):
    root, _, _ = package
    load_exit_completion(root)
    (root / 'issuer.html').write_text('Different evidence')
    with pytest.raises(ValueError, match='evidence changed'):
        load_exit_completion(root)


def test_error_response_cannot_be_promoted_to_evidence(package):
    root, doc, save = package
    p = root / 'issuer.html.source.json'
    meta = json.loads(p.read_text()); meta['http_status'] = 502
    p.write_text(json.dumps(meta))
    doc['evidence_sha256'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    save()
    with pytest.raises(ValueError, match='successful primary'):
        load_exit_completion(root)


@pytest.mark.parametrize('field,value', [
    ('pay_date', '2025-10-17'), ('ordinary_share_available_date', '2025-10-17'),
    ('certificate_trading_modeled', True), ('fractional_cash_per_share', 10),
    ('fractional_cash_pay_date', '2025-12-03'), ('shares_per_share', float('nan'))])
def test_unsafe_settlement_rejected(package, field, value):
    root, doc, save = package
    doc['overrides']['2374-2025-09-17'][field] = value
    save()
    with pytest.raises(ValueError):
        load_exit_completion(root)


def test_fractional_expense_and_unpaid_cash_are_not_available_capital(package):
    root, _, _ = package
    overrides, _ = load_exit_completion(root)
    for sid, day, quantity in [('2374', '2025-09-17', 1000), ('2543', '2025-09-02', 1010)]:
        terms = overrides[f'{sid}-{day}']
        provider = SimpleNamespace(overrides=overrides, on_date=lambda s, d: [dict(
            action_id=f'{s}-{d}-stock', stock_id=s, date=d, kind='stock_dividend', **deepcopy(terms))])
        account = SimpleNamespace(holdings={sid: {'qty': quantity}})
        rows = FractionalCashActions(provider, account).on_date(sid, day)
        stock = rows[-1]
        assert stock['fractional_cash_per_share'] == 0
        assert stock['pay_date'] == terms['pay_date']
        if sid == '2374':
            assert len(rows) == 1
            assert stock['fractional_settlement']['fraction'] == pytest.approx(.79631)
            assert stock['certificate_restriction']['ordinary_share_available_date'] == '2025-12-03'
        else:
            assert rows[0]['gross_cash_amount'] == 7
            assert rows[0]['pay_date'] is None


def test_local_frozen_primary_evidence():
    root = Path(__file__).resolve().parents[1]
    if not (root / '.cache/exit-corporate-20260927').exists():
        pytest.skip('Primary evidence retained in local research cache')
    terms, _ = load_exit_completion(root)
    assert terms['2374-2025-09-17']['shares_per_share'] == .11979631
    assert terms['2543-2025-09-02']['pay_date'] == '2025-10-13'
