import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.backtest_corporate_completion import DOCUMENT, load_corporate_completion
from skills.scenario_exit_replay import FractionalCashActions


@pytest.fixture
def package(tmp_path):
    evidence = tmp_path / 'evidence.html'
    evidence.write_text('Issuer original evidence')
    terms = dict(shares_per_share=.025, pay_date='2025-11-14',
        fractional_cash_per_share=10, fractional_cash_rounding='floor_ntd',
        fractional_cash_pay_date=None, evidence_files=['evidence.html'])
    doc = dict(schema=1, secondary_sources_used_in_overrides=False, finmind_requests=0,
        evidence_sha256={'evidence.html': hashlib.sha256(evidence.read_bytes()).hexdigest()},
        overrides={'2881-2025-09-25': terms})
    (tmp_path / DOCUMENT).parent.mkdir(parents=True)
    (tmp_path / DOCUMENT).write_text(json.dumps(doc))
    return tmp_path, doc


def save(package):
    root, doc = package
    (root / DOCUMENT).write_text(json.dumps(doc))


def test_evidence_is_rehashed_each_load(package):
    root, _ = package
    assert load_corporate_completion(root)['2881-2025-09-25']['pay_date'] == '2025-11-14'
    path = root / 'evidence.html'
    path.write_text('Issuer modified evidence')
    with pytest.raises(ValueError, match='evidence changed'):
        load_corporate_completion(root)


def test_unverified_fractional_cash_never_assumes_share_delivery_date(package):
    root, doc = package
    doc['overrides']['2881-2025-09-25']['fractional_cash_pay_date'] = '2025-11-14'
    save(package)
    with pytest.raises(ValueError, match='remain a rounded receivable'):
        load_corporate_completion(root)


@pytest.mark.parametrize('name', ['../evidence.html', '/tmp/evidence.html', './evidence.html'])
def test_no_evidence_path_escape(package, name):
    root, doc = package
    doc['evidence_sha256'] = {name: 'a' * 64}
    save(package)
    with pytest.raises(ValueError, match='relative and canonical'):
        load_corporate_completion(root)


@pytest.mark.parametrize('rate', [float('nan'), float('inf'), True, -.01, 0])
def test_invalid_share_rate(package, rate):
    root, doc = package
    doc['overrides']['2881-2025-09-25']['shares_per_share'] = rate
    save(package)
    with pytest.raises(ValueError, match='finite and positive'):
        load_corporate_completion(root)


def test_certificate_cannot_become_ordinary_before_conversion(package):
    root, doc = package
    row = doc['overrides']['2881-2025-09-25']
    row.update(certificate_delivery_date='2025-08-29', ordinary_share_available_date='2025-10-09',
               pay_date='2025-08-29', certificate_trading_modeled=False,
               valuation_basis='ordinary_share_close_proxy')
    doc['overrides'] = {'2880-2025-08-13': row}
    save(package)
    with pytest.raises(ValueError, match='ordinary conversion'):
        load_corporate_completion(root)


def test_new_primary_package_and_fractional_adapter():
    root = Path(__file__).resolve().parents[1]
    if not (root / '.cache/backtest-corporate-completion-20260925').exists():
        pytest.skip('Local frozen primary evidence is not distributed with source code')
    overrides = load_corporate_completion(root)
    assert overrides['2881-2025-09-25']['shares_per_share'] == .025
    assert overrides['2880-2025-08-13']['pay_date'] == '2025-10-09'
    assert overrides['2880-2026-08-13']['pay_date'] == '2026-09-18'
    assert overrides['2880-2026-08-13']['pay_date'] > '2026-09-09'
    terms = overrides['2881-2025-09-25']
    provider = SimpleNamespace(overrides=overrides, on_date=lambda sid, day: [dict(
        action_id='2881-2025-09-25-stock', stock_id=sid, date=day, kind='stock_dividend', **terms)])
    account = SimpleNamespace(holdings={'2881': {'qty': 150}})
    rows = FractionalCashActions(provider, account).on_date('2881', '2025-09-25')
    cash, stock = rows
    assert cash['gross_cash_amount'] == 7
    assert cash['pay_date'] is None
    assert stock['pay_date'] == '2025-11-14'
    assert stock['fractional_cash_per_share'] == 0
    assert stock['fractional_settlement']['payment_date_verified'] is False
