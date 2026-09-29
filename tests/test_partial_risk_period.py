import pytest
from skills.partial_risk_period import validate_period, reconcile_runs


def contract(**changes):
    return dict(start='2022-01-03', end='2026-09-09', selector_start='2022-01-03',
        selector_end='2026-09-09', identity_start='2021-01-01', identity_end='2026-09-09',
        sessions=['2019-01-02', '2021-01-04', '2022-01-03', '2026-09-09']) | changes


@pytest.mark.parametrize('start', ['2019-01-02', '2021-01-04'])
def test_warmup_prices_do_not_extend_selector_coverage(start):
    with pytest.raises(ValueError, match='selector/identity'):
        validate_period(**contract(start=start))


def test_valid_period_keeps_exact_endpoints():
    assert validate_period(**contract()) == ['2022-01-03', '2026-09-09']
    with pytest.raises(ValueError, match='audited market sessions'):
        validate_period(**contract(start='2022-01-04'))


def report(**changes):
    return dict(start='2022-01-03', end='2026-09-09', source_sha256={'a': 'hash'},
        preparation=False, all_completed=True, cases=dict(original={}, cap40={}, benchmark={})) | changes


@pytest.mark.parametrize('change', [dict(start='2025-01-02'), dict(preparation=True),
    dict(all_completed=False), dict(source_sha256={'a': 'changed'}), dict(cases={'original': {}})])
def test_cannot_publish_unmatched_or_incomplete_runs(change):
    with pytest.raises(ValueError):
        reconcile_runs(report(), report(**change))
