import pytest
from skills.repaired_execution_context import benchmark_exclusion_with_market, dated_market_resolver
from test_historical_selector_replay import identities


def test_sealed_benchmark_exclusion_does_not_break_unrelated_stock_routing():
    exclusion = dict(stock_id='0050',kind='trading_suspension',start='2025-06-11',end='2025-06-18')
    original = identities()
    original['trading_exclusions'].append(benchmark_exclusion_with_market(exclusion))
    resolve = dated_market_resolver(original)
    assert resolve('2025-06-12','1101') == resolve('2025-06-12','0050') == 'TWSE'
    assert 'market' not in exclusion


def test_unknown_market_fails_at_preflight_instead_of_after_account_replay():
    original = identities()
    original['trading_exclusions'].append(dict(stock_id='1101',kind='managed_board',start='2020-01-01',end=None))
    with pytest.raises(ValueError,match='Missing explicit'): dated_market_resolver(original)


def test_outside_coverage_and_managed_identity_never_supply_capacity_venue():
    original = identities(exclusions=[dict(stock_id='1101',market='TWSE',kind='managed_board',start='2025-01-01',end=None)])
    resolve = dated_market_resolver(original)
    assert resolve('2024-01-01','1101') == 'TWSE'
    assert resolve('2025-01-01','1101') is None
    assert resolve('1990-01-01','1101') is None
