"""Small offline checks for frozen-input provenance and valuation diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_diffusion as driver


@pytest.fixture
def input_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'CACHE', tmp_path)
    # verify_inputs hashes bytes; parquet decoding belongs to prepare_inputs.
    # Distinct tiny files make accidental filename/digest substitutions visible.
    for name in driver.INPUT_FILES:
        (tmp_path / name).write_bytes(f'synthetic immutable {name}'.encode())
    info = {
        'schema': 1,
        'input_transform_sha256': driver.input_transform_hash(),
        'transform_versions': {'numpy': driver.np.__version__, 'pandas': driver.pd.__version__,
                               'duckdb': driver.duckdb.__version__},
        'files_sha256': {name: driver.sha(tmp_path / name) for name in driver.INPUT_FILES},
    }
    (tmp_path / 'inputs.json').write_text(json.dumps(info))
    return tmp_path, info


def test_unchanged_input_transform_runtime_and_files_allow_reuse(input_manifest):
    _, info = input_manifest
    assert driver.verify_inputs() == info


@pytest.mark.parametrize('change', ['stale_transform', 'missing_transform', 'numpy', 'pandas', 'duckdb'])
def test_input_provenance_rejects_stale_transform_or_runtime(input_manifest, change):
    path, info = input_manifest
    if change == 'stale_transform':
        info['input_transform_sha256'] = '0' * 64
    elif change == 'missing_transform':
        del info['input_transform_sha256']
    else:
        info['transform_versions'][change] = 'unrelated-runtime-version'
    (path / 'inputs.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='Input transformation changed'):
        driver.verify_inputs()
    # --prepare-inputs must not silently accept/rewrite a stale existing cache.
    before = (path / 'inputs.json').read_bytes()
    with pytest.raises(ValueError, match='Input transformation changed'):
        driver.prepare_inputs()
    assert (path / 'inputs.json').read_bytes() == before


def test_valid_manifest_cannot_hide_mutated_input_file(input_manifest):
    path, _ = input_manifest
    (path / 'turnover.parquet').write_bytes(b'different units or observations')
    with pytest.raises(ValueError, match='Frozen input changed: turnover.parquet'):
        driver.verify_inputs()


@pytest.mark.parametrize('changed_function', ['prepare_inputs', 'save_matrix'])
def test_transform_digest_covers_both_preparation_and_serialization(monkeypatch, changed_function):
    before = driver.input_transform_hash()

    def different_transform(*args, **kwargs):
        raise AssertionError('This replacement is inspected, never executed')

    monkeypatch.setattr(driver, changed_function, different_transform)
    assert driver.input_transform_hash() != before


def valuation_prices():
    days = pd.bdate_range('2022-01-03', periods=7)
    close = pd.DataFrame({
        '0050': [100., 101., 102., 103., 104., 105., 106.],
        # Resume on Jan 6, source disagreement on Jan 7, large later jumps.
        '1101': [100., 100., np.nan, 130., 140., 200., 300.],
        '1102': [100., 100., np.nan, 130., 140., 200., 300.],
    }, index=days)
    other = close.copy()
    other.loc[days[4]:, ['1101', '1102']] = [[120., 120.], [180., 180.], [280., 280.]]
    return close, other


def test_price_anomalies_detect_resumed_quote_and_cross_basis_valuation():
    close, other = valuation_prices()
    anomalies = driver.price_anomalies(close, other)
    assert '2022-01-05' not in anomalies  # Missing quote is not a zero mark.
    resumed = next(item for item in anomalies['2022-01-06'] if item['stock_id'] == '1101')
    assert resumed['observed_return_since_last_quote'] == pytest.approx(.30)
    disagreement = next(item for item in anomalies['2022-01-07'] if item['stock_id'] == '1101')
    assert disagreement['observed_return_since_last_quote'] == pytest.approx(140 / 130 - 1)
    assert disagreement['price_basis_difference'] == pytest.approx(140 / 120 - 1)
    assert all(item['stock_id'] != '0050' for batch in anomalies.values() for item in batch)


def test_valuation_audit_keeps_exit_day_and_resumption_but_ignores_nonheld_and_future():
    close, other = valuation_prices()
    anomalies = driver.price_anomalies(close, other)
    sim = {
        'curve': [{'date': str(day.date()), 'nav': 1.} for day in close.index[:6]],
        'executions': [
            {'date': '2022-01-03', 'stock_id': '1101', 'side': 'buy', 'units': 1.},
            {'date': '2022-01-07', 'stock_id': '1101', 'side': 'sell', 'units': 1.},
        ],
    }
    audit = driver.valuation_audit(sim, anomalies)
    assert [(item['date'], item['stock_id']) for item in audit['findings']] == [
        ('2022-01-06', '1101'), ('2022-01-07', '1101')]
    # The exit-session valuation mattered even though end-of-day units are 0.
    # Jan 10 is post-sale, Jan 11 is outside the curve, and 1102 was never held.
    assert audit['unresolved_valuation_days'] == 2


def test_entry_day_anomaly_is_visible_even_without_prior_holdings():
    sim = {'curve': [{'date': '2022-01-03', 'nav': 1.}],
           'executions': [{'date': '2022-01-03', 'stock_id': '1101', 'side': 'buy', 'units': 1.}]}
    anomalies = {'2022-01-03': [{'stock_id': '1101', 'observed_return_since_last_quote': .3,
                                'price_basis_difference': .01}]}
    audit = driver.valuation_audit(sim, anomalies)
    assert len(audit['findings']) == 1
    assert audit['findings'][0]['stock_id'] == '1101'


def test_multiple_held_anomalies_on_one_date_count_as_one_day():
    sim = {'curve': [{'date': '2022-01-03', 'nav': 1.}],
           'executions': [{'date': '2022-01-03', 'stock_id': sid, 'side': 'buy', 'units': 1.}
                          for sid in ('1101', '1102')]}
    anomalies = {'2022-01-03': [{'stock_id': sid, 'observed_return_since_last_quote': .3,
                                'price_basis_difference': .01} for sid in ('1101', '1102')]}
    audit = driver.valuation_audit(sim, anomalies)
    assert audit['unresolved_valuation_days'] == 1
    assert audit['finding_count'] == 2
    assert len(audit['findings']) == 2


def test_future_quote_changes_cannot_rewrite_earlier_valuation_findings():
    close, other = valuation_prices()
    before = driver.price_anomalies(close, other)
    close.loc['2022-01-10':, '1101'] *= 5
    other.loc['2022-01-10':, '1101'] *= 7
    after = driver.price_anomalies(close, other)
    assert {day: rows for day, rows in before.items() if day <= '2022-01-07'} == {
        day: rows for day, rows in after.items() if day <= '2022-01-07'}


def test_annual_returns_compound_with_year_boundary_and_initial_cost():
    curve = [
        {'date': '2022-01-03', 'nav': .99},  # Initial transaction cost is included.
        {'date': '2022-12-30', 'nav': 1.10},
        {'date': '2023-01-03', 'nav': .88},  # -20%, then +50% compounds to +20%.
        {'date': '2023-12-29', 'nav': 1.32},
        {'date': '2024-01-02', 'nav': .99},
    ]
    annual = driver.annual_returns(curve)
    assert annual == pytest.approx({'2022': .10, '2023': .20, '2024': -.25})
    assert np.prod([1 + value for value in annual.values()]) == pytest.approx(curve[-1]['nav'])
    assert driver.annual_returns([{'date': '2022-01-03', 'nav': .99}]) == pytest.approx({'2022': -.01})
    assert driver.annual_returns([]) == {}
