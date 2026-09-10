"""Driver helper checks with tiny local fixtures, no historical simulation."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_regime_switch as driver


def event(entry='2022-01-07', signal='2022-01-06'):
    return {'event_id': 'e', 'signal_date': signal, 'entry_date': entry,
            'members': ['1101', '1102'], 'priority': 1.25}


def test_delay_events_uses_exact_market_sessions_and_preserves_original_signal():
    days = pd.bdate_range('2022-01-03', periods=8)
    supplied = [event()]
    original = copy.deepcopy(supplied)
    same = driver.delay_events(supplied, days, 0)
    delayed = driver.delay_events(supplied, days, np.int64(1))
    assert same == original and same[0] is not supplied[0]
    assert delayed[0]['entry_date'] == '2022-01-10'
    assert delayed[0]['signal_date'] == original[0]['signal_date']
    assert delayed[0]['members'] == original[0]['members']
    assert delayed[0]['priority'] == original[0]['priority']
    assert supplied == original


@pytest.mark.parametrize('entry', ['2022-01-02', '2022-01-08', '2022-02-01', '2022-01-07 09:00:00'])
def test_delay_events_never_rounds_missing_dates_to_an_available_session(entry):
    days = pd.bdate_range('2022-01-03', periods=8)
    with pytest.raises(ValueError, match='exact market session'):
        driver.delay_events([event(entry=entry)], days, 0)


def test_delay_events_cannot_drop_or_move_beyond_calendar_end():
    days = pd.bdate_range('2022-01-03', periods=8)
    supplied = [event(entry=str(days[-1].date()), signal=str(days[-2].date()))]
    assert driver.delay_events(supplied, days, 0)[0]['entry_date'] == str(days[-1].date())
    with pytest.raises(ValueError, match='no execution session'):
        driver.delay_events(supplied, days, 1)


@pytest.mark.parametrize('delay', [-1, True, 0.5, 1.0, '1', None])
def test_delay_events_rejects_invalid_delays_without_wrapping_or_advancing(delay):
    days = pd.bdate_range('2022-01-03', periods=8)
    with pytest.raises(ValueError, match='delay'):
        driver.delay_events([event(entry=str(days[0].date()), signal='2021-12-31')], days, delay)


def test_combine_audits_deduplicates_stock_days_but_counts_unique_calendar_dates():
    one = {'date': '2023-04-06', 'stock_id': '1101', 'price_basis_difference': .03}
    two = {'date': '2023-04-06', 'stock_id': '1102', 'price_basis_difference': .04}
    later = {'date': '2023-04-07', 'stock_id': '1101', 'price_basis_difference': .05}
    a = {'finding_count': 3, 'unresolved_valuation_days': 2, 'findings': [later, two, one]}
    b = {'finding_count': 1, 'unresolved_valuation_days': 1, 'findings': [copy.deepcopy(one)]}
    saved = copy.deepcopy((a, b))
    combined = driver.combine_audits(a, b)
    assert combined == {'finding_count': 3, 'unresolved_valuation_days': 2, 'findings': [one, two, later]}
    assert (a, b) == saved
    assert driver.combine_audits(b, a) == combined


def test_combine_audits_empty_accounts_do_not_invent_findings():
    assert driver.combine_audits() == {'finding_count': 0, 'unresolved_valuation_days': 0, 'findings': []}
    assert driver.combine_audits({'findings': []}, {'findings': []}) == driver.combine_audits()


@pytest.fixture
def prior_probe(tmp_path, monkeypatch):
    probe = tmp_path / 'probe'
    diffusion = tmp_path / 'diffusion'
    probe.mkdir(); diffusion.mkdir()
    monkeypatch.setattr(driver, 'PROBE', probe)
    monkeypatch.setattr(driver.source, 'CACHE', diffusion)
    def write(path, text):
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()
    signal_hash = write(diffusion / 'signals.json', 'sealed source signals')
    record = {'protocol_sha256': write(probe / 'protocol.md', 'prior fixed protocol'),
              'script_sha256': write(probe / 'probe.py', '# prior probe'),
              'state_files_sha256': {name: write(probe / name, 'tiny fixture ' + name)
                                     for name in ('states-official.parquet', 'states-snapshot.parquet')},
              'diffusion_signal_manifest_sha256': signal_hash}
    write(probe / 'report.json', json.dumps(record))
    return probe, diffusion, record


def test_prior_control_reference_pins_report_code_states_and_source_signal_manifest(prior_probe):
    _, _, record = prior_probe
    report, hashes = driver.prior_probe()
    assert report == record
    assert set(hashes) == {'.cache/regime-probe-20260910/' + name for name in (
        'protocol.md', 'probe.py', 'states-official.parquet', 'states-snapshot.parquet', 'report.json')}


@pytest.mark.parametrize('changed', ['protocol.md', 'probe.py', 'states-official.parquet', 'states-snapshot.parquet'])
def test_changed_prior_control_artifact_cannot_be_treated_as_reproduced(prior_probe, changed):
    probe, _, _ = prior_probe
    (probe / changed).write_text('changed prior reference')
    with pytest.raises(ValueError, match='source changed'):
        driver.prior_probe()


def test_prior_control_must_use_same_diffusion_signal_snapshot(prior_probe):
    _, diffusion, _ = prior_probe
    (diffusion / 'signals.json').write_text('different signals')
    with pytest.raises(ValueError, match='same sealed inputs'):
        driver.prior_probe()


def test_prior_control_cannot_omit_one_price_basis_state_file(prior_probe):
    probe, _, report = prior_probe
    del report['state_files_sha256']['states-snapshot.parquet']
    (probe / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='same sealed inputs'):
        driver.prior_probe()
