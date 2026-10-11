from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

from skills.all_entry_capital_halt import install_capital_halt, load_capital_halt

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT/'docs/all_entry_2352_capital_20261011.json'


def test_dated_halt_uses_bound_announcement_and_never_mutates_frozen_terms(monkeypatch):
    import skills.all_entry_capital_halt as m
    halt = dict(stock_id='2352', start='2025-09-25', end='2025-10-07')
    monkeypatch.setattr(m, 'load_capital_halt', lambda *a:(halt, {'proof':'sha'}))
    source = dict(verified_halts=[], source_sha256={}, overrides={'old':'unchanged'})
    before = deepcopy(source)
    runner = SimpleNamespace(read=lambda p:source)
    install_capital_halt(runner, DOCUMENT, ROOT)
    for _ in range(2):
        result = runner.read(ROOT/'docs/stock_universe_2019_corporate_terms.json')
        assert result['verified_halts'] == [halt]
        assert result['source_sha256'] == {'proof':'sha'}
        assert result['overrides'] == before['overrides']
    assert runner.read(ROOT/'other.json') is source
    assert source == before


def test_production_notice_has_exact_preannounced_dates_when_local_evidence_exists():
    if not (ROOT/'.cache/all-entry-accounts-20261011/corporate-evidence/2352').is_dir():
        pytest.skip('Local bound research evidence is not shipped in source checkout')
    halt, refs = load_capital_halt(ROOT, DOCUMENT)
    assert halt['start'] == '2025-09-25' and halt['end'] == '2025-10-07'
    assert halt['announcement_date'] < halt['start']
    assert refs[halt['source_path']] == halt['source_sha256']
