from collections import Counter
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import prepare_scanner_daily_poc as mod


def action_report(end=mod.END):
    return dict(start='2026-10-05', end=end, corporate_events_extension_complete=True,
                corporate_action_coverage=[dict(kind=market + '_' + kind, complete=True,
                    start='2026-10-05', end=end) for market in ('twse', 'tpex')
                    for kind in ('ex_rights', 'capital_reduction', 'par_value_change')])


def manifest():
    return dict(events_extension_complete=True, event_extension_report='report.json',
                source_sha256={'report.json': 'hash'})


def test_action_calendar_requires_complete_bound_current_interval():
    assert mod.corporate_coverage_reason(manifest(), lambda _: action_report()) is None
    assert mod.corporate_coverage_reason({}, lambda _: pytest.fail('Must not read unbound source')) == 'corporate_event_coverage_incomplete'
    changed = manifest(); changed['source_sha256'] = {}
    assert mod.corporate_coverage_reason(changed, lambda _: pytest.fail('Must not read unbound source')) == 'corporate_event_evidence_unbound'
    assert mod.corporate_coverage_reason(manifest(), lambda _: action_report('2026-10-02')) == 'corporate_event_interval_incomplete'
    report = action_report(); report['corporate_action_coverage'].pop()
    assert mod.corporate_coverage_reason(manifest(), lambda _: report) == 'corporate_event_interval_incomplete'


def test_terminal_t_plus_one_promotion_preserves_only_candidate_identity():
    row = dict(event_id='a', signal_date='2026-10-02', entry_date=None, members=['2330'], priority=1)
    promoted = dict(row, entry_date='2026-10-05')
    assert mod.candidate_identity(row) == mod.candidate_identity(promoted)
    assert mod.candidate_identity(row) != mod.candidate_identity(dict(promoted, priority=2))


def test_original_duplicate_candidates_are_rejected():
    class Reader:
        def read_json(self, path):
            if path == 'signals.json':
                return {'entries': {'median50m': [{'event_id': 'a'}]}}
            return {'entries': [{'event_id': 'a'}]}
    with pytest.raises(ValueError, match='Duplicate'):
        mod.candidate_rows(Reader())


def test_offline_fetch_is_explicitly_forbidden():
    obj = mod.CachedContinuation.__new__(mod.CachedContinuation)
    with pytest.raises(RuntimeError, match='cache-only'):
        obj._fetch('2330', '2026-10-02')


def test_missing_and_conflicting_tapes_do_not_become_poc_opportunities():
    calendar = pd.bdate_range(end=mod.END, periods=21).strftime('%Y-%m-%d').tolist()
    row = mod.base_row(dict(signal_id='new', stock_id='2330', signal_date=mod.END), calendar)
    result = mod.assess(row, {})
    assert result['status'] == 'pending_data' and result['available'] is False
    assert len(result['missing_dates']) == 20
    assert max(result['missing_dates']) == '2026-10-02'
    result = mod.assess(row, {'2330-' + calendar[0]: {'status': 'unknown', 'audit': {'conflict': True}}})
    assert result['status'] == 'unknown' and result['reason'] == 'ordinary_tape_conflict'


def test_extra_receipt_must_match_exact_query_and_raw_hash(tmp_path, monkeypatch):
    obj = mod.CachedContinuation.__new__(mod.CachedContinuation)
    obj.extra_ticks = tmp_path
    obj._mark = lambda *args: None
    path = tmp_path / 'receipts' / '2330-2026-10-02.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'query': {'dataset': 'TaiwanStockPriceTick', 'data_id': '2330', 'start_date': '2026-10-01'}, 'status': 'empty'}))
    with pytest.raises(ValueError, match='query changed'):
        obj._stored('2330', '2026-10-02')


def test_prefix_rejects_changed_old_prices(tmp_path):
    class Reader:
        def __init__(self, directory, end):
            self.directory = directory
            self.manifest = {'end': end}
        def verify(self, name):
            return self.directory / name
    base = tmp_path / 'base'; base.mkdir()
    new = tmp_path / 'new'; new.mkdir()
    pd.DataFrame({'date': [pd.Timestamp('2026-10-02')], '2330': [100.]}).to_parquet(base / 'close-official.parquet')
    pd.DataFrame({'date': [pd.Timestamp('2026-10-02'), pd.Timestamp('2026-10-05')], '2330': [101., 102.]}).to_parquet(new / 'close-official.parquet')
    with pytest.raises(ValueError, match='matrix prefix changed'):
        mod.verify_prefix(Reader(base, mod.OLD_END), Reader(new, mod.END))


def test_signal_day_corporate_actions_are_unknown_without_fetching(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    obj = mod.CachedContinuation.__new__(mod.CachedContinuation)
    obj.days = pd.bdate_range(end=mod.END, periods=21)
    obj.calendar = obj.days.strftime('%Y-%m-%d').tolist()
    obj.manifest = manifest()
    obj.signals = [dict(signal_id='s-' + sid, stock_id=sid, signal_date=mod.END)
                   for sid in ('2601', '5512', '2330')]
    obj.events = pd.DataFrame({'stock_id': ['2601', '5512'], 'event_date': pd.to_datetime([mod.END, mod.END])})
    obj.official = pd.DataFrame(index=pd.MultiIndex.from_product([['2601', '5512', '2330'], obj.calendar[:-1]]))
    obj.rows = {}; obj.structural = {}; obj.allowed = set()
    obj._path = lambda _: SimpleNamespace(close=np.full(21, 100.), raw_close=np.full(21, 100.))
    monkeypatch.setattr(mod, 'corporate_coverage_reason', lambda _: None)
    monkeypatch.setattr(mod, 'path_issue', lambda *args: None)
    obj._seed_continuation()
    for sid in ('2601', '5512'):
        assert obj.rows['s-' + sid]['reason'] == 'corporate_action_or_nonconstant_price_scale'
        assert not any(stock == sid for stock, _ in obj.allowed)
    assert len(obj.allowed) == 20
    assert {stock for stock, _ in obj.allowed} == {'2330'}


def test_runtime_algorithm_and_loader_modules_are_hash_bound():
    import inspect
    from scripts.scan_market_strategies import load_poc
    from skills.strategy_scanner.data import _Inputs
    refs = mod.run_code_refs()
    # Locate the functions actually invoked, so inherited algorithms cannot be
    # omitted merely because their constructor is intentionally bypassed.
    for implementation in (mod.assess, mod.base_row, mod.DailyProfiles.day,
                           mod.DailyProfiles.refresh, _Inputs, load_poc):
        path = Path(inspect.getsourcefile(implementation))
        key = str(path.relative_to(mod.ROOT))
        assert refs[key] == mod.digest(path)


def test_report_contains_actual_runtime_source_hashes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    code_refs = mod.run_code_refs()
    source_root = mod.ROOT
    for name in code_refs:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / name).read_bytes())
    old_profiles = tmp_path / 'old-profiles.json'
    old_profiles.write_text('[]')
    old_report = tmp_path / 'old-report.json'
    old_report.write_text(json.dumps({'profiles': {'path': 'old-profiles.json'}}))
    class Inputs:
        hashes = {'manifest.json': 'manifest-hash'}
        def __init__(self, *args): pass
    class Provider:
        refs = {}; rows = {}; days_cache = {}; unavailable = {}; manifest = {}
        def __init__(self, *args, **kwargs): pass
        def offline(self): pass
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'local', lambda p: Path(p).resolve())
    monkeypatch.setattr(mod, '_Inputs', Inputs)
    monkeypatch.setattr(mod, 'verify_prefix', lambda *args: None)
    monkeypatch.setattr(mod, 'candidate_rows', lambda *args: [])
    monkeypatch.setattr(mod, 'load_poc', lambda *args, **kwargs: ([], {'profiles_sha256': mod.digest(old_profiles)}))
    monkeypatch.setattr(mod, 'CachedContinuation', Provider)
    report = mod.run(SimpleNamespace(output=tmp_path / 'output', bundle=tmp_path / 'bundle',
                                    base_report=old_report, tick_directory=None))
    assert all(report['source_sha256'][key] == expected for key, expected in code_refs.items())
    saved = json.loads((tmp_path / 'output' / 'report.json').read_text())
    assert saved['source_sha256'] == report['source_sha256']
