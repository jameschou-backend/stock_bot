import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import research_candle_volume_extension as cli
from skills import volume_profile_budget_extension as extension
from skills import volume_profile_data as parent


def bare_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(parent, 'ROOT', tmp_path)
    monkeypatch.setattr(parent, 'PILOT', tmp_path/'pilot')
    monkeypatch.setattr(parent, 'local', lambda p: p)
    import scripts.prepare_volume_profile as preparation
    monkeypatch.setattr(preparation, 'local', lambda p: Path(p) if Path(p).is_absolute() else tmp_path/p)
    provider = object.__new__(extension.ExtendedAccountProfileData)
    provider.directory = tmp_path/'profiles-v1'
    provider.directory.mkdir()
    provider.maximum, provider.online = extension.MAXIMUM, True
    provider._lock, provider._stop = threading.Lock(), threading.Event()
    provider._config = SimpleNamespace(finmind_token='test-only', finmind_requests_per_hour=6000)
    provider.refs, provider.reuse_index, provider.receipt_hashes = {}, {}, {}
    return provider


def install_fetch(monkeypatch, provider, *, fail=False):
    import app.finmind
    calls = []
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        sid, day = kwargs['data_id'], args[1].isoformat()
        assert (provider.directory/'attempts'/f'{sid}-{day}.json').exists()
        assert (provider.directory/'receipts'/f'{sid}-{day}.json').exists()
        if fail:
            raise app.finmind.FinMindError('fixture error')
        return pd.DataFrame({'stock_id': [sid], 'date': [day], 'price': [10.], 'volume': [1.]})
    monkeypatch.setattr(app.finmind, 'fetch_dataset', fetch)
    return calls


def test_financial_and_acquisition_methods_are_inherited_without_override():
    assert extension.ExtendedAccountProfileData._raw is parent.AccountProfileData._raw
    assert extension.ExtendedAccountProfileData.__call__ is parent.AccountProfileData.__call__
    assert extension.ExtendedAccountProfileData.snapshot is parent.AccountProfileData.snapshot
    assert extension.MAXIMUM == 8000 and extension.INITIAL_MAXIMUM == 4800


def test_cross_original_cap_first_unsent_request_preserves_attempts(tmp_path, monkeypatch):
    provider = bare_provider(tmp_path, monkeypatch)
    attempts = provider.directory/'attempts'; attempts.mkdir()
    for n in range(4800):
        (attempts/f'old-{n}.json').write_text('{}')
    calls = install_fetch(monkeypatch, provider)
    first = provider._raw(('2501', '2024-03-20'))
    assert first['status'] == 'received'
    assert len(list(attempts.glob('*.json'))) == 4801
    assert (attempts/'old-0.json').read_text() == '{}'
    assert provider._raw(('2501', '2024-03-20')) == first
    assert len(calls) == 1
    assert calls[0][1]['requests_per_hour'] == 5400
    assert calls[0][1]['max_retries'] == 0
    assert calls[0][1]['timeout'] == 40


def test_total_8000_cap_atomic_across_workers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    provider = bare_provider(tmp_path, monkeypatch)
    attempts = provider.directory/'attempts'; attempts.mkdir()
    for n in range(7999):
        (attempts/f'old-{n}.json').write_text('{}')
    calls = install_fetch(monkeypatch, provider)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(provider._raw, [('2501', '2024-03-20'), ('6535', '2024-01-02')]))
    assert sorted(r['status'] for r in results) == ['received', 'request_budget_or_quota_paused']
    assert len(calls) == 1 and len(list(attempts.glob('*.json'))) == 8000
    paused = next(r['query'] for r in results if r['status'] != 'received')
    key = paused['data_id']+'-'+paused['start_date']+'.json'
    assert not (attempts/key).exists()
    assert not (provider.directory/'receipts'/key).exists()


@pytest.mark.parametrize('status', ['started', 'provider_error', 'quota_paused', 'empty'])
def test_old_failure_receipts_are_never_retried(tmp_path, monkeypatch, status):
    provider = bare_provider(tmp_path, monkeypatch)
    calls = install_fetch(monkeypatch, provider)
    receipt = provider.directory/'receipts/2330-2024-01-02.json'
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(status=status, query=dict(
        dataset='TaiwanStockPriceTick', data_id='2330', start_date='2024-01-02'))))
    before = receipt.read_bytes()
    assert provider._raw(('2330', '2024-01-02'))['status'] == status
    assert not calls and receipt.read_bytes() == before


def test_orphan_and_current_quota_stop_are_not_reset(tmp_path, monkeypatch):
    provider = bare_provider(tmp_path, monkeypatch)
    calls = install_fetch(monkeypatch, provider)
    attempt = provider.directory/'attempts/2330-2024-01-02.json'
    attempt.parent.mkdir()
    attempt.write_text(json.dumps(dict(status='started', query=dict(
        dataset='TaiwanStockPriceTick', data_id='2330', start_date='2024-01-02'))))
    assert provider._raw(('2330', '2024-01-02'))['status'] == 'orphaned_started_attempt'
    provider._stop.set()
    assert provider._raw(('2501', '2024-03-20'))['status'] == 'request_budget_or_quota_paused'
    assert not calls


def test_new_provider_error_remains_nonretryable(tmp_path, monkeypatch):
    provider = bare_provider(tmp_path, monkeypatch)
    calls = install_fetch(monkeypatch, provider, fail=True)
    assert provider._raw(('2501', '2024-03-20'))['status'] == 'provider_error'
    assert provider._raw(('2501', '2024-03-20'))['status'] == 'provider_error'
    assert len(calls) == 1


def test_ledger_manifest_detects_deleted_or_modified_old_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(extension, 'INITIAL_MAXIMUM', 2)
    directory, evidence = tmp_path/'profiles-v1', tmp_path/'extension'
    attempts = directory/'attempts'; attempts.mkdir(parents=True)
    for key in ('old-a', 'old-b'):
        (attempts/(key+'.json')).write_text('{}')
    refs = extension.preserve_initial_ledger(directory, evidence, root=tmp_path)
    assert len(refs) == 3
    (attempts/'new.json').write_text('{}')
    assert extension.preserve_initial_ledger(directory, evidence, root=tmp_path) == refs
    (attempts/'old-a.json').unlink()
    with pytest.raises(ValueError, match='Original attempt or receipt changed'):
        extension.preserve_initial_ledger(directory, evidence, root=tmp_path)


def test_missing_initial_ledger_cannot_start_from_empty_directory(tmp_path):
    with pytest.raises(ValueError, match='exactly 4800'):
        extension.preserve_initial_ledger(tmp_path/'fresh', tmp_path/'evidence', root=tmp_path)


def test_constructor_fixed_directory_cap_and_immutable_source_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(extension, 'ROOT', tmp_path)
    monkeypatch.setattr(extension, 'DIRECTORY', tmp_path/'profiles-v1')
    monkeypatch.setattr(extension, 'EVIDENCE', tmp_path/'evidence')
    monkeypatch.setattr(extension, 'EXTENSION_SOURCES', ('new.py',))
    source = tmp_path/'new.py'; source.write_text('source bytes')
    refs = {'new.py': extension.digest(source)}
    monkeypatch.setattr(extension, 'extension_sources', lambda: dict(refs))
    monkeypatch.setattr(extension, 'preserve_initial_ledger', lambda *a: {})
    calls = []
    def initialize(self, bundle, **kwargs):
        calls.append((bundle, kwargs))
        self.directory = kwargs['directory']; self.directory.mkdir()
        self.maximum = kwargs['maximum_requests']; self.refs = {}
        self._stop = threading.Event(); self._stop.set()
    monkeypatch.setattr(parent.AccountProfileData, '__init__', initialize)
    def mark(self, path, expected=None):
        assert extension.digest(path) == expected
        self.refs[str(path.relative_to(tmp_path))] = expected
    monkeypatch.setattr(parent.AccountProfileData, '_mark', mark)
    provider = extension.ExtendedAccountProfileData(tmp_path/'bundle')
    assert calls[0][1] == dict(online=False, maximum_requests=4800, directory=tmp_path/'profiles-v1')
    assert provider.maximum == 8000 and provider._stop.is_set()
    assert len(provider.refs) == 2
    snapshot = next(tmp_path/p for p in provider.refs if 'source-snapshots' in p)
    assert snapshot.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='Immutable'):
        extension.immutable_bytes(snapshot, b'changed')


@pytest.mark.parametrize('arms', [(), ('invented',), ('original', 'original'),
    ('poc_red',), ('poc_dry',), ('poc_red_dry',)])
def test_cli_rejects_unregistered_duplicate_or_completed_replacement_arms(arms):
    with pytest.raises(ValueError):
        cli.validate_arms(arms)


@pytest.mark.parametrize('flag', ['profile_fetch', 'execution_fetch', 'odd_fetch'])
def test_anchor_replay_rejects_network_before_provider(tmp_path, flag):
    with pytest.raises(ValueError, match='fully offline'):
        cli.run_extension(tmp_path/'out', cli.ANCHOR_ARMS, **{flag: True})


def test_prior_4800_anchor_report_cannot_authorize_extension(tmp_path, monkeypatch):
    report = tmp_path/'report.json'
    report.write_text(json.dumps(dict(profile_data=dict(maximum_adapter_requests=4800))))
    monkeypatch.setattr(cli.sealed, 'require_anchor_report', lambda *a: {})
    with pytest.raises(ValueError, match='fixed 8000'):
        cli.require_extension_anchor_report(report, tmp_path)


def test_extension_anchor_requires_current_sources_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.sealed, 'require_anchor_report', lambda *a: {})
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'EVIDENCE', tmp_path/'evidence')
    monkeypatch.setattr(cli, 'EXTENSION_SOURCES', ('new.py',))
    ledger = tmp_path/'evidence/initial-attempt-ledger.json'
    ledger.parent.mkdir(); ledger.write_text('{}')
    ledger_sha = extension.digest(ledger)
    source = tmp_path/'new.py'; source.write_text('current')
    expected = extension.digest(source)
    monkeypatch.setattr(cli, 'extension_sources', lambda root: {'new.py': expected})
    name = '.cache/extension/source-snapshots/'+expected+'/new.py'
    snapshot = tmp_path/name; snapshot.parent.mkdir(parents=True); snapshot.write_bytes(source.read_bytes())
    report = tmp_path/'report.json'
    value = dict(profile_data=dict(maximum_adapter_requests=8000), source_sha256={
        'new.py': expected, name: expected, 'evidence/initial-attempt-ledger.json': ledger_sha})
    report.write_text(json.dumps(value))
    assert cli.require_extension_anchor_report(report, tmp_path)['new.py'] == expected
    snapshot.write_text('tampered')
    with pytest.raises(ValueError, match='snapshot changed'):
        cli.require_extension_anchor_report(report, tmp_path)
    value['source_sha256']['new.py'] = 'old'
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='anchor source changed'):
        cli.require_extension_anchor_report(report, tmp_path)
    ledger.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='initial ledger differs'):
        cli.require_extension_anchor_report(report, tmp_path)


def test_wrapper_injects_provider_without_financial_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    provider = SimpleNamespace(_mark=lambda *a: None)
    monkeypatch.setattr(cli, 'ExtendedAccountProfileData', lambda bundle, online: provider)
    monkeypatch.setattr(cli, 'require_extension_anchor_report', lambda path: {'anchor': 'hash'})
    calls = []
    monkeypatch.setattr(cli.sealed, 'run', lambda *args, **kwargs: calls.append((args, kwargs)) or {'all_completed': True})
    output = tmp_path/'.cache/red-volume-exit-20261003/test'
    result = cli.run_extension(output, cli.INCOMPLETE_ARMS, profile_fetch=True,
                               execution_fetch=True, odd_fetch=True, anchor_report=tmp_path/'anchors')
    assert result['all_completed']
    args, kwargs = calls[0]
    assert args == (output, cli.INCOMPLETE_ARMS, provider, cli.select_candidates)
    assert kwargs == dict(execution_fetch=True, limit_overlay=None, odd_fetch=True, anchor_report=tmp_path/'anchors')
