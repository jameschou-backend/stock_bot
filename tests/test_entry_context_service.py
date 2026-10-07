"""Supplementary screening must not borrow old dates or future outcomes."""
import hashlib
import json

import pandas as pd
import pytest

from app.entry_context_service import EntryContextProvider, STRATEGY_IDS
from app.research_terminal_service import EvidenceError, ResearchTerminal


def put(root, path, value=None):
    target = root/path
    target.parent.mkdir(parents=True, exist_ok=True)
    if value is not None:
        target.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return dict(path=path, sha256=hashlib.sha256(target.read_bytes()).hexdigest())


@pytest.fixture
def provider(tmp_path):
    dates = pd.to_datetime(['2026-09-29', '2026-09-30', '2026-10-01', '2026-10-06'])
    ids = ['0050', '2395', '6505', '6187', '6213', '6257']
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    matrix = pd.DataFrame({'date': dates, **{sid: [20., 21., 22., 40.] for sid in ids}})
    for name in ('close-official', 'close-quality', 'raw-close', 'raw-volume'):
        matrix.to_parquet(bundle/(name+'.parquet'), index=False)
    pd.DataFrame({'date': dates, **{sid: [True]*4 for sid in ids}}).to_parquet(bundle/'eligibility.parquet', index=False)
    pd.DataFrame([dict(date=d, stock_id=s, open=10., high=12., low=9., close=10., volume=1000)
                  for d in dates for s in ids]).to_parquet(bundle/'quotes-unmasked.parquet', index=False)
    pd.DataFrame([dict(stock_id=s, name='股票'+s) for s in ids]).to_parquet(bundle/'companies.parquet', index=False)
    manifest = dict(schema='poc_latest_input_bundle_v1', start='2026-09-29', end='2026-10-06',
                    files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()})
    bundle_desc = put(tmp_path, 'bundle/manifest.json', manifest)
    contexts = [dict(date=str(d.date()), market_breadth_value=.55 if i == 3 else .4,
                     market_breadth_coverage=.9, valid60_stocks=600, eligible_stocks=667, market_narrow=i != 3, parent_candidates=[2, 1, 0, 3][i])
                for i, d in enumerate(dates)]
    events = []
    for i, sids in enumerate([['2395', '6505'], ['6505'], [], ['6187', '6213', '6257']]):
        for sid in sids:
            narrow = i != 3
            contraction = sid == '6505'
            peer = None if i == 1 else True
            events.append(dict(stock_id=sid, name='股票'+sid, signal_date=contexts[i]['date'],
                contraction=contraction, contraction_ratio=.5 if contraction else 1.2,
                market_narrow=narrow, market_breadth_value=contexts[i]['market_breadth_value'],
                market_breadth_coverage=.9, peer_breadth=peer, peer_breadth_value=.7 if peer else None,
                peer_issue=None if peer else 'fewer_than_three_correlated_peers',
                contraction_and_narrow=contraction and narrow,
                peer_and_narrow=None if peer is None else peer and narrow,
                group_cutoff_date='2026-08-31' if i < 3 else '2026-09-30',
                future_net_return=.99))  # Projection must never emit this field.
    report = dict(schema='entry60_current_signal_check_v1', start='2026-09-29', end='2026-10-06',
        parent='legacy_course_breakout', account_backtest=False, live_qualified=False,
        source_sha256={bundle_desc['path']: bundle_desc['sha256']}, all_first_signals=events)
    report_desc = put(tmp_path, 'report.json', report)
    publication = dict(schema='entry_context_terminal_v1', start='2026-09-29', end='2026-10-06',
        dates=[str(d.date()) for d in dates], day_context=contexts, live_qualified=False,
        artifacts=dict(report=report_desc, bundle=bundle_desc))
    descriptor = put(tmp_path, 'publication.json', publication)
    (tmp_path/'publication.sha256').write_text(descriptor['sha256'])
    return EntryContextProvider(tmp_path, descriptor)


def test_exact_latest_and_recent_rules(provider):
    assert provider.metadata()['source_end'] == '2026-10-06'
    for sid in STRATEGY_IDS:
        result = provider.signals('2026-10-06', sid)
        assert result['total'] == 0
        assert result['context_summary']['parent_candidates'] == 3
        assert result['context_summary']['not_matched'] == 3
        assert result['context_summary']['unknown'] == 0
        assert result['context_summary']['market_breadth_value'] == .55
    assert [r['stock_id'] for r in provider.signals('2026-09-29', STRATEGY_IDS[0])['rows']] == ['6505']
    assert [r['stock_id'] for r in provider.signals('2026-09-29', STRATEGY_IDS[1])['rows']] == ['2395', '6505']
    assert provider.signals('2026-09-29', STRATEGY_IDS[1], True, '6505')['total'] == 1
    assert all(row['signal_only'] for row in provider.catalog())


def test_zero_parent_events_are_distinct_from_unknown_candidates(provider):
    zero = provider.signals('2026-10-01', STRATEGY_IDS[1])
    assert zero['context_summary']['parent_candidates'] == 0
    assert zero['context_summary']['unknown'] == 0
    unknown = provider.signals('2026-09-30', STRATEGY_IDS[1])
    assert unknown['context_summary']['parent_candidates'] == 1
    assert unknown['context_summary']['unknown'] == 1
    assert unknown['total'] == 0
    assert unknown['context_summary']['checks'][0]['metrics']['peer_breadth'] is None


def test_chart_and_signals_never_emit_future_or_outcome_data(provider):
    chart = provider.stock('6505', '2026-09-29', strategy_id=STRATEGY_IDS[0])
    assert len(chart['candles']) == 1
    assert chart['candles'][0]['close'] == 20
    assert chart['candles'][0]['high'] == 24
    assert [m['date'] for m in chart['markers']] == ['2026-09-29']
    assert chart['results'][0]['status'] == 'matched'
    assert all(m['strategy_id'] == STRATEGY_IDS[0] for m in chart['markers'])
    assert 'future_net_return' not in json.dumps(chart)
    assert 'future_net_return' not in json.dumps(provider.signals('2026-09-29'))
    absent = provider.stock('6505', '2026-10-01', strategy_id=STRATEGY_IDS[1])
    assert absent['assessment']['status'] == 'no_parent_first_event'
    assert absent['results'] == []
    assert all(m['date'] <= '2026-10-01' for m in absent['markers'])


@pytest.mark.parametrize('filename', ['publication.json', 'publication.sha256', 'report.json',
                                      'bundle/manifest.json', 'bundle/close-official.parquet'])
def test_tampered_evidence_is_not_displayed(provider, filename):
    target = provider.evidence.root/filename
    with target.open('ab') as stream:
        stream.write(b' ')
    # Whitespace on the sidecar is harmless; replace its digest instead.
    if filename.endswith('.sha256'):
        target.write_text('0'*64)
    with pytest.raises(EvidenceError):
        provider.signals()


def test_chart_cache_cannot_hide_changed_source(provider):
    provider.stock('6505', '2026-09-29')
    with (provider.evidence.root/'bundle/quotes-unmasked.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(EvidenceError):
        provider.stock('6505', '2026-09-29')


def test_missing_supplement_is_explicit_without_ordinary_data_substitution(tmp_path):
    provider = EntryContextProvider(tmp_path)
    assert provider.metadata()['status'] == 'not_available'
    assert provider.metadata()['dates'] == []
    assert provider.catalog() == []
    with pytest.raises(EvidenceError):
        provider.signals()


def test_terminal_dispatch_does_not_load_older_ordinary_snapshot(provider):
    terminal = ResearchTerminal(provider.evidence.root, entry_context_descriptor=provider.descriptor)
    assert terminal.signals('2026-10-06', STRATEGY_IDS[0])['total'] == 0
    assert terminal.stock('6505', '2026-09-29', 120, STRATEGY_IDS[0])['markers']
    assert terminal._scan is None


@pytest.mark.parametrize('selected', ['2026-10-07', '2026-10-04', '2026-09-28', 'bad'])
def test_dates_cannot_silently_substitute_latest(provider, selected):
    with pytest.raises(ValueError):
        provider.signals(selected)


def reseal_report(provider, change):
    root = provider.evidence.root
    report = json.loads((root/'report.json').read_text())
    change(report)
    descriptor = put(root, 'report.json', report)
    publication = json.loads((root/'publication.json').read_text())
    publication['artifacts']['report'] = descriptor
    provider.descriptor = put(root, 'publication.json', publication)
    (root/'publication.sha256').write_text(provider.descriptor['sha256'])


@pytest.mark.parametrize('update', [
    {'contraction': True, 'contraction_and_narrow': True},
    {'peer_breadth': False, 'peer_and_narrow': False},
    {'group_cutoff_date': '2026-09-01'},
    {'contraction_and_narrow': None},
])
def test_resealed_invalid_rule_or_future_group_is_rejected(provider, update):
    reseal_report(provider, lambda r: r['all_first_signals'][0].update(update))
    with pytest.raises(EvidenceError):
        provider.signals()


def test_unknown_is_preserved_even_when_other_condition_false(provider):
    def change(report):
        report['all_first_signals'][-1].update(peer_breadth=None, peer_breadth_value=None,
            peer_issue='fewer_than_three_correlated_peers', peer_and_narrow=None)
    reseal_report(provider, change)
    result = provider.signals('2026-10-06', STRATEGY_IDS[1])
    assert result['context_summary']['matched'] == 0
    assert result['context_summary']['unknown'] == 1
    assert result['context_summary']['not_matched'] == 2


def test_removed_loaded_publication_does_not_disappear_silently(provider):
    provider.metadata()
    (provider.evidence.root/'publication.json').unlink()
    with pytest.raises(EvidenceError):
        provider.metadata()


def test_api_routes_strategy_scoped_chart(provider, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app import research_terminal_api as api
    from app.workbench_api import require_local
    terminal = ResearchTerminal(provider.evidence.root, entry_context_descriptor=provider.descriptor)
    monkeypatch.setattr(api, 'get_terminal', lambda: terminal)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[require_local] = lambda: None
    with TestClient(app) as client:
        result = client.get('/research-terminal/api/stocks/6505', params={
            'date': '2026-09-29', 'strategy_id': STRATEGY_IDS[0]})
        assert result.status_code == 200
        assert result.json()['strategy_id'] == STRATEGY_IDS[0]
        latest = client.get('/research-terminal/api/signals', params={
            'date': '2026-10-06', 'strategy_id': STRATEGY_IDS[1]})
        assert latest.status_code == 200
        assert latest.json()['context_summary']['parent_candidates'] == 3
