import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.research_terminal_service import EvidenceError, ResearchTerminal


def put(root, name, value=None):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is not None:
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def terminal(tmp_path):
    dates = pd.to_datetime(['2026-09-29', '2026-09-30', '2026-10-01', '2026-10-02', '2026-10-05'])
    ids = ['0050', '1101', '2308']
    bundle = 'bundle'
    matrix = pd.DataFrame({'date': dates, **{sid: [20., 21., 22., 23., 1000.] for sid in ids}})
    (tmp_path/bundle).mkdir()
    matrix.to_parquet(tmp_path/bundle/'close-official.parquet', index=False)
    matrix.to_parquet(tmp_path/bundle/'close-quality.parquet', index=False)
    pd.DataFrame({'date': dates, **{sid: [True]*5 for sid in ids}}).to_parquet(
        tmp_path/bundle/'eligibility.parquet', index=False)
    quotes = pd.DataFrame([dict(date=d, stock_id=sid, open=10., high=12., low=9., close=10., volume=1000)
                           for d in dates for sid in ids])
    quotes.to_parquet(tmp_path/bundle/'quotes-unmasked.parquet', index=False)
    pd.DataFrame([dict(stock_id=sid, name='股票'+sid) for sid in ids]).to_parquet(
        tmp_path/bundle/'companies.parquet', index=False)
    manifest = dict(schema='poc_latest_input_bundle_v1', start='2026-09-29', end='2026-10-05',
                    files_sha256={p.name: put(tmp_path, str(p.relative_to(tmp_path)))['sha256']
                                  for p in (tmp_path/bundle).iterdir()})
    manifest_descriptor = put(tmp_path, bundle+'/manifest.json', manifest)
    profiles_descriptor = put(tmp_path, 'poc/profiles.json', [])
    poc_descriptor = put(tmp_path, 'poc/report.json', dict(end='2026-10-05', profiles=profiles_descriptor,
        source_sha256={manifest_descriptor['path']: manifest_descriptor['sha256']}))
    strategies = [dict(id='poc_up_red', name='POC 紅 K', kind='entry', status='active', family='poc', version='v1'),
                  dict(id='momentum', name='動能', kind='entry', status='active', family='momentum', version='v1'),
                  dict(id='rank', name='排序', kind='ranking', status='active', version='v1'),
                  dict(id='future', name='未實作', kind='entry', status='requires_data', version='v1')]
    days = []
    for i, d in enumerate(dates):
        stocks = []
        for sid in ids[1:]:
            stocks.append(dict(stock_id=sid, name='股票'+sid, regime='trend_up', results={
                'poc_up_red': dict(status='matched' if sid == '2308' else 'unknown',
                    first_signal=i == 4 if sid == '2308' else None, reasons=['理由'], metrics={'poc_after': 10.}, regime_fit=True),
                'momentum': dict(status='matched', first_signal=i == 0, reasons=['動能理由'], metrics={}, regime_fit=True),
                'rank': dict(status='matched', first_signal=False, reasons=['排序'], metrics={}, regime_fit=True)}))
        days.append(dict(date=str(d.date()), stocks=stocks, market_regime='trend_up', counts={}))
    scan = dict(schema='multi_strategy_scan_v1', source_end='2026-10-05', strategies=strategies,
                evaluated_strategy_ids=['poc_up_red', 'momentum', 'rank'], live_qualified=False,
                provenance=dict(source_hashes={'manifest.json': manifest_descriptor['sha256']},
                                poc={'report_sha256': poc_descriptor['sha256']}), days=days)
    scan_descriptor = put(tmp_path, 'scan/scan.json', scan)
    receipt_descriptor = put(tmp_path, 'scan/receipt.json', dict(files_sha256={'scan.json': scan_descriptor['sha256']}))
    audit = put(tmp_path, 'audit.json', dict(schema='strategy_scanner_daily_continuation_v1',
                artifacts=dict(scanner=receipt_descriptor, bundle=manifest_descriptor, poc=poc_descriptor)))
    study = dict(schema='strategy_scanner_signal_outcomes_v1', start='2024-01-02', end='2026-10-02',
        account_independent=True, live_qualified=False, horizons=[5, 20, 60],
        strategies=['poc_up_red', 'momentum'], strategy_definitions=strategies[:2], summary=[],
        signal_counts={'poc_up_red': {}, 'momentum': {}}, costs={'buy_fee': .001425},
        signal_policy='known_first_day_only_T_close', entry_price='T+1_adjusted_open_proxy',
        exit_price='T+h_adjusted_close_proxy_including_entry_session')
    study_descriptor = put(tmp_path, 'study/summary.json', study)
    rows = [dict(strategy_id='poc_up_red', signal_date='2025-01-02', stock_id='2308', horizon=20,
        entry_date='2025-01-03', exit_date='2025-02-03', status='evaluated', gross_return=.12,
        net_return=.1, benchmark_net_return=.05, excess_vs0050=.05),
        dict(strategy_id='poc_up_red', signal_date='2025-01-03', stock_id='1101', horizon=20,
        entry_date='2025-01-06', exit_date='2025-02-04', status='evaluated', gross_return=-.08,
        net_return=-.1, benchmark_net_return=.02, excess_vs0050=-.12),
        dict(strategy_id='poc_up_red', signal_date='2026-10-02', stock_id='2308', horizon=20,
        entry_date=None, exit_date=None, status='immature', gross_return=None,
        net_return=None, benchmark_net_return=None, excess_vs0050=None),
        dict(strategy_id='momentum', signal_date='2026-09-30', stock_id='2308', horizon=5,
        entry_date='2026-10-01', exit_date=None, status='immature', gross_return=None,
        net_return=None, benchmark_net_return=None, excess_vs0050=None)]
    pd.DataFrame(rows).to_parquet(tmp_path/'study/events.parquet', index=False)
    event_descriptor = put(tmp_path, 'study/events.parquet')
    study_receipt = put(tmp_path, 'study/receipt.json', dict(files_sha256={
        'summary.json': study_descriptor['sha256'], 'events.parquet': event_descriptor['sha256']}))
    study_audit = put(tmp_path, 'study-audit.json', dict(receipts={'study': study_receipt}))
    return ResearchTerminal(tmp_path, daily_audit=audit['path'], daily_sha=audit['sha256'],
                            study_audit=study_audit['path'], study_sha=study_audit['sha256'])


def test_overview_counts_entries_separately_from_rankings(terminal):
    x = terminal.overview()
    assert x['universe_count'] == 2
    assert x['latest']['entry_matches'] == 3
    assert x['entry_strategies'] == 2
    assert x['pending_strategies'] == 1
    assert x['qualification']['live_qualified'] is False


def test_first_signal_and_search_preserve_unknown_state(terminal):
    x = terminal.signals('2026-10-05', 'poc_up_red', True)
    assert x['total'] == 1
    assert x['rows'][0]['stock_id'] == '2308'
    assert terminal.signals('2026-10-02', 'poc_up_red', True)['total'] == 0
    assert terminal.signals('2026-10-05', 'all', False, '1101')['total'] == 1
    assert terminal.signals('2026-10-05', 'all', False, '1101')['rows'][0]['assessment']['unknown_entry_rules'] == ['POC 紅 K']


def test_chart_never_includes_future_day_or_signal_and_uses_adjusted_ohlc(terminal):
    x = terminal.stock('2308', '2026-10-02', 120)
    assert [c['date'] for c in x['candles']] == ['2026-09-29', '2026-09-30', '2026-10-01', '2026-10-02']
    assert x['candles'][-1]['close'] == 23.
    assert x['candles'][-1]['high'] == pytest.approx(27.6)
    assert x['candles'][-1]['volume'] == 1000
    assert max(m['date'] for m in x['markers']) == '2026-10-02'
    assert not any(m['first_signal'] is True and m['strategy_id'] == 'poc_up_red' for m in x['markers'])


def test_historical_chart_works_without_daily_snapshot_and_excludes_outcomes(terminal):
    terminal._load()
    terminal._days.pop('2026-09-30')
    x = terminal.stock('2308', '2026-09-30', 120)
    assert x['results'] == []
    assert x['assessment']['status'] == 'historical_reference'
    assert x['coverage']['daily_snapshot_available'] is False
    marker = next(m for m in x['markers'] if m['date'] == '2026-09-30')
    assert marker['strategy_id'] == 'momentum'
    assert marker['first_signal'] is True
    assert marker['coverage_type'] == 'known_first_event'
    assert not any('net_return' in m or 'exit_date' in m for m in x['markers'])
    assert max(m['date'] for m in x['markers']) <= '2026-09-30'


def test_daily_and_historical_markers_are_not_duplicated(terminal):
    x = terminal.stock('2308', '2026-10-02', 120)
    keys = [(m['date'], m['strategy_id']) for m in x['markers']]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize('method,args', [
    ('signals', ('2026-10-06',)), ('signals', ('2026-10-05', 'future')),
    ('stock', ('../secrets',)), ('stock', ('00631L',)), ('stock', ('2308', None, 501)),
    ('stock', ('2308', None, True)), ('signals', ('not-a-date',))])
def test_out_of_scope_requests_are_not_silently_substituted(terminal, method, args):
    with pytest.raises(ValueError):
        getattr(terminal, method)(*args)


def test_changed_sealed_scan_is_rejected(terminal):
    with (terminal.root/'scan/scan.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(EvidenceError, match='SHA256'):
        terminal.overview()


def test_cache_does_not_hide_source_changes(terminal):
    terminal.stock('2308')
    with (terminal.root/'bundle/quotes-unmasked.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(EvidenceError, match='變更'):
        terminal.stock('2308')


def test_poc_profiles_are_verified_before_accepting_screening(terminal):
    (terminal.root/'poc/profiles.json').write_text('[1]')
    with pytest.raises(EvidenceError, match='SHA256'):
        terminal.overview()


def test_signal_backtest_reaggregates_real_events_and_separates_immature(terminal):
    params = dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=20)
    job = terminal.backtest(params)
    x = job['result']
    assert x['stats']['events'] == 3
    assert x['stats']['evaluated'] == 2
    assert x['stats']['immature'] == 1
    assert x['stats']['win_rate'] == .5
    assert x['stats']['mean_net_return'] == 0.
    assert x['stats']['mean_excess_vs0050'] == pytest.approx(-.035)
    assert x['cumulative_return'] is None and x['max_drawdown'] is None
    assert x['live_qualified'] is False
    assert terminal.backtest(params)['cache_hit'] is True
    assert terminal.backtest_events(job['job_id'], 1, 1)['rows'][0]['stock_id'] == '1101'
    # Dates filter signal events, not the exit. This definition is explicit.
    one = terminal.backtest(dict(params, start='2025-01-02', end='2025-01-02'))
    assert one['result']['stats']['events'] == 1
    assert one['result']['stats']['mean_net_return'] == .1


@pytest.mark.parametrize('update', [dict(horizon=10), dict(start='2019-01-01'),
    dict(end='2026-10-05'), dict(strategy_id='rank'), dict(mode='made_up'),
    dict(start='2026-01-01', end='2025-01-01')])
def test_backtest_invalid_parameters_do_not_invent_results(terminal, update):
    params = dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=20)
    with pytest.raises(ValueError):
        terminal.backtest(dict(params, **update))


def test_changed_event_file_invalidates_cached_result(terminal):
    params = dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=20)
    job = terminal.backtest(params)
    with (terminal.root/'study/events.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(EvidenceError):
        terminal.get_backtest(job['job_id'])


def test_bad_job_identifier_cannot_read_arbitrary_file(terminal):
    with pytest.raises(ValueError):
        terminal.get_backtest('../../.env')


def test_absent_rally_study_is_explicit_not_synthetic(terminal):
    assert terminal.research()['status'] == 'not_available'
    assert terminal.research_episodes()['rows'] == []


def write_rally(terminal, *, manifest_sha=None):
    from app.research_terminal_service import RALLY_REPORT
    terminal._load()
    rows = [dict(stock_id='2308', anchor_date='2025-06-01', horizon=20, prior_signals=[]),
            dict(stock_id='1101', anchor_date='2026-06-01', horizon=60, prior_signals=[])]
    cases = put(terminal.root, 'rally/all-episodes.json', rows)
    cases['count'] = len(rows)
    report = dict(schema='rally_precursor_study_v1', live_qualified=False,
        source_provenance={'source_hashes': {'manifest.json': manifest_sha or terminal.source_hashes['manifest']}},
        all_episodes_artifact=cases)
    target = put(terminal.root, RALLY_REPORT, report)
    (terminal.root/RALLY_REPORT).with_suffix('.sha256').write_text(target['sha256'])


def test_rally_pages_include_cases_without_precursors_and_all_filters(terminal):
    write_rally(terminal)
    assert terminal.research()['status'] == 'ready'
    result = terminal.research_episodes('2308', '2025', 20)
    assert result['total'] == 1
    assert result['rows'][0]['prior_signals'] == []
    assert terminal.research_episodes(offset=1, limit=1)['rows'][0]['stock_id'] == '1101'
    assert terminal.research_episodes('1101', '2025')['total'] == 0


def test_rally_cannot_use_a_different_source_bundle(terminal):
    write_rally(terminal, manifest_sha='0'*64)
    with pytest.raises(EvidenceError, match='封存來源不一致'):
        terminal.research()


def test_rally_pagination_rechecks_source_after_cache(terminal):
    write_rally(terminal)
    terminal.research_episodes()
    (terminal.root/'rally/all-episodes.json').write_text('[]')
    with pytest.raises(EvidenceError):
        terminal.research_episodes()


@pytest.mark.parametrize('kwargs', [dict(stock_id='../secret'), dict(year='../../'),
    dict(horizon=5), dict(limit=201), dict(offset=-1)])
def test_rally_pagination_is_bounded(terminal, kwargs):
    with pytest.raises(ValueError):
        terminal.research_episodes(**kwargs)


def test_account_replay_delegates_only_fixed_verified_engine(monkeypatch, terminal):
    from app import workbench_jobs
    got = []
    def submit(request):
        got.append(request)
        return dict(job_id='a'*32, status='queued', message='ready')
    monkeypatch.setattr(workbench_jobs, 'submit', submit)
    result = terminal.backtest(dict(mode='account_replay', replay_mode='daily',
                                   replay_policy='mixed', replay_stress='control'))
    assert got[0].kind == 'verified_backtest'
    assert got[0].replay_preflight is False
    assert result['job_id'] == 'account_'+'a'*32
    assert result['scope']['start'] == '2022-01-03'


def test_overview_includes_supplement_catalog_counts_without_mixing_daily_scope(terminal):
    before = terminal.overview()

    class Supplement:
        def metadata(self):
            return dict(status='ready', source_end='2026-10-06', dates=['2026-10-06'])

        def catalog(self):
            return [dict(id=sid, name=sid, kind='entry', status='active', signal_only=True)
                    for sid in ('entry_contraction_narrow', 'entry_peer_narrow')]

    terminal._entry_context_provider = Supplement()
    after = terminal.overview()
    catalog = terminal.strategies()
    assert after['catalog_count'] == len(catalog['catalog']) == before['catalog_count'] + 2
    assert after['active_strategies'] == catalog['counts']['active'] == before['active_strategies'] + 2
    assert after['entry_strategies'] == before['entry_strategies'] + 2
    assert after['pending_strategies'] == before['pending_strategies']
    assert after['latest'] == before['latest']
    assert after['source_end'] == before['source_end'] == '2026-10-05'
    assert after['dates'] == before['dates']
    assert after['entry_context']['source_end'] == '2026-10-06'
