"""Synthetic policy reports and provenance; never read real caches or fetch data."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import research_technical as driver
from scripts import prepare_technical_inputs as prep
from skills.million_replay import Replay
from skills.scenario_exit_replay import ScenarioExitReplay
from skills.technical_signals import TechnicalSignals


class Feeds:
    def __init__(self, quotes, directory, offline):
        self.quotes = quotes.set_index(['date', 'stock_id'])
        self.days = sorted(quotes.date.unique())
        self.directory, self.offline = directory, offline

    def get_limits(self, sid):
        return {str(pd.Timestamp(day).date()): {'upper': 100_000., 'lower': .001} for day in self.days}

    def get_odd(self, day, sid, market):
        price = float(self.quotes.loc[(pd.Timestamp(day), sid), 'close'])
        return dict(odd_shares=100_000, odd_last=price, odd_bid=price-.01,
                    odd_ask=price+.01, bid_qty=10_000, ask_qty=10_000)

    def manifest(self):
        path = self.directory / 'index.json'
        value = driver.read(path)
        for name, digest in value['files_sha256'].items():
            if driver.sha(self.directory / name) != digest:
                raise ValueError('Synthetic execution evidence changed')
        return dict(value, manifest_sha256=driver.sha(path), cache_directory=str(self.directory.resolve()))


class Corporate:
    def __init__(self, directory, offline):
        self.directory, self.offline = directory, offline
        self.requests, self.loaded, self.request_observer = 0, set(), None

    def prepare(self, sid):
        if not self.offline and sid not in self.loaded:
            self.requests += 1
            if self.request_observer:
                self.request_observer(self.requests)
        self.loaded.add(sid)

    def on_date(self, sid, day):
        return []

    def manifest(self):
        return dict(files_sha256={'1101.parquet': driver.sha(self.directory / '1101.parquet')},
                    overrides={}, requests=self.requests)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    child = tmp_path / 'inputs'
    feeds, dividends = child / 'execution-feeds', child / 'dividends'
    feeds.mkdir(parents=True); dividends.mkdir()
    (feeds / 'source.json').write_text('{}')
    (dividends / '1101.parquet').write_bytes(b'synthetic dividend evidence')
    index = dict(schema=1, parser_sha256='synthetic', entries={'one': 'source'},
        files_sha256={'source.json': driver.sha(feeds / 'source.json')}, limitations=['daily only'],
        request_counters={'official_http_requests': 10, 'finmind_requests': 2})
    driver.write(feeds / 'index.json', index)
    driver.write(child / 'manifest.json', dict(seed_files_sha256={
        'execution-feeds/source.json': driver.sha(feeds / 'source.json'),
        'dividends/1101.parquet': driver.sha(dividends / '1101.parquet')},
        seed_execution_index=index, seed_dividend_hashes={'1101.parquet': driver.sha(dividends / '1101.parquet')}))
    (tmp_path / 'spec.md').write_text('frozen synthetic specification')
    (tmp_path / 'code.py').write_text('# synthetic code')
    driver.write(tmp_path / 'overrides.json', {'overrides': {}})
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'INPUT', child)
    monkeypatch.setattr(driver, 'PARENT', tmp_path / 'parent')
    monkeypatch.setattr(driver, 'OVERRIDES', tmp_path / 'overrides.json')
    days = pd.bdate_range('2021-01-04', periods=210)
    adjusted = pd.DataFrame({'0050': 100.+np.arange(len(days))*.1, '1101': 100.}, index=days)
    adjusted.loc[days[132]:, '1101'] = 85.
    adjusted.loc[days[160]:days[180], '0050'] = 90.
    quotes = pd.DataFrame([dict(date=day, stock_id=sid, open=price, high=price+1,
        low=price-1, close=price, volume=2_000_000) for day in days for sid, price in [('0050', 100.), ('1101', 50.)]])
    companies = pd.DataFrame([dict(stock_id='1101', name='Synthetic', market='TWSE')])
    entries = [dict(event_id='first', members=['1101'], priority=.1,
        signal_date=str(days[129].date()), entry_date=str(days[130].date()))]
    data = driver.RunInputs(quotes, companies, days, entries, pd.DataFrame(), TechnicalSignals(adjusted, quotes, days), {},
                            str(days[129].date()), str(days[-1].date()))
    calls = []
    def pair(data, *, offline):
        calls.append(offline)
        return Feeds(data.quotes, feeds, offline), Corporate(dividends, offline)
    args = (data.quotes, data.companies, data.days, data.entries)
    data.parent['benchmark'] = Replay(*args, *pair(data, offline=True), start=data.start, end=data.end, benchmark=True).run()
    data.parent['strategy'] = ScenarioExitReplay(*args, *pair(data, offline=True), start=data.start, end=data.end,
        exit_signals=data.features, mode='loss12').run()
    calls.clear()
    def context():
        seed = driver.read(child / 'manifest.json')
        for name, digest in seed['seed_files_sha256'].items():
            if driver.sha(child / name) != digest:
                raise ValueError('Synthetic seed changed')
        return dict(schema=1, mode_order=list(driver.MODES), research_kind='technical', control_exit_mode='loss12',
            source_files_sha256={name: driver.sha(tmp_path / name) for name in ('spec.md', 'overrides.json', 'inputs/manifest.json')},
            code_sha256={'code.py': driver.sha(tmp_path / 'code.py')}, runtime_versions={'synthetic': '1'},
            child_cache_directory=str(child))
    monkeypatch.setattr(driver, 'prepare_inputs', lambda **kwargs: None)
    monkeypatch.setattr(driver, 'verify_sources', context)
    monkeypatch.setattr(driver, 'load_inputs', lambda: data)
    monkeypatch.setattr(driver, 'provider_pair', pair)
    monkeypatch.setattr(driver, 'ReplayMarketFeeds', lambda *args, **kwargs: Feeds(quotes, feeds, True))
    return dict(root=tmp_path, input=child, output=tmp_path / 'output', data=data, pair=pair, calls=calls)


def test_all_policies_seal_then_identically_reproduce_offline(sandbox):
    output = sandbox['output']
    report = driver.research(output)
    assert list(report['cases']) == list(driver.MODES)
    assert report['research_kind'] == 'technical' and report['control_exit_mode'] == 'loss12'
    assert all(report[key] is False for key in ('live_qualified', 'unseen_validation', 'auto_promote'))
    assert report['cases']['control']['account'] == sandbox['data'].parent['strategy']
    assert report['benchmark']['account'] == sandbox['data'].parent['benchmark']
    summary = driver.read(output / 'summary.json')
    assert len(summary['comparisons']) == 6 and 'account' not in summary and 'cases' not in summary
    assert {row['mode'] for row in summary['annual']} == {*driver.MODES, 'benchmark'}
    for row in summary['comparisons']:
        assert sum(row['mean_' + key + '_weight'] for key in ('cash', 'etf', 'stock', 'receivable')) == pytest.approx(1)
    assert summary['performance']['offline_api_requests'] == 0
    meta = driver.verify_report(output)
    assert {'code.py', 'spec.md', 'inputs/manifest.json', 'inputs/execution-feeds/index.json',
            'inputs/dividends/1101.parquet', 'output/cases/support_risk2_pattern.json'} <= set(meta['verification_files_sha256'])
    assert 'output/manifest.json' not in meta['verification_files_sha256']
    assert driver.research(output, offline=True) == report
    with pytest.raises(ValueError, match='Sealed technical research exists'):
        driver.research(output)


def test_interrupted_case_resumes_only_completed_checkpoints(sandbox, monkeypatch):
    original = driver.run_case
    attempted, failed = [], False
    def interrupt(mode, data, feeds, corp):
        nonlocal failed
        attempted.append((mode, feeds.offline))
        if mode == 'support_risk2_pattern' and not feeds.offline and not failed:
            failed = True
            raise ValueError('Synthetic source outage')
        return original(mode, data, feeds, corp)
    monkeypatch.setattr(driver, 'run_case', interrupt)
    with pytest.raises(ValueError, match='outage'):
        driver.research(sandbox['output'])
    assert (sandbox['output'] / 'cases/risk2.manifest.json').is_file()
    assert not (sandbox['output'] / 'manifest.json').exists()
    attempted.clear()
    driver.research(sandbox['output'])
    assert ('risk2', False) not in attempted and ('risk2', True) in attempted
    assert len(driver.read(sandbox['output'] / 'preparation_sessions.json')['sessions']) == 2


@pytest.mark.parametrize('mode', ['control', 'benchmark'])
def test_control_and_benchmark_must_match_every_parent_account_field(sandbox, mode):
    data = deepcopy(sandbox['data'])
    data.parent['strategy' if mode == 'control' else 'benchmark']['daily'][0]['nav'] += 1
    with pytest.raises(ValueError, match='exactly reproduce'):
        driver.run_case(mode, data, *sandbox['pair'](data, offline=True))


@pytest.mark.parametrize('name', ['spec.md', 'code.py', 'overrides.json'])
def test_modified_context_cannot_resume(sandbox, name):
    output = sandbox['output']; output.mkdir()
    driver.write(output / 'run.json', {'context': driver.verify_sources()})
    (sandbox['root'] / name).write_text('changed')
    with pytest.raises(ValueError, match='new explicit --output'):
        driver.research(output)


@pytest.mark.parametrize('field', ['sizing_decisions', 'add_decisions', 'pattern_decisions', 'exit_decisions', 'exit_states', 'summary', 'account'])
def test_offline_any_policy_journal_mismatch_prevents_sealing(sandbox, monkeypatch, field):
    original = driver.run_case
    def mismatch(mode, data, feeds, corp):
        result = original(mode, data, feeds, corp)
        if mode == 'risk2' and feeds.offline:
            result[field] = {'tampered': True}
        return result
    monkeypatch.setattr(driver, 'run_case', mismatch)
    with pytest.raises(ValueError, match='Offline account/decision/state'):
        driver.research(sandbox['output'])
    assert not (sandbox['output'] / 'manifest.json').exists()


@pytest.mark.parametrize('changed', ['output', 'code', 'seed', 'new_dividend', 'index', 'inventory'])
def test_sealed_full_source_closure_detects_tampering(sandbox, changed):
    output = sandbox['output']
    driver.research(output)
    if changed == 'output':
        (output / 'summary.json').write_text('{}')
    elif changed == 'code':
        (sandbox['root'] / 'code.py').write_text('different')
    elif changed == 'seed':
        (sandbox['input'] / 'dividends/1101.parquet').write_bytes(b'changed')
    elif changed == 'new_dividend':
        (sandbox['input'] / 'dividends/1102.parquet').write_bytes(b'new')
    elif changed == 'index':
        path = sandbox['input'] / 'execution-feeds/index.json'
        path.write_text(path.read_text() + ' ')
    else:
        meta = driver.read(output / 'manifest.json')
        del meta['verification_files_sha256']['code.py']
        driver.write(output / 'manifest.json', meta)
    with pytest.raises(ValueError):
        driver.verify_report(output)


def test_source_change_during_case_stops_before_checkpoint(sandbox, monkeypatch):
    original = driver.run_case
    def changed(mode, *args):
        case = original(mode, *args)
        if mode == 'risk2':
            (sandbox['root'] / 'code.py').write_text('changed during execution')
        return case
    monkeypatch.setattr(driver, 'run_case', changed)
    with pytest.raises(ValueError, match='changed during execution'):
        driver.research(sandbox['output'])
    assert not (sandbox['output'] / 'cases/risk2.manifest.json').exists()


def test_add_and_pattern_compare_with_frozen_combination_not_control(sandbox):
    report = driver.research(sandbox['output'])
    summaries = {name: case['summary'] for name, case in report['cases'].items()}
    for row in report['summary']['comparisons']:
        base = 'support_risk2' if row['mode'] in ('support_risk2_add', 'support_risk2_pattern') else 'control'
        assert row['comparison_base_mode'] == base
        assert row['excess_total_return_vs_base'] == pytest.approx(
            row['total_return'] - summaries[base]['total_return'])
        assert row['drawdown_difference_vs_base'] == pytest.approx(
            row['max_drawdown'] - summaries[base]['max_drawdown'])
    assert report['summary']['comparison_groups'] == {
        'support_and_sizing': ['control', 'support20', 'risk2', 'support_risk2'],
        'adding': ['support_risk2', 'support_risk2_add'],
        'pattern': ['support_risk2', 'support_risk2_pattern']}


def test_technical_statistics_counts_decisions_without_inventing_fills():
    class Engine:
        sizing_decisions = [{'reason': 'risk_budget', 'requested_qty': 100}, {'reason': 'invalid_risk'}]
        add_decisions = [{'action': 'request_add', 'filled_qty': 0}]
        pattern_decisions = [{'reason': 'contraction_missing'}, {'reason': 'breakout_missing'}]
        exit_decisions = []
    result = driver.technical_statistics(Engine())
    assert result['sizing_decision_count'] == 2
    assert result['sizing_reason_counts'] == {'risk_budget': 1, 'invalid_risk': 1}
    assert result['add_decision_count'] == 1
    assert result['add_action_counts'] == {'request_add': 1}
    assert 'add_fill_count' not in result


def test_output_cannot_alias_parent_input_or_symlink(sandbox):
    for path in (sandbox['input'], sandbox['root'] / 'parent', sandbox['root']):
        with pytest.raises(ValueError, match='overlaps'):
            driver.research(path)
    linked = sandbox['root'] / 'linked'
    linked.symlink_to(sandbox['output'], target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        driver.research(linked)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    root = tmp_path
    parent_input = root / prep.PARENT_INPUT
    feeds, dividends = parent_input / 'execution-feeds', parent_input / 'dividends'
    feeds.mkdir(parents=True); dividends.mkdir()
    (feeds / 'raw.json').write_text('{}')
    (feeds / 'rows.parquet').write_bytes(b'rows')
    (dividends / '1101.parquet').write_bytes(b'dividends')
    matrix = root / 'matrix.parquet'; matrix.write_bytes(b'large immutable matrix')
    index = dict(schema=1, parser_sha256='parser', limitations=['daily'],
        entries={'entry': dict(raw_file='raw.json', rows_file='rows.parquet')},
        files_sha256={name: driver.sha(feeds / name) for name in ('raw.json', 'rows.parquet')},
        request_counters={'finmind_requests': 12, 'official_http_requests': 90})
    driver.write(feeds / 'index.json', index)
    driver.write(parent_input / 'manifest.json', {'references': {'quotes': {'path': 'matrix.parquet', 'sha256': driver.sha(matrix)}}})
    inventory = {str(path.relative_to(root)): driver.sha(path)
        for path in [matrix, *feeds.iterdir(), *dividends.iterdir(), parent_input / 'manifest.json']}
    parent_manifest = root / prep.PARENT / 'manifest.json'
    driver.write(parent_manifest, {'verification_files_sha256': inventory})
    calls = []
    def verified(path):
        calls.append(path)
        meta = driver.read(parent_manifest)
        for name, digest in meta['verification_files_sha256'].items():
            if driver.sha(root / name) != digest:
                raise ValueError('Parent sealed source changed')
        return meta
    monkeypatch.setattr(prep.parent, 'verify_report', verified)
    return dict(root=root, destination=root / 'child', parent_input=parent_input, calls=calls, index=index)


def test_preparation_is_independent_and_idempotent_without_network(preparation):
    p = preparation
    result = prep.prepare(root=p['root'], destination=p['destination'])
    assert result['finmind_requests'] == result['official_http_requests'] == 0
    assert len(p['calls']) == 2  # before copy and mutation boundary
    for child, original in result['clone_parent_paths'].items():
        assert not os.path.samefile(p['destination'] / child, p['root'] / original)
        assert driver.sha(p['destination'] / child) == driver.sha(p['root'] / original)
    assert not os.path.samefile(p['destination'] / 'execution-feeds/index.json', p['parent_input'] / 'execution-feeds/index.json')
    assert not (p['destination'] / 'matrix.parquet').exists()
    assert prep.prepare(root=p['root'], destination=p['destination']) == result


@pytest.mark.parametrize('mutation', ['seed', 'hardlink', 'index_hardlink', 'counter', 'parent', 'reference', 'preparer', 'new_bad_dividend'])
def test_preparation_rejects_alias_or_changed_inherited_source(preparation, mutation):
    p = preparation
    prep.prepare(root=p['root'], destination=p['destination'])
    child = p['destination']
    if mutation == 'seed':
        (child / 'dividends/1101.parquet').write_bytes(b'changed')
    elif mutation in ('hardlink', 'index_hardlink'):
        name = 'dividends/1101.parquet' if mutation == 'hardlink' else 'execution-feeds/index.json'
        (child / name).unlink(); os.link(p['parent_input'] / name, child / name)
    elif mutation == 'counter':
        index = deepcopy(p['index']); index['request_counters']['finmind_requests'] -= 1
        driver.write(child / 'execution-feeds/index.json', index)
    elif mutation == 'parent':
        (p['root'] / 'matrix.parquet').write_bytes(b'changed')
    elif mutation == 'new_bad_dividend':
        (child / 'dividends/invalid.txt').write_text('bad')
    else:
        data = driver.read(child / 'manifest.json')
        if mutation == 'reference':
            data['references']['quotes']['path'] = 'other.parquet'
        else:
            data['preparation_code_sha256'] = 'changed'
        driver.write(child / 'manifest.json', data)
    with pytest.raises(ValueError):
        prep.verify(child, root=p['root'])


def test_preparation_accepts_only_append_only_child_sources(preparation):
    p = preparation
    prep.prepare(root=p['root'], destination=p['destination'])
    child = p['destination']; index = deepcopy(p['index'])
    (child / 'execution-feeds/new.json').write_text('{}')
    index['files_sha256']['new.json'] = driver.sha(child / 'execution-feeds/new.json')
    index['entries']['new'] = dict(raw_file='new.json', rows_file='rows.parquet')
    index['request_counters']['official_http_requests'] += 1
    driver.write(child / 'execution-feeds/index.json', index)
    (child / 'dividends/1102.parquet').write_bytes(b'new independent evidence')
    prep.verify(child, root=p['root'])
    assert driver.read(p['parent_input'] / 'execution-feeds/index.json') == p['index']


def test_candidate_audit_counts_entire_pool_before_slot_and_pattern_gates(sandbox):
    data = deepcopy(sandbox['data'])
    data.entries.append(dict(data.entries[0], event_id='overlapping-candidate'))
    audit = driver.candidate_feature_audit(data)
    assert audit['summary']['candidate_count'] == 2
    assert len(audit['rows']) == 2
    assert audit['counts_are_orders_or_fills'] is False
    assert audit['summary']['pattern_pass_false_count'] == 2
    assert audit['rows'][0]['support_available'] is True
    assert audit['rows'][0]['risk_available'] is True
    assert audit['rows'][0]['signal_date'] == data.entries[0]['signal_date']
    assert audit['units']['raw_volume'] == 'shares'
    data.entries[0]['signal_date'] = data.entries[0]['entry_date']
    with pytest.raises(ValueError, match='original prior signal date'):
        driver.candidate_feature_audit(data)


def test_candidate_audit_is_sealed_and_reproduced_before_account_runs(sandbox):
    report = driver.research(sandbox['output'])
    assert driver.read(sandbox['output'] / 'candidate_features.json') == report['candidate_features']
    assert report['summary']['candidate_feature_summary']['candidate_count'] == 1
    altered = deepcopy(report)
    altered['candidate_features']['rows'][0]['support20'] += 1
    with pytest.raises(ValueError, match='candidate feature audit reproduction'):
        driver._offline_all(sandbox['data'], altered)
    (sandbox['output'] / 'candidate_features.json').write_text('{}')
    with pytest.raises(ValueError, match='output hash changed'):
        driver.verify_report(sandbox['output'])


@pytest.mark.parametrize('mutation', ['planned_risk', 'risk_cap', 'signal_date', 'filled_qty'])
def test_technical_audit_rejects_risk_lag_and_fill_mismatch(mutation):
    from types import SimpleNamespace
    days = pd.bdate_range('2026-01-05', periods=3)
    row = dict(date=str(days[1].date()), signal_date=str(days[0].date()), event_id='e', stock_id='1101',
        requested_qty=100, filled_qty=0, planned_risk=19_999., risk_cap=20_000., prior_nav=1_000_000.)
    engine = SimpleNamespace(sizing_decisions=[row], add_decisions=[], pattern_decisions=[], exit_decisions=[], exit_states={})
    valid = driver.audit_technical(engine, {'trades': []}, days, 'risk2')
    assert valid['risk_checked_positive_requests'] == 1
    assert valid['actual_loss_is_capped'] is False
    if mutation == 'planned_risk':
        row['planned_risk'] = 20_000.01
    elif mutation == 'risk_cap':
        row['risk_cap'] = 20_001.
    elif mutation == 'signal_date':
        row['signal_date'] = row['date']
    else:
        row['filled_qty'] = 1
    with pytest.raises(ValueError):
        driver.audit_technical(engine, {'trades': []}, days, 'risk2')


def test_support_only_does_not_claim_two_percent_risk_and_ratchet_cannot_fall():
    from types import SimpleNamespace
    days = pd.bdate_range('2026-01-05', periods=3)
    size = dict(date=str(days[1].date()), signal_date=str(days[0].date()), event_id='e', stock_id='1101',
        requested_qty=100, filled_qty=0, planned_risk=40_000., risk_cap=20_000., prior_nav=1_000_000.)
    engine = SimpleNamespace(sizing_decisions=[size], add_decisions=[], pattern_decisions=[], exit_decisions=[], exit_states={})
    assert driver.audit_technical(engine, {'trades': []}, days, 'support20')['risk_checked_positive_requests'] == 0
    engine.exit_decisions = [dict(date=str(days[i].date()), signal_date=str(days[i-1].date()),
        event_id='e', support_floor=101.-i) for i in (1, 2)]
    engine.exit_states = {'e': {'support_floor': 99.}}
    with pytest.raises(ValueError, match='moved down'):
        driver.audit_technical(engine, {'trades': []}, days, 'support20')


def test_only_one_successful_add_per_cohort_and_attempt_counts_exclude_checks():
    from types import SimpleNamespace
    days = pd.bdate_range('2026-01-05', periods=3)
    rows = [dict(date=str(days[i].date()), signal_date=str(days[i-1].date()), event_id='e', stock_id='1101',
        requested_qty=1, filled_qty=1, planned_risk=10., risk_cap=20_000., prior_nav=1_000_000.) for i in (1, 2)]
    engine = SimpleNamespace(sizing_decisions=[], add_decisions=rows, pattern_decisions=[], exit_decisions=[], exit_states={})
    trades = [dict(date=row['date'], event_id='e', stock_id='1101', side='buy', reason='pyramid_add', qty=1, total_cost=20.) for row in rows]
    with pytest.raises(ValueError, match='more than one successful add'):
        driver.audit_technical(engine, {'trades': trades}, days, 'support_risk2_add')
    engine.add_decisions = [rows[0], dict(rows[1], requested_qty=0, filled_qty=0, failure='add_gain_below_10pct')]
    result = driver.technical_statistics(engine, {'trades': trades[:1]})
    assert result['add_decision_count'] == 2
    assert result['add_attempt_count'] == result['add_successful_cohorts'] == result['add_filled_trade_count'] == 1
    assert result['add_stock_trade_cost'] == 20.


def test_fee_funded_exit_metadata_counts_actual_nonpositive_sales_without_double_counting_costs():
    from types import SimpleNamespace
    account = {'trades': [
        {'side': 'sell', 'reason': 'loss12', 'cash_change': -7.5, 'total_cost': 20.},
        {'side': 'sell', 'reason': 'support20', 'cash_change': 0., 'total_cost': 20.},
        {'side': 'sell', 'reason': 'loss12', 'cash_change': 100., 'total_cost': 20.},
        {'side': 'buy', 'reason': 'leader_entry', 'cash_change': -120., 'total_cost': 20.},
    ]}
    result = driver.technical_statistics(SimpleNamespace(), account)
    assert result['nonpositive_proceeds_exit_trade_count'] == 2
    assert result['fee_funded_exit_trade_count'] == 1
    assert result['zero_proceeds_exit_trade_count'] == 1
    assert result['fee_funded_exit_cash_paid'] == 7.5
    assert account['trades'][0]['total_cost'] == 20.


def test_corporate_source_document_is_part_of_context_and_cannot_change_silently(tmp_path, monkeypatch):
    child = tmp_path / 'inputs'; child.mkdir()
    for path in (child / 'manifest.json', tmp_path / 'spec.md', tmp_path / 'overrides.json',
                 tmp_path / 'corporate-source.md', tmp_path / 'code.py', tmp_path / 'parent.txt'):
        path.write_text('source')
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'INPUT', child)
    monkeypatch.setattr(driver, 'SPEC', tmp_path / 'spec.md')
    monkeypatch.setattr(driver, 'OVERRIDES', tmp_path / 'overrides.json')
    monkeypatch.setattr(driver, 'SOURCE_DOCS', (tmp_path / 'corporate-source.md',))
    monkeypatch.setattr(driver, 'CODE', ('code.py',))
    monkeypatch.setattr(driver, 'verify_inputs', lambda path: {
        'parent_files_sha256': {'parent.txt': driver.sha(tmp_path / 'parent.txt')}})
    first = driver.verify_sources()
    assert first['source_files_sha256']['corporate-source.md'] == driver.sha(tmp_path / 'corporate-source.md')
    (tmp_path / 'corporate-source.md').write_text('changed')
    assert driver.verify_sources() != first
