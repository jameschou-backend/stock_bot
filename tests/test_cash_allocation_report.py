"""Synthetic policy reports and provenance; never read real caches or fetch data."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import research_cash_allocation as driver
from scripts import prepare_cash_allocation_inputs as prep
from skills.million_replay import Replay
from skills.scenario_exit_replay import ScenarioExitReplay, ExitSignals


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
    data = driver.RunInputs(quotes, companies, days, entries, pd.DataFrame(), ExitSignals(adjusted, days), {},
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
        return dict(schema=1, mode_order=list(driver.MODES), stock_exit_mode='loss12',
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
    assert report['stock_exit_mode'] == report['exit_mode'] == 'loss12'
    assert all(report[key] is False for key in ('live_qualified', 'unseen_validation', 'auto_promote'))
    assert report['cases']['always_0050']['account'] == sandbox['data'].parent['strategy']
    assert report['benchmark']['account'] == sandbox['data'].parent['benchmark']
    assert not any(row['stock_id'] == '0050' for row in report['cases']['cash']['account']['trades'])
    summary = driver.read(output / 'summary.json')
    assert len(summary['comparisons']) == 3 and 'account' not in summary and 'cases' not in summary
    assert {row['mode'] for row in summary['annual']} == {*driver.MODES, 'benchmark'}
    for row in summary['comparisons']:
        assert sum(row['mean_' + key + '_weight'] for key in ('cash', 'etf', 'stock', 'receivable')) == pytest.approx(1)
    assert summary['performance']['offline_api_requests'] == 0
    meta = driver.verify_report(output)
    assert {'code.py', 'spec.md', 'inputs/manifest.json', 'inputs/execution-feeds/index.json',
            'inputs/dividends/1101.parquet', 'output/cases/trend_0050.json'} <= set(meta['verification_files_sha256'])
    assert 'output/manifest.json' not in meta['verification_files_sha256']
    assert driver.research(output, offline=True) == report
    with pytest.raises(ValueError, match='Sealed cash-allocation research exists'):
        driver.research(output)


def test_interrupted_case_resumes_only_completed_checkpoints(sandbox, monkeypatch):
    original = driver.run_case
    attempted, failed = [], False
    def interrupt(mode, data, feeds, corp):
        nonlocal failed
        attempted.append((mode, feeds.offline))
        if mode == 'trend_0050' and not feeds.offline and not failed:
            failed = True
            raise ValueError('Synthetic source outage')
        return original(mode, data, feeds, corp)
    monkeypatch.setattr(driver, 'run_case', interrupt)
    with pytest.raises(ValueError, match='outage'):
        driver.research(sandbox['output'])
    assert (sandbox['output'] / 'cases/cash.manifest.json').is_file()
    assert not (sandbox['output'] / 'manifest.json').exists()
    attempted.clear()
    driver.research(sandbox['output'])
    assert ('cash', False) not in attempted and ('cash', True) in attempted
    assert len(driver.read(sandbox['output'] / 'preparation_sessions.json')['sessions']) == 2


@pytest.mark.parametrize('mode', ['always_0050', 'benchmark'])
def test_control_and_benchmark_must_match_every_parent_account_field(sandbox, mode):
    data = deepcopy(sandbox['data'])
    data.parent['strategy' if mode == 'always_0050' else 'benchmark']['daily'][0]['nav'] += 1
    with pytest.raises(ValueError, match='exactly reproduce'):
        driver.run_case(mode, data, *sandbox['pair'](data, offline=True))


@pytest.mark.parametrize('name', ['spec.md', 'code.py', 'overrides.json'])
def test_modified_context_cannot_resume(sandbox, name):
    output = sandbox['output']; output.mkdir()
    driver.write(output / 'run.json', {'context': driver.verify_sources()})
    (sandbox['root'] / name).write_text('changed')
    with pytest.raises(ValueError, match='new explicit --output'):
        driver.research(output)


@pytest.mark.parametrize('field', ['allocation_decisions', 'exit_decisions', 'exit_states'])
def test_offline_any_policy_journal_mismatch_prevents_sealing(sandbox, monkeypatch, field):
    original = driver.run_case
    def mismatch(mode, data, feeds, corp):
        result = original(mode, data, feeds, corp)
        if mode == 'cash' and feeds.offline:
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
        if mode == 'cash':
            (sandbox['root'] / 'code.py').write_text('changed during execution')
        return case
    monkeypatch.setattr(driver, 'run_case', changed)
    with pytest.raises(ValueError, match='changed during execution'):
        driver.research(sandbox['output'])
    assert not (sandbox['output'] / 'cases/cash.manifest.json').exists()


def test_etf_dollar_attribution_counts_paid_and_pending_once_and_costs_in_cashflows():
    account = dict(settings={'initial_cash': 1000}, daily=[dict(date='2026-01-02', cash=300, market_value=720, receivable=20, nav=1040)],
        holdings=[dict(date='2026-01-02', stock_id='0050', market_value=200, price=20),
                  dict(date='2026-01-02', stock_id='1101', market_value=520, price=52)],
        trades=[dict(stock_id='0050', side='buy', cash_change=-205, total_cost=5),
                dict(stock_id='0050', side='sell', cash_change=55, total_cost=5)],
        cash_ledger=[dict(stock_id='0050', kind='dividend_payment', cash_change=7),
                     dict(stock_id='1101', kind='dividend_payment', cash_change=15)],
        receivables=[dict(stock_id='0050', kind='cash', amount=10), dict(stock_id='1101', kind='cash', amount=10)])
    result = driver.allocation_statistics(account)
    assert result['etf_net_pnl'] == 67  # -205 +55 +7 +200 +10
    assert result['stock_and_other_net_pnl'] == -27
    assert result['etf_total_cost'] == 10
    assert result['etf_trade_count'] == 2
    assert result['mean_cash_weight'] == pytest.approx(300 / 1040)
    assert result['attribution_is_selection_alpha'] is False
    account['daily'][0]['nav'] += 1
    with pytest.raises(ValueError, match='reconcile'):
        driver.allocation_statistics(account)


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
