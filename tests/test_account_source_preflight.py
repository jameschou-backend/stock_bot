from copy import deepcopy

import pandas as pd
import pytest

from skills.account_source_preflight import (merge_sources, inventory, gate, digest,
    read, write, source_identity, CachedPreparationFeeds)
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from tests.test_replay_market_feeds import limit_record, odd_record, Response


def donor(path, price=110.):
    limits = limit_record()
    limits['data'][0]['limit_up'] = price
    feed = ReplayMarketFeeds(path/'execution-feeds', token='test', official_min_interval=0,
        finmind_fetch=lambda *a, **kw: pd.DataFrame(limits['data']),
        http_get=lambda *a, **kw: Response(odd_record()['payload']))
    feed.get_limits('0050')
    feed.get_odd('2022-01-04', '0050', 'twse')
    (path/'dividends').mkdir()
    pd.DataFrame([dict(stock_id='0050', CashEarningsDistribution=1.)]).to_parquet(path/'dividends/0050.parquet')
    return feed


def test_merge_reuses_identical_evidence_without_network_or_mutating_parents(tmp_path):
    first, second = tmp_path/'a', tmp_path/'b'
    donor(first); donor(second)
    before = source_identity(first)
    output = tmp_path/'new/inputs'
    result = merge_sources(output, [first, second], {'0050', '2330'})
    assert not result['conflicts'] and result['requests'] == 0
    assert source_identity(first) == before
    feeds = ReplayMarketFeeds(output/'execution-feeds', offline=True)
    assert feeds.get_limits('0050')['2022-01-04']['upper'] == 110
    assert feeds.get_odd('2022-01-04', '0050', 'twse')['odd_shares'] == 1200
    report = inventory(output, {'0050', '2330'})
    assert report['missing_limit_stocks'] == report['missing_dividend_stocks'] == ['2330']
    assert feeds.manifest()['request_counters']['finmind_requests'] == 0


def test_conflicting_donors_are_reported_and_cannot_silently_replace(tmp_path):
    first, second = tmp_path/'a', tmp_path/'b'
    donor(first); donor(second, price=111.)
    result = merge_sources(tmp_path/'new/inputs', [first, second], {'0050'})
    assert result['conflicts'] == [dict(kind='execution', key='limits:0050', donor=str(second))]


def test_tampered_donor_rejected_even_if_its_index_reuses_seen_hashes(tmp_path):
    first, second = tmp_path/'a', tmp_path/'b'
    donor(first); donor(second)
    (second/'execution-feeds/limits-0050.raw.json').write_text('{}')
    with pytest.raises(ReplayDataUnavailable, match='changed or is missing'):
        merge_sources(tmp_path/'new/inputs', [first, second], {'0050'})


def test_renormalized_fabricated_values_rejected_even_with_consistent_hashes(tmp_path):
    path = tmp_path/'a'; donor(path)
    rows = path/'execution-feeds/limits-0050.rows.json'
    value = read(rows); value['rows']['2022-01-04']['upper'] = 999
    write(rows, value)
    index = path/'execution-feeds/index.json'; state = read(index)
    state['files_sha256'][rows.name] = digest(rows); write(index, state)
    with pytest.raises(ReplayDataUnavailable, match='differs from parser'):
        merge_sources(tmp_path/'new/inputs', [path], {'0050'})


def test_missing_date_is_not_invented_by_catalog_coverage(tmp_path):
    path = tmp_path/'a'; donor(path)
    output = tmp_path/'new/inputs'; merge_sources(output, [path], {'0050'})
    assert inventory(output, {'0050'})['missing_limit_stocks'] == []
    feed = ReplayMarketFeeds(output/'execution-feeds', offline=True)
    assert '2022-01-05' not in feed.get_limits('0050')
    receipt = dict(identity={'sources':'fixed'}, cases={'normal':dict(completed=False, reason='Missing 2022-01-05')})
    result = gate(receipt, receipt['identity'], ['normal'])
    assert not result['ready'] and result['issues'][0]['message'] == 'Missing 2022-01-05'


@pytest.mark.parametrize('change', ['sources', 'missing_case', 'failed_case', 'extra_case'])
def test_gate_requires_exact_identity_and_every_complete_case(change):
    identity = {'signals':'fixed', 'sources':'original'}
    receipt = dict(identity=deepcopy(identity), cases={k:dict(completed=True) for k in ('normal','stress')})
    assert gate(receipt, identity, ['normal','stress'])['ready']
    if change == 'sources': identity['sources'] = 'changed'
    elif change == 'missing_case': del receipt['cases']['stress']
    elif change == 'extra_case': receipt['cases']['unrequested'] = dict(completed=True)
    else: receipt['cases']['stress'] = dict(completed=False, reason='Unresolved stock delivery')
    result = gate(receipt, identity, ['normal','stress'])
    assert not result['ready'] and not result['live_qualified']


def test_memoized_feed_does_not_let_callers_mutate_evidence(tmp_path):
    path = tmp_path/'a'; donor(path)
    feed = CachedPreparationFeeds(path/'execution-feeds', offline=True)
    feed.get_limits('0050')['2022-01-04']['upper'] = 0
    feed.get_odd('2022-01-04', '0050', 'TWSE')['odd_shares'] = 0
    assert feed.get_limits('0050')['2022-01-04']['upper'] == 110
    assert feed.get_odd('2022-01-04', '0050', 'TWSE')['odd_shares'] == 1200


def recovery_budget(tmp_path, monkeypatch, *, age_days=0):
    from datetime import datetime, timezone, timedelta
    from scripts import prepare_holder_flow_accounts as script
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    proof = odd_record()
    proof['retrieved_at'] = (datetime.now(timezone.utc)-timedelta(days=age_days)).isoformat()
    write(tmp_path/'response.json', proof)
    cache = tmp_path/'cache';cache.mkdir()
    write(cache/'official-recovery.json', dict(path='response.json', sha256=digest(tmp_path/'response.json')))
    budget = script.ExecutionBudget(cache/'budget.json', maximum={'finmind':2, 'official':2})
    return script, budget


def test_recovered_endpoint_stops_on_new_security_response(tmp_path, monkeypatch):
    from skills.replay_market_feeds import URLS
    script, budget = recovery_budget(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(script.RequestBudget, 'official', lambda *a, **kw: calls.append(kw) or Response({}, 428))
    with pytest.raises(ReplayDataUnavailable, match='security response'):
        budget.official(URLS['twse'], params={'date':'20220308'})
    with pytest.raises(ReplayDataUnavailable, match='endpoint stopped'):
        budget.official(URLS['twse'], params={'date':'20220309'})
    assert len(calls) == 1


def test_expired_recovery_never_sends_http(tmp_path, monkeypatch):
    from skills.replay_market_feeds import URLS
    script, budget = recovery_budget(tmp_path, monkeypatch, age_days=2)
    monkeypatch.setattr(script.RequestBudget, 'official', lambda *a, **kw: pytest.fail('No HTTP allowed'))
    with pytest.raises(ReplayDataUnavailable, match='fresh reviewed response'):
        budget.official(URLS['twse'])


def test_recovery_cannot_unlock_other_endpoints(tmp_path, monkeypatch):
    script, budget = recovery_budget(tmp_path, monkeypatch)
    def held(*a, **kw):
        raise ReplayDataUnavailable('Original hold retained')
    monkeypatch.setattr(script.Budget, 'official', held)
    with pytest.raises(ReplayDataUnavailable, match='Original hold retained'):
        budget.official('https://www.twse.com.tw/another-endpoint')


def test_failed_gate_prevents_formal_account_runner(tmp_path, monkeypatch):
    from scripts import prepare_holder_flow_accounts as script
    monkeypatch.setattr(script, 'study', lambda cache: (None, {}, {}, {}))
    monkeypatch.setattr(script, 'check', lambda *args: dict(ready=False))
    monkeypatch.setattr(script, 'prepared_case', lambda *a, **kw: pytest.fail('Blocked gate must not run account'))
    output = tmp_path/'run'
    with pytest.raises(ValueError, match='no performance backtest started'):
        script.replay(tmp_path, output)
    assert not output.exists()


def replay_fixture(tmp_path, monkeypatch):
    from scripts import prepare_holder_flow_accounts as script
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    monkeypatch.setattr(script, 'study', lambda cache: (None, {}, {}, {}))
    monkeypatch.setattr(script, 'check', lambda *args: dict(ready=True))
    monkeypatch.setattr(script, 'identity', lambda *args: {'sources': 'unchanged'})
    monkeypatch.setattr(script, 'jobs', lambda *args: iter([('normal', None, {}), ('stress', None, {})]))
    monkeypatch.setattr(script, 'append_trial_registry', lambda row: None)
    monkeypatch.setattr(script, 'CachedPreparationFeeds', lambda *args, **kw: None)
    monkeypatch.setattr(script, 'prepared_case', lambda *args, **kw:
        dict(completed=True, summary={'test_fixture': True}, account={'fixture': True}))
    return script


def test_complete_offline_runs_compare_and_detect_tampered_output(tmp_path, monkeypatch):
    script = replay_fixture(tmp_path, monkeypatch)
    first, second = tmp_path/'a', tmp_path/'b'
    script.replay(tmp_path, first)
    script.replay(tmp_path, second)
    script.compare(tmp_path, [first, second])
    assert read(tmp_path/'reproducibility.json')['passed'] is True
    assert read(first/'manifest.json')['live_qualified'] is False
    write(second/'normal.json', {'completed': True, 'fabricated': True})
    with pytest.raises(ValueError, match='Replay output changed'):
        script.compare(tmp_path, [first, second])


def test_incomplete_run_cannot_publish_complete_manifest(tmp_path, monkeypatch):
    script = replay_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(script, 'prepared_case', lambda *a, **kw:
        dict(completed=False, reason='A later source disappeared'))
    output = tmp_path/'run'
    with pytest.raises(ValueError, match='Prepared account changed behavior'):
        script.replay(tmp_path, output)
    assert not (output/'manifest.json').exists()


def test_source_change_during_replay_prevents_complete_manifest(tmp_path, monkeypatch):
    script = replay_fixture(tmp_path, monkeypatch)
    identities = iter([{'sources': 'original'}, {'sources': 'changed'}])
    monkeypatch.setattr(script, 'identity', lambda *a: next(identities))
    output = tmp_path/'run'
    with pytest.raises(ValueError, match='sources changed during replay'):
        script.replay(tmp_path, output)
    assert not (output/'manifest.json').exists()


def test_official_suspension_rejects_both_lots_without_inventing_prices():
    from skills.account_source_preflight import SuspensionOrders
    class Parent:
        def _execute_order(self, *args):
            raise ReplayDataUnavailable('Raw quote missing')
    class Account(SuspensionOrders, Parent):
        verified_suspensions = {('2409','2022-09-29'):dict(known_date='2022-09-08',evidence_files=['notice'])}
        names = {}; holdings = {}; orders = []
        def raw(self, *args): return None
    account = Account()
    assert account._execute_order(pd.Timestamp('2022-09-29'),'2409','buy',1234,'entry','event') == 0
    assert [(r['channel'],r['requested_qty'],r['filled_qty']) for r in account.orders] == [('board',1000,0),('odd',234,0)]
    assert all(r['failure'] == 'official_trading_suspension' for r in account.orders)
    with pytest.raises(ReplayDataUnavailable, match='Raw quote missing'):
        account._execute_order(pd.Timestamp('2022-10-11'),'2409','buy',1234,'entry','event')


def test_suspension_conflicting_with_real_trades_blocks():
    from skills.account_source_preflight import SuspensionOrders
    class Account(SuspensionOrders):
        verified_suspensions = {('2409','2022-09-29'):dict(known_date='2022-09-08')}
        def raw(self, *args): return 500
    with pytest.raises(ReplayDataUnavailable, match='conflicts with official suspension'):
        Account()._execute_order(pd.Timestamp('2022-09-29'),'2409','buy',100,'entry','event')


def test_suspension_requires_timely_unmodified_evidence(tmp_path):
    from skills.account_source_preflight import suspension_terms
    path=tmp_path/'notice';path.write_text('official test evidence')
    doc=dict(evidence_sha256={'notice':digest(path)},suspensions=[dict(stock_id='2409',known_date='2022-09-08',
        start='2022-09-29',end='2022-10-07',evidence_files=['notice'])])
    assert ('2409','2022-10-07') in suspension_terms(doc,tmp_path)
    doc['suspensions'][0]['known_date']='2022-10-01'
    with pytest.raises(ValueError,match='Invalid official suspension'):
        suspension_terms(doc,tmp_path)
    doc['suspensions'][0]['known_date']='2022-09-08';path.write_text('modified')
    with pytest.raises(ValueError,match='evidence changed'):
        suspension_terms(doc,tmp_path)
