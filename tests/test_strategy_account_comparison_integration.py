"""Independent contracts between frozen candidates and the account runner."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import prepare_strategy_account_comparison as prepare
from scripts import research_strategy_account_comparison as account
from skills.poc_range_execution import RangeGapOrders


@pytest.fixture(scope='module')
def real_candidates():
    days = pd.bdate_range('2023-10-02', account.END).difference(pd.DatetimeIndex(['2024-01-01']))
    close = 200. - np.arange(len(days))*.05
    close[days.get_loc('2023-12-29')] += 15
    close[-1] += 15
    rows = []
    for sid, volume in [('0050', 3_000_000), ('2317', 1_000_000),
                        ('2330', 1_000_000), ('2454', 2_000_000)]:
        for day, price in zip(days, close):
            rows.append(dict(date=day, stock_id=sid, open=price-1, high=price+1,
                             low=price-2, close=price, volume=volume, amount=price*volume,
                             adjusted_close=price, eligible=True, quality=True))
    return prepare.prepare_entries(pd.DataFrame(rows), days)


def test_prepared_candidates_roundtrip_through_actual_runner_loader(real_candidates, tmp_path, monkeypatch):
    payload, calendar = real_candidates
    output = tmp_path/'candidates'
    prepare.write_inputs(output, payload, calendar, {}, root=tmp_path)
    original_loader = prepare.load_candidates
    # Root location is a test seam; actual preparation/hash/contract code runs.
    monkeypatch.setattr(account, 'ROOT', tmp_path)
    monkeypatch.setattr(prepare, 'load_candidates',
                        lambda folder, verify_sources=True: original_loader(folder, verify_sources=verify_sources, root=tmp_path))
    entries, loaded, refs = account.load_rsi_candidates(output)
    assert entries == payload['entries']
    assert loaded['_calendar'] == calendar
    assert loaded['pending_entries'] == payload['pending_entries']
    assert set(refs) == {f'candidates/{name}' for name in
                         ['manifest.json', 'manifest.sha256', 'rsi-entries.json', 'calendar.json']}
    assert all(row['entry_date'] is not None for row in entries)
    assert all(row['entry_date'] is None for row in loaded['pending_entries'])
    boundary = [e for e in entries if e['entry_date'] == account.START]
    assert [e['members'][0] for e in boundary] == ['2454', '2317', '2330']
    assert all(e['signal_date'] == '2023-12-29' for e in boundary)


def test_rsi_arms_use_identical_candidates_without_original_red_or_first_red_gate(real_candidates):
    payload, calendar = real_candidates
    class RejectAnyRedRead:
        def filter_entries(self, _):
            raise AssertionError('RSI family must not acquire the original red-candle gate')
    expected = payload['entries']
    for arm in account.RSI_ARMS:
        rows, decisions, first = account.select_arm_entries(
            arm, [], RejectAnyRedRead(), account.START, account.END, rsi_entries=expected)
        assert rows == expected and rows is not expected
        assert not decisions and not first
        account.validate_candidate_calendar(rows, pd.DatetimeIndex(calendar))
        assert account.ARM_RULES[arm][0] is False


@pytest.mark.parametrize('problem', ['same_day', 'extra_day', 'pending', 'etf', 'nan_priority'])
def test_runner_rejects_calendar_identity_or_priority_changes(real_candidates, problem):
    payload, calendar = real_candidates
    row = deepcopy(payload['entries'][0])
    if problem == 'same_day': row['entry_date'] = row['signal_date']
    elif problem == 'extra_day': row['entry_date'] = calendar[calendar.index(row['entry_date'])+1]
    elif problem == 'pending': row['entry_date'] = None
    elif problem == 'etf': row['members'] = ['0050']
    else: row['priority'] = float('nan')
    with pytest.raises(ValueError):
        account.validate_candidate_calendar([row], pd.DatetimeIndex(calendar))


def test_stock_and_benchmark_arms_share_execution_pricing_and_dates():
    assert len(account.ARMS) == 8 and len(set(account.ARMS)) == 8
    assert {(r[2], r[3]) for r in account.ARM_RULES.values()} == {(.7, .3)}
    assert (account.START, account.END) == (prepare.START, prepare.END)
    stock, benchmark = account.engine_types(RangeGapOrders, account.HistoricalOddEra)
    native, _ = account.engine_types(RangeGapOrders, account.HistoricalOddEra, native_time20=True)
    for cls in [stock, benchmark, native]:
        assert cls._execute_order is RangeGapOrders._execute_order
        assert cls.mro().index(RangeGapOrders) < cls.mro().index(account.MidpointBenchmark if cls is benchmark else account.MidpointExitReplay)
    assert native.mro().index(account.NativeTime20Scenario) < native.mro().index(account.ScenarioExitReplay)


class Planner:
    def __init__(self, slots=3):
        self.slots, self.reserved = slots, []
    def snapshot(self):
        return dict(slots=self.slots, reserved=list(self.reserved))
    def can_continue(self):
        return len(self.reserved) < self.slots
    def probe(self, event):
        return dict(attempted=self.can_continue(), allocation=100., raw_qty=1,
                    stock_id=event['members'][0], event_id=event['event_id'])
    def consider(self, event):
        result = self.probe(event)
        if result['attempted']:
            self.reserved.append(event['event_id'])
        return result


def profile_events():
    return [dict(event_id=f'original-2024-01-02-{sid}', members=[sid],
                 signal_date='2024-01-02', entry_date='2024-01-03', priority=priority)
            for sid, priority in [('2317', 3.), ('2330', 2.), ('2454', 1.)]]


def test_three_common_known_arms_share_missing_policy_and_preserve_true_profile_evidence():
    events = profile_events()
    calls = Counter()
    def provider(_, event):
        sid = event['members'][0]
        calls[sid] += 1
        if sid == '2454':
            return dict(available=False, poc_up=None, reason='ordinary_tape_conflict', recoverable=False)
        return dict(available=True, poc_up=sid == '2330', reason=None)
    shared = account.MemoizedProfiles(provider)
    expected = dict(red_known=['2317', '2330'], poc_priority_known=['2330', '2317'], poc_filter=['2330'])
    for arm in ['red_known', 'poc_priority_known', 'poc_filter']:
        planner = Planner()
        initial = planner.snapshot()
        result = account.select_known_profiles(events, arm=account.SELECTION_ARMS[arm],
            provider=lambda event: account.profile_for_arm(arm, shared, event), planner=planner,
            planner_factory=lambda _: Planner())
        assert [e['members'][0] for e in result['events']] == expected[arm]
        assert planner.snapshot() == initial
        exclusions = result['certificate']['profile_data_exclusions']
        assert len(exclusions) == 1 and exclusions[0]['stock_id'] == '2454'
        assert exclusions[0]['available'] is False and exclusions[0]['poc_up'] is None
        cohorts = [dict(event_id=e['event_id']) for e in result['events']]
        account.audit_profile_admissions(dict(cohorts=cohorts), shared.values, hard_filter=arm == 'poc_filter')
    # The availability-control arm changes a private return copy, not evidence.
    assert shared.values[events[0]['event_id']]['poc_up'] is False
    assert calls == {'2317': 1, '2330': 1, '2454': 1}


def test_unacquired_profile_stops_instead_of_becoming_false_or_quality_exclusion():
    events = profile_events()
    shared = account.MemoizedProfiles(lambda _, event:
        dict(available=False, poc_up=None, reason='raw_tape_unavailable', recoverable=True))
    with pytest.raises(RuntimeError, match='acquisition is incomplete'):
        account.select_known_profiles(events, arm='poc_filter', provider=shared,
            planner=Planner(), planner_factory=lambda _: Planner())
    assert not shared.values  # A failed acquisition is not a cached negative.


def test_rsi_funded_cohorts_must_match_candidate_clock_and_identity(real_candidates):
    payload, _ = real_candidates
    e = payload['entries'][0]
    cohort = dict(event_id=e['event_id'], stock_id=e['members'][0],
                  signal_date=e['signal_date'], entry_date=e['entry_date'])
    assert account.audit_candidate_identity(dict(cohorts=[cohort]), payload['entries'])['funded_events_checked'] == 1
    cohort['signal_date'] = cohort['entry_date']
    with pytest.raises(ValueError, match=r'registered T\+1'):
        account.audit_candidate_identity(dict(cohorts=[cohort]), payload['entries'])


@pytest.mark.parametrize('old_purpose,new_purpose', [('board', 'profiles'), ('profiles', 'board')])
def test_cross_purpose_orphan_tick_cannot_reach_another_gateway(tmp_path, old_purpose, new_purpose):
    """A crashed stock/day query remains reserved across both tick consumers."""
    from types import SimpleNamespace
    from skills import strategy_comparison_data as adapter
    prereg = tmp_path / adapter.PREREG
    prereg.parent.mkdir(); prereg.write_text('frozen comparison test scope')
    (tmp_path / adapter.BUDGET_ADDENDUM).write_text('same total budget, additive reallocation')
    budget = adapter.ComparisonBudget(tmp_path, online=True)
    query = dict(dataset='TaiwanStockPriceTick', data_id='2330', start_date='2024-01-03')
    budget.reserve(old_purpose, query)
    original = tmp_path / adapter.BASE / (old_purpose+'-v1') / 'attempts' / '2330-2024-01-03.json'
    adapter.write(original, dict(query=query, status='started'))
    before = original.read_bytes()
    calls = []
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame([dict(stock_id='2330', date='2024-01-03', price=100., volume=1)])
    config = SimpleNamespace(finmind_token='fake-test-only', finmind_requests_per_hour=6000)
    limiter = lambda _: SimpleNamespace(get_stats=lambda: SimpleNamespace(remaining_requests=5000, retry_after_seconds=0))
    client = adapter.ComparisonTickReceipts(tmp_path, tmp_path / adapter.BASE / (new_purpose+'-v1'),
        purpose=new_purpose, budget=budget, online=True, config=config, limiter=limiter, fetcher=fetch)
    try:
        with pytest.raises(adapter.ComparisonDataRequired, match='incomplete|orphan'):
            client.get('2330', '2024-01-03')
    finally:
        assert original.read_bytes() == before
        assert calls == [], 'A second purpose dispatched the already-reserved stock/day query'
        assert budget.snapshot()['attempted_calls'] == 1


@pytest.mark.parametrize('online', [False, True])
@pytest.mark.parametrize('origin', ['same_scope', 'old_executable'])
def test_pending_financial_request_stops_without_gateway_or_new_attempt(tmp_path, online, origin):
    from types import SimpleNamespace
    from skills import strategy_comparison_data as adapter
    prereg = tmp_path / adapter.PREREG
    prereg.parent.mkdir(); prereg.write_text('frozen comparison test scope')
    (tmp_path / adapter.BUDGET_ADDENDUM).write_text('same total budget, additive reallocation')
    budget = adapter.ComparisonBudget(tmp_path, online=online)
    folder = (adapter.BASE if origin == 'same_scope' else '.cache/poc-executable-20261004') + '/execution-v1'
    path = tmp_path / folder / 'attempts' / '2330-TaiwanStockPriceLimit.json'
    query = dict(stock_id='2330', dataset='TaiwanStockPriceLimit', start='2018-01-01', end=account.END)
    adapter.write(path, dict(query=query, status='started'))
    before = path.read_bytes(); calls = []
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('Orphaned financial request reached provider')
    client = adapter.ComparisonExecutionData(tmp_path, budget=budget, online=online,
        source_refs={}, config=SimpleNamespace(finmind_token='fake-test-only', finmind_requests_per_hour=6000),
        fetcher=fetch)
    with pytest.raises((adapter.ComparisonDataRequired, ValueError)):
        client._request('2330', 'TaiwanStockPriceLimit', '2018-01-01')
    assert calls == [] and path.read_bytes() == before
    assert budget.snapshot()['attempted_calls'] == 0
    assert not (client.directory / 'receipts' / path.name).exists()
