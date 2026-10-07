from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from skills import strategy_comparison_data as mod
from skills.poc_range_data import RangeOddAcquisition
from test_prepare_volume_profile_odd import payload, response


@pytest.fixture
def root(tmp_path):
    path = tmp_path / mod.PREREG
    path.parent.mkdir(); path.write_text('fixed comparison, independent source acquisition')
    (tmp_path / mod.BUDGET_ADDENDUM).write_text('unused board allowance moves to financial; total unchanged')
    return tmp_path


def query(sid='2330', day='2024-01-03'):
    return dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)


def event(sid='2330'):
    return dict(event_id='red-' + sid, members=[sid], signal_date='2024-01-02', entry_date='2024-01-03')


def profile(available=True, reason=None):
    return dict(stock_id='2330', signal_date='2024-01-02', available=available,
                poc_up=True if available else None, reason=reason)


def test_new_budget_scope_cannot_reset_old_or_change_prereg(root):
    old = root / '.cache/poc-executable-20261004/board-v1/attempts/old.json'
    mod._write(old, {'started': True}, exclusive=True); before = old.read_bytes()
    budget = mod.ComparisonBudget(root, online=True)
    budget.reserve('board', query())
    assert budget.snapshot()['attempted_calls'] == 1
    assert mod.ComparisonBudget(root, online=True).snapshot()['attempted_calls'] == 1
    assert old.read_bytes() == before
    with pytest.raises(mod.ComparisonDataRequired, match='already reserved'):
        mod.ComparisonBudget(root, online=True).reserve('board', query())
    (root / mod.PREREG).write_text('changed scope')
    with pytest.raises(ValueError, match='scope changed'):
        mod.ComparisonBudget(root, online=True)


def test_offline_scope_never_creates_authorization(root):
    budget = mod.ComparisonBudget(root)
    with pytest.raises(mod.ComparisonDataRequired, match='offline'):
        budget.reserve('board', query())
    assert not budget.auth.exists() and budget.snapshot()['attempted_calls'] == 0


@pytest.mark.parametrize('purpose, value', [
    ('board', query(sid='00631L')), ('board', query(day='2026-10-05')),
    ('board', dict(query(), dataset='TaiwanStockPrice')),
    ('financial', dict(query(), end_date='2026-10-02')),
    ('financial', dict(query(), dataset='TaiwanStockDividend', start_date='2020-01-01', end_date='2026-10-02')),
])
def test_budget_rejects_scope_outside_exact_dataset_stock_and_dates(root, purpose, value):
    budget = mod.ComparisonBudget(root, online=True)
    with pytest.raises(ValueError):
        budget.reserve(purpose, value)
    assert budget.snapshot()['attempted_calls'] == 0


def test_budget_parallel_purpose_limits_and_total_are_durable(root, monkeypatch):
    monkeypatch.setattr(mod, 'ORIGINAL_BUDGETS', {'board': 3, 'profiles': 1, 'financial': 0})
    monkeypatch.setattr(mod, 'BUDGETS', {'board': 2, 'profiles': 1, 'financial': 1})
    monkeypatch.setattr(mod, 'MAXIMUM_FINMIND', 4)
    budget = mod.ComparisonBudget(root, online=True)
    def reserve(i):
        try:
            budget.reserve('board', query(sid=f'{2000+i}'))
            return True
        except mod.ComparisonDataRequired:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(reserve, range(4)))
    assert sum(values) == 2
    budget.reserve('profiles', query())
    budget.reserve('financial', dict(dataset='TaiwanStockDividend', data_id='2330',
                                    start_date='2018-01-01', end_date='2026-10-02'))
    assert budget.snapshot()['attempted_calls'] == 4
    with pytest.raises(mod.ComparisonDataRequired, match='budget exhausted'):
        mod.ComparisonBudget(root, online=True).reserve('profiles', query(sid='2308'))


def store(root, budget, purpose='board', online=False, fetcher=None):
    cfg = SimpleNamespace(finmind_token='fake-test-only', finmind_requests_per_hour=6000)
    limiter = lambda limit: SimpleNamespace(get_stats=lambda: SimpleNamespace(remaining_requests=5000, retry_after_seconds=0))
    return mod.ComparisonTickReceipts(root, root / mod.BASE / (purpose + '-v1'), purpose=purpose,
        budget=budget, online=online, fetcher=fetcher, config=cfg, limiter=limiter)


def save_tape(root, folder, sid='2330', day='2024-01-03', *, status='received'):
    path = root / folder / 'raw' / f'{sid}-{day}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'stock_id': sid, 'date': day, 'price': 100, 'volume': 1}]).to_parquet(path)
    receipt = root / folder / 'receipts' / f'{sid}-{day}.json'
    mod.write(receipt, dict(query=query(sid, day), status=status,
        raw_path=str(path.relative_to(root)), raw_sha256=mod.digest(path)))
    return path, receipt


def test_tape_cached_reuse_zero_budget_and_cross_purpose(root):
    budget = mod.ComparisonBudget(root)
    path, _ = save_tape(root, mod.BASE + '/profiles-v1')
    client = store(root, budget)
    assert client.get('2330', '2024-01-03')['raw_sha256'] == mod.digest(path)
    assert budget.snapshot()['attempted_calls'] == 0 and client.calls == 0
    path.write_bytes(b'mutated')
    with pytest.raises(ValueError, match='hash changed'):
        client.get('2330', '2024-01-03')


def test_missing_tape_stops_arm_instead_of_unavailable_false(root):
    client = store(root, mod.ComparisonBudget(root))
    with pytest.raises(mod.ComparisonDataRequired, match='not_requested'):
        client.get('2330', '2024-01-03')
    assert not (client.directory / 'attempts').exists()


def test_prior_failed_and_orphan_tapes_cannot_be_bypassed(root):
    budget = mod.ComparisonBudget(root)
    client = store(root, budget)
    _, receipt = save_tape(root, '.cache/poc-executable-20261004/board-v1', status='provider_error')
    save_tape(root, mod.BASE + '/profiles-v1')
    with pytest.raises(mod.ComparisonDataRequired, match='prior_cached_attempt_unavailable'):
        client.get('2330', '2024-01-03')
    receipt.unlink()
    mod.write(receipt.parent.parent / 'attempts' / receipt.name, {'query': query(), 'status': 'started'})
    with pytest.raises(mod.ComparisonDataRequired, match='Prior tick attempt is incomplete'):
        client.get('2330', '2024-01-03')


def test_new_gateway_uses_fixed_scope_no_retry_and_single_shared_count(root):
    calls = []
    def fetch(dataset, start, **kwargs):
        calls.append((dataset, start, kwargs))
        return pd.DataFrame([{'stock_id': '2330', 'date': '2024-01-03', 'price': 100, 'volume': 1}])
    budget = mod.ComparisonBudget(root, online=True)
    client = store(root, budget, online=True, fetcher=fetch)
    assert client.get('2330', '2024-01-03')['status'] == 'received'
    assert len(calls) == 1 and calls[0][2]['max_retries'] == 0
    assert calls[0][2]['requests_per_hour'] == 6000  # common gateway reserves 10% once
    assert budget.snapshot()['purpose_attempts']['board'] == 1
    other = store(root, budget, purpose='profiles', online=True, fetcher=fetch)
    assert other.get('2330', '2024-01-03')['status'] == 'received'
    assert len(calls) == 1 and budget.snapshot()['attempted_calls'] == 1


def test_known_tick_version_conflict_is_quality_evidence(root):
    client = store(root, mod.ComparisonBudget(root))
    client.reuse_index['2330-2024-01-03'] = {'content_conflict': True}
    assert client.get('2330', '2024-01-03')['status'] == 'cached_versions_conflict'


@pytest.mark.parametrize('reason', sorted(mod.QUALITY_REASONS))
def test_only_observed_quality_unknowns_may_be_excluded(reason):
    result = profile(False, reason)
    assert mod.require_profile_evidence(result, event()) is result


@pytest.mark.parametrize('reason', ['raw_tape_unavailable', 'not_requested', 'quota_paused', None])
def test_profile_absence_cannot_be_converted_to_negative_signal(reason):
    with pytest.raises(mod.ComparisonDataRequired):
        mod.require_profile_evidence(profile(False, reason), event())


def test_profile_stock_date_and_boolean_are_validated():
    with pytest.raises(ValueError, match='stock/date'):
        mod.require_profile_evidence(profile(), event('2308'))
    with pytest.raises(ValueError, match='boolean'):
        mod.require_profile_evidence(dict(profile(), poc_up=1), event())


def test_profile_memo_is_immutable_across_arm_consumers():
    client = mod.StrategyComparisonData.__new__(mod.StrategyComparisonData)
    client.profile_memo = {event()['event_id']: profile()}
    result = client.profile(mod.ARM, event())
    result['poc_up'] = False
    assert client.profile(mod.ARM, event())['poc_up'] is True
    with pytest.raises(ValueError, match='stock/date'):
        client.profile(mod.ARM, dict(event('2308'), event_id=event()['event_id']))


def test_financial_offline_missing_is_required_not_order_skip(root):
    client = mod.ComparisonExecutionData(root, budget=mod.ComparisonBudget(root), source_refs={})
    with pytest.raises(mod.ComparisonDataRequired, match='cache missing'):
        client.finmind('2330', 'TaiwanStockDividend')


def test_financial_old_raw_receipt_is_reconstructed_and_validated(root):
    folder = root / '.cache/poc-executable-20261004/execution-v1'
    key = '2330-TaiwanStockPriceLimit'
    q = dict(stock_id='2330', dataset='TaiwanStockPriceLimit', start='2018-01-01', end='2026-10-02')
    attempt = folder / 'attempts' / (key + '.json')
    mod.write(attempt, dict(query=q, status='started'))
    raw = folder / 'raw' / (key + '.parquet'); raw.parent.mkdir(parents=True)
    frame = pd.DataFrame([dict(stock_id='2330', date='2024-01-03', limit_down=90., limit_up=110., reference_price=100.)])
    frame.to_parquet(raw, index=False)
    receipt = folder / 'receipts' / (key + '.json')
    mod.write(receipt, dict(query=q, status='received', attempt_sha256=mod.digest(attempt), raw_sha256=mod.digest(raw)))
    budget = mod.ComparisonBudget(root)
    client = mod.ComparisonExecutionData(root, budget=budget, source_refs={})
    assert client.finmind('2330', 'TaiwanStockPriceLimit').equals(frame)
    assert budget.snapshot()['attempted_calls'] == 0
    assert str(raw.relative_to(root)) in client.refs
    frame.loc[0, 'stock_id'] = '2308'; frame.to_parquet(raw, index=False)
    fresh = mod.ComparisonExecutionData(root, budget=budget, source_refs={})
    with pytest.raises(ValueError):
        fresh.finmind('2330', 'TaiwanStockPriceLimit')


class Session:
    def __init__(self, values=()): self.values, self.calls = list(values), []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.values.pop(0)


def odd_engine(day):
    return SimpleNamespace(day_plans={'x': dict(date=day, stock_id='1216', planned_qty=1001, odd_qty=1, board_qty=1000)})


def test_odd_new_scope_preserves_global_hold_and_expired_offline_replay(root):
    now = [datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp()]
    hold = root / '.cache/official-origin-holds/www.twse.com.tw.json'
    mod._write(hold, dict(status='blocked', observed_at='2026-10-02T00:00:00Z', evidence_sha256={}))
    before = hold.read_bytes()
    session = Session([response(payload('TWSE', '2026-09-23', '1216'))])
    client = mod.ComparisonOddAcquisition(root, online=True, session=session, clock=lambda: now[0])
    result = client.get('2026-09-23', '1216', 'twse', odd_engine('2026-09-23'))
    assert result['odd_shares'] == 10000 and hold.read_bytes() == before
    assert client.expected_auth['maximum_attempts'] == 100 and client.expected_auth['prereg_path'] == mod.PREREG
    assert not (root / '.cache/poc-range-first-20261005/odd-v1/authorization.json').exists()
    now[0] += 90000
    offline = mod.ComparisonOddAcquisition(root, session=Session(), clock=lambda: now[0])
    assert offline.get('2026-09-23', '1216', 'twse')['odd_shares'] == 10000
    assert offline.snapshot()['maximum_attempts'] == 100


def test_odd_newer_hold_still_blocks_new_scope(root):
    now = datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp()
    session = Session()
    client = mod.ComparisonOddAcquisition(root, online=True, session=session, clock=lambda: now)
    mod._write(root / '.cache/official-origin-holds/www.twse.com.tw.json',
               dict(status='blocked', observed_at=client._now(), evidence_sha256={}))
    with pytest.raises(mod.ReplayDataUnavailable, match='Newer origin stop'):
        client.get('2026-09-23', '1216', 'twse', odd_engine('2026-09-23'))
    assert not session.calls


def test_odd_hundred_attempt_limit_not_inherited_four_hundred(root):
    session = Session()
    client = mod.ComparisonOddAcquisition(root, online=True, session=session)
    for i in range(100): mod.write(client.cache / 'attempts' / f'{i}.json', {'status': 'started'})
    with pytest.raises(mod.ComparisonDataRequired, match='100-attempt'):
        client.get('2026-09-23', '1216', 'twse', odd_engine('2026-09-23'))
    assert not session.calls


def test_odd_missing_offline_never_creates_authorization(root):
    client = mod.ComparisonOddAcquisition(root, session=Session())
    with pytest.raises(mod.ReplayDataUnavailable, match='day missing'):
        client.get('2026-09-23', '1216', 'twse', odd_engine('2026-09-23'))
    assert not client.auth.exists()


@pytest.mark.parametrize('message, expected_type', [
    ('Supplementary odd stock absent: 1216 2024-01-03', mod.ReplayDataUnavailable),
    ('Conflicting supplementary odd rows: 1216 2024-01-03', mod.ReplayDataUnavailable),
    ('Existing intraday table lacks stock: TWSE-2024-01-03 1216', mod.ReplayDataUnavailable),
    ('Conflicting official intraday rows: TWSE-2024-01-03 1216', mod.ReplayDataUnavailable),
    ('Intraday official odd day missing: TWSE-2024-01-03', mod.ComparisonDataRequired),
    ('Prior intraday odd request failed; no automatic retry', mod.ComparisonDataRequired),
])
def test_account_known_odd_quality_skip_is_distinct_from_unacquired_day(monkeypatch, message, expected_type):
    client = mod.StrategyComparisonData.__new__(mod.StrategyComparisonData)
    client.intraday_odds = SimpleNamespace(legacy={})
    def fail(*args):
        raise mod.ReplayDataUnavailable(message)
    monkeypatch.setattr(mod.RangeAccountData, 'get_odd', fail)
    with pytest.raises(expected_type):
        client.get_odd('2024-01-03', '1216', 'TWSE')


def seed_original_budget(root, counts):
    """Saved v1 receipts are immutable input, not recreated acquisitions."""
    expected = dict(schema='strategy_comparison_finmind_authorization_v1',
        authorization_basis='current_user_requested_fair_poc_red_rsi_account_comparison',
        request_description_is_paraphrase=True, prereg_path=mod.PREREG,
        prereg_sha256=mod.digest(root / mod.PREREG), start='2024-01-02', end=mod.END,
        tick_warmup_start='2023-11-01', financial_start=mod.START,
        purpose_maximums=mod.ORIGINAL_BUDGETS, maximum_attempts=mod.MAXIMUM_FINMIND,
        effective_shared_hourly_limit=5400, retries=0, previous_budgets_reset=False)
    base = root / mod.BASE / 'finmind-budget-v1'
    auth = base / 'authorization.json'
    mod._write(auth, dict(expected, created_at='2026-10-07T00:00:00+00:00'), exclusive=True)
    for purpose, count in counts.items():
        for i in range(count):
            mod._write(base / 'attempts' / purpose / f'old-{i}.json',
                dict(purpose=purpose, authorization_sha256=mod.digest(auth), status='started'), exclusive=True)
    return {p: p.read_bytes() for p in base.rglob('*.json')}


def test_reallocation_keeps_original_bytes_and_counts_both_authorizations(root):
    frozen = seed_original_budget(root, {'board': 156, 'profiles': 127, 'financial': 400})
    budget = mod.ComparisonBudget(root, online=True)
    budget.reserve('financial', dict(dataset='TaiwanStockDividend', data_id='2330',
        start_date='2018-01-01', end_date='2026-10-02'))
    summary = budget.snapshot()
    assert summary['attempted_calls'] == 684
    assert summary['purpose_attempts'] == {'board': 156, 'profiles': 127, 'financial': 401}
    assert summary['original_authorization_attempts'] == {'board': 156, 'profiles': 127, 'financial': 400}
    assert summary['purpose_maximums'] == {'board': 1200, 'profiles': 1800, 'financial': 1000}
    assert all(p.read_bytes() == content for p, content in frozen.items())
    auth = mod.read(budget.reallocation)
    assert auth['original_authorization']['sha256'] == mod.digest(budget.auth)
    assert mod.ComparisonBudget(root).snapshot()['attempted_calls'] == 684


def test_offline_old_ledger_remains_valid_without_minting_reallocation(root):
    seed_original_budget(root, {'financial': 400})
    budget = mod.ComparisonBudget(root)
    assert not budget.reallocation.exists()
    assert budget.snapshot()['purpose_maximums'] == mod.ORIGINAL_BUDGETS


def test_reallocation_cannot_transfer_already_spent_board_capacity(root):
    seed_original_budget(root, {'board': 1201})
    with pytest.raises(ValueError, match='already spent'):
        mod.ComparisonBudget(root, online=True)


@pytest.mark.parametrize('target', ['original_auth', 'addendum', 'new_auth', 'old_attempt'])
def test_reallocation_history_chain_changes_are_detected(root, target):
    seed_original_budget(root, {'financial': 1})
    budget = mod.ComparisonBudget(root, online=True)
    if target == 'original_auth':
        record = mod.read(budget.auth); record['purpose_maximums']['financial'] = 1000
        budget.auth.write_text(json.dumps(record))
    elif target == 'addendum':
        (root / mod.BUDGET_ADDENDUM).write_text('changed rules')
    elif target == 'new_auth':
        record = mod.read(budget.reallocation); record['maximum_attempts'] = 9999
        budget.reallocation.write_text(json.dumps(record))
    else:
        (budget.directory / 'attempts' / 'financial' / 'old-0.json').unlink()
    with pytest.raises(ValueError): budget.snapshot()


def test_financial_parent_ceiling_override_is_local_and_durable(root):
    budget = mod.ComparisonBudget(root, online=True)
    cfg = SimpleNamespace(finmind_token='fake-test-only', finmind_requests_per_hour=6000)
    calls = []
    def fetch(dataset, start, end, **kwargs):
        calls.append(kwargs['data_id'])
        return pd.DataFrame([dict(stock_id=kwargs['data_id'], date='2024-01-03',
            limit_down=90., limit_up=110., reference_price=100.)])
    client = mod.ComparisonExecutionData(root, budget=budget, source_refs={}, online=True, config=cfg, fetcher=fetch)
    assert client.maximum == 1000
    for i in range(400): mod.write(client.directory / 'attempts' / f'previous-{i}.json', {'status': 'started'})
    assert len(client.finmind('2330', 'TaiwanStockPriceLimit')) == 1
    assert calls == ['2330'] and len(list((client.directory / 'attempts').glob('*.json'))) == 401
    assert client.maximum == 1000 and mod.LatestExecutionData(root, source_refs={}).maximum == 400


def test_parent_exhaustion_creates_no_attempt_and_is_not_a_retry(root):
    parent = mod.LatestExecutionData(root, online=True, source_refs={}, maximum_requests=0)
    with pytest.raises(ValueError, match='Persistent execution request budget exhausted'):
        parent._request('2330', 'TaiwanStockPriceLimit', '2018-01-01')
    assert not list((parent.directory / 'attempts').glob('*.json'))
    assert not list((parent.directory / 'receipts').glob('*.json'))


@pytest.mark.parametrize('purpose, sibling', [('board', 'profiles'), ('profiles', 'board')])
def test_cross_purpose_new_orphan_cannot_hide_behind_legacy_success(root, purpose, sibling):
    save_tape(root, '.cache/poc-executable-20261004/board-v1')
    orphan = root / mod.BASE / (sibling + '-v1') / 'attempts' / '2330-2024-01-03.json'
    mod.write(orphan, dict(query=query(), status='started'))
    client = store(root, mod.ComparisonBudget(root), purpose=purpose)
    with pytest.raises(mod.ComparisonDataRequired, match='Prior tick attempt is incomplete'):
        client.get('2330', '2024-01-03')
    assert client.calls == 0 and not (client.directory / 'receipts').exists()


def test_empty_receipt_is_not_retried_after_reallocation(root):
    save_tape(root, mod.BASE + '/board-v1', status='empty')
    calls = []
    client = store(root, mod.ComparisonBudget(root, online=True), online=True,
                   fetcher=lambda *a, **k: calls.append(1))
    with pytest.raises(mod.ComparisonDataRequired, match='empty'):
        client.get('2330', '2024-01-03')
    assert calls == []
