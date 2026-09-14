from copy import deepcopy
from datetime import timedelta

import pytest
from app import capacity_source_guard as g, capacity_guard_runner as runner
from app import capacity_source_review as review
from app import capacity_forward as cap, forward_journal as j, forward_simulation as sim
from tests.test_capacity_forward import prepare
from tests.test_forward_portfolio import clock
from tests.test_forward_simulation import observation
from tests.test_forward_corporate_audit import sources


def setup(tmp_path, age=0):
    root, original, _ = prepare(tmp_path)
    now = clock('2026-09-14', 10)()
    g.activate(root, lambda: now)
    evidence = tmp_path / 'sources'
    sources(root / 'strategy.sqlite3', evidence, at=lambda: now - timedelta(seconds=age))
    return root, original, evidence, now


def quote(seconds=0, volume=1000):
    result = observation(seconds, volume)
    q = result['body']['quote']
    q.update(stock_id='1101', channel='board')
    q['asks'][0]['shares'] = 50000
    result['hash'] = j.digest(result['body'])
    return result


def test_activation_preserves_old_books_and_rejects_changed_fingerprints(tmp_path, monkeypatch):
    root, _, _ = prepare(tmp_path)
    before = {role: cap.verify(root / (role + '.sqlite3')) for role in cap.ROLES}
    assert g.activate(root, clock())['status'] == 'activated'
    assert g.activate(root, clock())['status'] == 'already_active'
    assert before == {role: cap.verify(root / (role + '.sqlite3')) for role in cap.ROLES}
    monkeypatch.setattr(g, 'hashes', lambda: {})
    with pytest.raises(ValueError, match='版本不符'):
        g.verify(root)


def test_missing_guard_and_changed_parent_never_bypass_checks(tmp_path, monkeypatch):
    root, _, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match='尚未啟用'):
        g.verify(root)
    g.activate(root, clock())
    monkeypatch.setitem(cap.RULES, 'max_drawdown', '0.99')
    with pytest.raises(ValueError, match='版本'):
        g.verify(root)


def test_fresh_pair_uses_unchanged_matcher_and_duplicate_is_not_new_fill(tmp_path):
    root, original, evidence, now = setup(tmp_path)
    before = sim.read(original)
    assert not g.match(root, 'strategy', quote(), lambda: now, evidence)['fills']
    later = lambda: now + timedelta(seconds=20)
    result = g.match(root, 'strategy', quote(20, 21000), later, evidence)
    assert len(result['fills']) == 1
    fills = [r for r in cap.verify(root / 'strategy.sqlite3') if r['kind'] == 'fill']
    assert fills[-1]['body']['qty'] == 2000
    assert not g.match(root, 'strategy', quote(20, 21000), later, evidence)['fills']
    assert sim.read(original) == before


@pytest.mark.parametrize('age', [3601, -1])
def test_stale_or_future_source_blocks_without_account_write(tmp_path, age):
    root, _, evidence, now = setup(tmp_path, age)
    path = root / 'strategy.sqlite3'
    before = cap.verify(path)
    result = g.match(root, 'strategy', quote(), lambda: now, evidence)
    assert result['status'] == 'blocked' and not result['fills']
    assert cap.verify(path) == before


def test_first_observation_passes_but_second_crosses_ttl_and_cannot_fill(tmp_path):
    root, _, evidence, now = setup(tmp_path, 3590)
    path = root / 'strategy.sqlite3'
    assert g.match(root, 'strategy', quote(), lambda: now, evidence)['status'] == 'observation'
    before = cap.verify(path)
    later = lambda: now + timedelta(seconds=20)
    result = g.match(root, 'strategy', quote(20, 21000), later, evidence)
    assert result['status'] == 'blocked' and cap.verify(path) == before
    # Isolated fixture proves the old matching core WOULD fill this pair when
    # called without the new check. This is not a market-data claim.
    assert sim.match(path, quote(20, 21000), later)['fills']


@pytest.mark.parametrize('jump', [11, -1])
def test_expiry_or_clock_reversal_after_fill_insert_rolls_back_entire_observation(tmp_path, monkeypatch, jump):
    root, _, evidence, now = setup(tmp_path, 3570)
    path = root / 'strategy.sqlite3'
    g.match(root, 'strategy', quote(), lambda: now, evidence)
    before = cap.verify(path)
    current = [now + timedelta(seconds=20)]
    real_submit = sim.p.submit
    inserted = []

    def submit(con, command, guarded_clock):
        result = real_submit(con, command, guarded_clock)
        if command['kind'] == 'fill':
            inserted.append(result['hash'])
            current[0] += timedelta(seconds=jump)
        return result

    monkeypatch.setattr(sim.p, 'submit', submit)
    result = g.match(root, 'strategy', quote(20, 21000), lambda: current[0], evidence)
    assert inserted and result['status'] == 'blocked' and not result['fills']
    assert cap.verify(path) == before


def test_unfinished_check_blocks_later_run_without_automatic_repair(tmp_path):
    root, _, _, now = setup(tmp_path)
    with j.connection(root / g.NAME) as con:
        j.append(con, 'crash:start', 'source_guard_check', {}, lambda: now)
    with pytest.raises(ValueError, match='中斷'):
        g.verify(root)


def test_post_commit_verification_failure_remains_unresolved_not_false_zero_fill(tmp_path, monkeypatch):
    root, _, evidence, now = setup(tmp_path)
    path = root / 'strategy.sqlite3'
    g.match(root, 'strategy', quote(), lambda: now, evidence)
    real_match, real_verify = sim.match, cap.verify
    committed = [False]

    def match(*args):
        result = real_match(*args)
        committed[0] = True
        return result

    def verify(*args):
        if committed[0]:
            raise ValueError('post-commit verification failure')
        return real_verify(*args)

    monkeypatch.setattr(sim, 'match', match)
    monkeypatch.setattr(cap, 'verify', verify)
    with pytest.raises(ValueError, match='post-commit'):
        g.match(root, 'strategy', quote(20, 21000), lambda: now + timedelta(seconds=20), evidence)
    assert any(r['kind'] == 'fill' for r in real_verify(path))
    monkeypatch.setattr(cap, 'verify', real_verify)
    with pytest.raises(ValueError, match='中斷'):
        g.verify(root)


def test_observer_rechecks_after_wait_and_keeps_shared_refresh_budget(tmp_path, monkeypatch):
    root, _, evidence, now = setup(tmp_path, 3590)
    current = [now]
    refresh_calls = []
    actual_inspect = g.corporate.inspect
    actual_match = g.match
    monkeypatch.setattr(runner.legacy, 'refresh', lambda paths, clock: refresh_calls.append(len(paths)) or dict(calls=0, reused=15))
    monkeypatch.setattr(g.corporate, 'inspect', lambda path, *args, **kw: actual_inspect(path, *(args or (evidence,)), **kw))
    monkeypatch.setattr(g, 'match', lambda root, role, obs, clock: actual_match(root, role, obs, clock, evidence))
    monkeypatch.setattr(runner.legacy.base, 'markets', lambda ids: {sid: 'tse' for sid in ids})
    # Seed sources for all three arms, including benchmark.
    for role in ('control', 'benchmark'):
        sources(root / (role + '.sqlite3'), evidence, at=lambda: now - timedelta(seconds=3590))

    def fetch(market, sid, channel):
        o = quote(0 if current[0] == now else 20, 1000 if current[0] == now else 21000)
        o['body']['quote'].update(stock_id=sid, channel=channel)
        o['hash'] = j.digest(o['body'])
        return o

    monkeypatch.setattr(runner.legacy.base, 'calendar_day', lambda day: True)
    result = runner.run(root, clock=lambda: current[0], sleeper=lambda seconds: current.__setitem__(0, now + timedelta(seconds=seconds)), fetcher=fetch)
    assert result['status'] == 'needs_attention' and refresh_calls == [3]
    assert any(o['status'] == 'blocked' for o in result['stages'][0]['detail']['observations'])
    assert all(not any(r['kind'] == 'fill' for r in cap.verify(root / (role + '.sqlite3'))) for role in cap.ROLES)


def test_closed_market_does_not_refresh_or_match(tmp_path, monkeypatch):
    root, _, _, now = setup(tmp_path)
    monkeypatch.setattr(runner.legacy.base, 'calendar_day', lambda day: False)
    monkeypatch.setattr(runner.legacy, 'refresh', lambda *a: pytest.fail('no refresh'))
    assert runner.run(root, clock=lambda: now)['status'] == 'closed_market'


def test_crossing_open_cannot_delegate_to_unguarded_legacy_observer(tmp_path, monkeypatch):
    root, _, _, _ = setup(tmp_path)
    times = iter([clock('2026-09-14', 8)(), clock('2026-09-14', 9)()])
    monkeypatch.setattr(runner.legacy, 'observe', lambda *a: pytest.fail('unguarded observer'))
    with pytest.raises(ValueError, match='跨入盤中'):
        runner.run(root, clock=lambda: next(times))


def test_historical_review_ignores_later_recorded_source_and_rejects_future_retrieval(tmp_path):
    root, _, evidence, now = setup(tmp_path)
    rows = cap.verify(root / 'strategy.sqlite3')
    history = g.corporate.source_history(evidence)
    assert review.at_time(rows, history, now)['source_freshness_passed']
    assert not review.at_time(rows, history, now + timedelta(seconds=3601))['source_freshness_passed']
    late = deepcopy(history)
    for r in late:
        r['recorded_at'] = (now + timedelta(seconds=1)).isoformat()
    assert not review.at_time(rows, late, now)['source_freshness_passed']
    future = deepcopy(history)
    future[0]['body']['retrieved_at'] = (now + timedelta(seconds=1)).isoformat()
    assert not review.at_time(rows, future, now)['source_freshness_passed']
