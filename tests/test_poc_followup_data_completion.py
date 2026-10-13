import hashlib
import json
from types import SimpleNamespace

import pytest

from skills import poc_followup_data_completion as module
from skills.strategy_comparison_data import ComparisonDataRequired


class Previous:
    def __init__(self, root):
        self.directory = root/'previous'
        self.directory.mkdir()
        self.auth = self.directory/'authorization.json'
        self.auth.write_text('{"unchanged":true}')
        self.refs = {}
        self.used = dict(profiles=7999, financial=751, board=2578)

    def validate(self):
        assert self.auth.read_text() == '{"unchanged":true}'

    def counts(self):
        return self.used.copy()

    def reserve(self, purpose, query):
        self.used[purpose] += 1
        key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()+'.json'
        path = self.directory/'attempts'/purpose/key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(query))


def test_additional_allowance_keeps_prior_counts_and_refuses_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ADDITIONAL', 2)
    spec = tmp_path/'docs/poc_followup_data_completion_20261014.json'
    spec.parent.mkdir(); spec.write_text('{}')
    previous = Previous(tmp_path)
    budget = module.AdditionalProfileBudget(previous, online=True, root=tmp_path)
    query = lambda sid: dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date='2024-02-01')
    budget.reserve('profiles', query('1111'))
    assert previous.counts()['profiles'] == 8000
    assert budget.additional_count() == 0
    with pytest.raises(ComparisonDataRequired, match='no automatic retry'):
        budget.reserve('profiles', query('1111'))
    budget.reserve('profiles', query('2222'))
    budget.reserve('profiles', query('3333'))
    assert previous.counts() == dict(profiles=8000, financial=751, board=2578)
    assert budget.additional_count() == 2
    called = []
    monkeypatch.setattr(module.TickData, '_fetch', lambda *args: called.append(args))
    ticks = module.CompletionTicks.__new__(module.CompletionTicks)
    ticks.budget, ticks.purpose = budget, 'profiles'
    with pytest.raises(ComparisonDataRequired, match='not dispatched'):
        ticks._fetch(query('4444'), tmp_path/'receipt', tmp_path/'attempt')
    assert called == []
    assert not (tmp_path/'attempt').exists()


@pytest.mark.parametrize('change', [{'data_id':'0050A'}, {'dataset':'Other'},
    {'start_date':'2026-10-13'}, {'end_date':'2026-10-02'}])
def test_additional_scope_cannot_expand_query(change):
    query = dict(dataset='TaiwanStockPriceTick', data_id='2330', start_date='2024-02-01')
    with pytest.raises(ValueError, match='scope'):
        module.validate_query(dict(query, **change))


def test_shared_quota_waits_without_acquiring_or_resetting():
    values = iter([SimpleNamespace(remaining_requests=0, retry_after_seconds=10, window_start_time=0),
                   SimpleNamespace(remaining_requests=4, retry_after_seconds=0, window_start_time=0)])
    waits = []
    limiter = SimpleNamespace(get_stats=lambda: next(values))
    module.wait_for_shared_capacity(limiter, sleep=waits.append, clock=lambda: 4000)
    assert waits == [10]
