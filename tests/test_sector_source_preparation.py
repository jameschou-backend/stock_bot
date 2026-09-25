import pandas as pd
import pytest

from scripts.prepare_sector_account_sources import RequestBudget, initialize
from skills.replay_market_feeds import ReplayDataUnavailable


def test_budget_reserves_failed_and_interrupted_attempts(tmp_path):
    path = tmp_path / 'budget.json'
    budget = RequestBudget(path, maximum={'finmind': 2, 'official': 1})
    def fail():
        raise ValueError('provider failed')
    with pytest.raises(ValueError, match='provider failed'):
        budget.call('finmind', 'sid:1234', fail)
    resumed = RequestBudget(path, maximum={'finmind': 2, 'official': 1})
    with pytest.raises(ReplayDataUnavailable, match='already attempted'):
        resumed.call('finmind', 'sid:1234', lambda: pd.DataFrame())
    resumed.call('finmind', 'sid:5678', lambda: pd.DataFrame())
    with pytest.raises(ReplayDataUnavailable, match='budget exhausted'):
        resumed.call('finmind', 'sid:9999', lambda: pd.DataFrame())
    assert resumed.state['attempts']['finmind'] == 2
    assert resumed.state['requests'][0]['error_type'] == 'ValueError'


def test_budget_cannot_be_silently_increased_on_resume(tmp_path):
    path = tmp_path / 'budget.json'
    RequestBudget(path, maximum={'finmind': 1, 'official': 1})
    with pytest.raises(ValueError, match='budget changed'):
        RequestBudget(path, maximum={'finmind': 200, 'official': 100})


def test_output_must_not_be_inside_sealed_parent(monkeypatch, tmp_path):
    monkeypatch.setattr('scripts.prepare_sector_account_sources.ROOT', tmp_path)
    parent = tmp_path / '.cache' / 'sealed' / 'inputs'
    with pytest.raises(ValueError, match='separate directory'):
        initialize(parent / 'accidental-child', parent)
    with pytest.raises(ValueError, match='separate directory'):
        initialize(parent.parent, parent)


def test_finmind_budget_uses_shared_client_without_retries(monkeypatch, tmp_path):
    calls = []
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        frame = pd.DataFrame()
        frame.attrs['cache_hit'] = True
        return frame
    monkeypatch.setattr('scripts.prepare_sector_account_sources.fetch_dataset', fetch)
    budget = RequestBudget(tmp_path / 'budget.json')
    budget.finmind('TaiwanStockPriceLimit', '2022-01-01', '2026-09-09', data_id='2492')
    assert calls[0][1] == dict(data_id='2492', max_retries=0, requests_per_hour=6000, timeout=30)
    assert budget.state['requests'][0]['shared_cache_hit'] is True
    with pytest.raises(ValueError, match='Dataset'):
        budget.finmind('ArbitraryDataset', '', '')
