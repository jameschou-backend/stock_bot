from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app import research_terminal_api as api
from app.research_terminal_service import EvidenceError
from app.workbench_api import require_local


def client(local=True):
    app = FastAPI()
    app.include_router(api.router)
    if local:
        app.dependency_overrides[require_local] = lambda: None
    return TestClient(app)


class Stub:
    def overview(self):
        return dict(source_end='2026-10-05', live_qualified=False)

    def backtest(self, params):
        return dict(job_id='a'*32, status='completed', params=params)

    def research_episodes(self, stock_id, year, horizon, offset, limit):
        return dict(total=1, rows=[], horizon=horizon, offset=offset, limit=limit)


def test_local_access_and_cross_origin_mutation_guard(monkeypatch):
    monkeypatch.setattr(api, 'get_terminal', lambda: Stub())
    body = dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=20)
    with client() as c:
        assert c.get('/research-terminal/api/overview').status_code == 200
        assert c.post('/research-terminal/api/backtests', json=body).status_code == 200
        assert c.post('/research-terminal/api/backtests', json=body,
                      headers={'Origin': 'https://unrelated.example'}).status_code == 403
        assert c.post('/research-terminal/api/backtests', json=body,
                      headers={'Origin': 'http://testserver'}).status_code == 200
    with client(local=False) as c:
        assert c.get('/research-terminal/api/overview').status_code == 403
        assert c.post('/research-terminal/api/backtests', json=body).status_code == 403


@pytest.mark.parametrize('body', [
    dict(mode='signal_study'),
    dict(mode='account_replay', strategy_id='poc_up_red'),
    dict(mode='account_replay', start='2024-01-01'),
    dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=10),
    dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=20, cost_bps=1),
    dict(mode='signal_study', strategy_id='poc_up_red', start='2024-01-02', end='2026-10-02', horizon=20, fresh=True),
    dict(mode='account_replay', command='cat .env'),
])
def test_backtest_contract_rejects_unimplemented_or_mixed_modes(body, monkeypatch):
    monkeypatch.setattr(api, 'get_terminal', lambda: Stub())
    with client() as c:
        assert c.post('/research-terminal/api/backtests', json=body).status_code == 422


def test_evidence_failure_is_explicit_503(monkeypatch):
    class Broken:
        def overview(self):
            raise EvidenceError('SHA256 核對失敗')
    monkeypatch.setattr(api, 'get_terminal', lambda: Broken())
    with client() as c:
        result = c.get('/research-terminal/api/overview')
        assert result.status_code == 503
        assert 'SHA256' in result.json()['detail']


def test_rally_horizon_query_string_is_parsed_as_integer(monkeypatch):
    monkeypatch.setattr(api, 'get_terminal', lambda: Stub())
    with client() as c:
        for route in ('episodes', 'examples'):
            result = c.get('/research-terminal/api/research/'+route+'?horizon=20&limit=1')
            assert result.status_code == 200
            assert result.json()['horizon'] == 20


@pytest.mark.parametrize('path', [
    '/research-terminal/api/stocks/2308?sessions=999999',
    '/research-terminal/api/backtests/'+'a'*32+'/events?limit=2001',
    '/research-terminal/api/backtests/'+'a'*32+'/events?offset=-1',
    '/research-terminal/api/signals?search='+'x'*65,
])
def test_expensive_or_unbounded_requests_fail_validation(path):
    with client() as c:
        assert c.get(path).status_code == 422
