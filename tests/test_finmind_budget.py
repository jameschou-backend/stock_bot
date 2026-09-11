from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date
from pathlib import Path
import time

import pandas as pd
import pytest
import requests

from app import finmind
from app.rate_limiter import RateLimiter, get_rate_limiter, reset_global_limiter


def _reserve(path):
    return RateLimiter(20, buffer_percent=0, path=path).acquire(timeout=0)


def test_processes_share_atomic_budget_and_restart(tmp_path):
    path = tmp_path / 'budget.db'
    RateLimiter(20, buffer_percent=0, path=path)
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(_reserve, [str(path)] * 50)) == 20
    assert not RateLimiter(20, buffer_percent=0, path=path).acquire()


def test_cap_stricter_caller_timeout_and_expiry(tmp_path, monkeypatch):
    path = tmp_path / 'budget.db'
    monkeypatch.setenv('FINMIND_STATE_PATH', str(path))
    monkeypatch.setenv('FINMIND_REQUESTS_PER_HOUR', '6000')
    assert get_rate_limiter(20000).effective_limit == 5400
    small = RateLimiter(1, buffer_percent=0, path=path)
    assert small.acquire()
    reset_global_limiter()
    assert not get_rate_limiter().acquire()
    started = time.monotonic()
    assert not small.acquire(timeout=.04)
    assert time.monotonic() - started < .5
    now = time.time()
    monkeypatch.setattr('app.rate_limiter.time.time', lambda: now + 3601)
    assert small.acquire()


def test_cooldown_shared(tmp_path):
    path = tmp_path / 'budget.db'
    a = RateLimiter(path=path)
    a.defer(60)
    b = RateLimiter(path=path)
    assert not b.acquire()
    assert b.remaining_requests() == 0
    assert b.get_stats().retry_after_seconds > 59


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setenv('FINMIND_STATE_PATH', str(tmp_path / 'budget.db'))
    monkeypatch.setenv('FINMIND_REQUESTS_PER_HOUR', '6000')
    monkeypatch.setattr(finmind, '_sleep_backoff', lambda *a: None)
    class Transport:
        calls = []
        status = 200
        payload = {'status': 200, 'data': [{'stock_id': '2330', 'close': 100}]}
        failure = None
        def get(self, *args, **kwargs):
            self.calls.append(kwargs)
            if self.failure:
                failure, self.failure = self.failure, None
                raise failure
            class Response:
                status_code = self.status
                headers = {'Retry-After': '5'}
                def json(_):
                    return self.payload
            return Response()
    t = Transport()
    monkeypatch.setattr(finmind, '_http_session', lambda: t)
    return t


def _fetch(**kwargs):
    return finmind.fetch_dataset('TaiwanStockPrice', date(2026, 9, 7),
                                 date(2026, 9, 7), token='test-secret', data_id='2330', **kwargs)


def test_duplicate_queries_singleflight_and_provenance(transport, tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(lambda _: _fetch(), range(12)))
    assert len(transport.calls) == 1
    assert sum(f.attrs['cache_hit'] for f in frames) == 11
    assert len({f.attrs['retrieved_at'] for f in frames}) == 1
    assert get_rate_limiter().get_stats().requests_in_window == 1
    assert all(b'test-secret' not in p.read_bytes() for p in tmp_path.rglob('*') if p.is_file())
    _fetch(force_refresh=True)
    assert len(transport.calls) == 2


def test_every_retry_is_charged(transport):
    transport.failure = requests.ConnectionError('url contains secret')
    _fetch()
    assert len(transport.calls) == 2
    assert get_rate_limiter().get_stats().requests_in_window == 2


@pytest.mark.parametrize('http,payload_status', [(429, 200), (402, 200), (200, 402), (200, 429)])
def test_quota_response_stops_retries_and_other_workers(transport, http, payload_status):
    transport.status = http
    transport.payload = {'status': payload_status, 'msg': 'test-secret'}
    with pytest.raises(finmind.FinMindQuotaError):
        _fetch()
    with pytest.raises(finmind.FinMindQuotaError):
        _fetch()
    assert len(transport.calls) == 1


def test_empty_data_is_not_cached(transport):
    transport.payload = {'status': 200, 'data': []}
    _fetch()
    _fetch()
    assert len(transport.calls) == 2


def test_invalid_query_cannot_spend_quota(transport):
    with pytest.raises(ValueError):
        _fetch(rate_limit=False)
    with pytest.raises(ValueError):
        finmind.fetch_dataset('TaiwanStockPrice', date.today(), data_id='2330,2317')
    assert not transport.calls


def test_bulk_uses_documented_date_route(transport):
    out = finmind.fetch_dataset_by_stocks('TaiwanStockPrice', date(2026, 9, 7),
        date(2026, 9, 8), ['2330', '2317', '2330'], token='test-secret')
    assert len(transport.calls) == 2
    assert all('data_id' not in c['params'] for c in transport.calls)
    assert set(out.stock_id) == {'2330'}


def test_unsupported_bulk_dataset_uses_single_ids(transport):
    finmind.fetch_dataset_by_stocks('TaiwanStockKBar', date(2026, 9, 7),
        date(2026, 9, 7), ['2330', '2317'], token='test-secret')
    assert [c['params']['data_id'] for c in transport.calls] == ['2330', '2317']


def test_monthly_bulk_queries_only_period_dates(transport):
    finmind.fetch_dataset_by_stocks('TaiwanStockMonthRevenue', date(2026, 8, 15),
        date(2026, 9, 9), ['2330', '2317'], token='test-secret')
    assert len(transport.calls) == 1
    assert transport.calls[0]['params']['start_date'] == '2026-09-01'
    assert 'data_id' not in transport.calls[0]['params']


def test_snapshot_shares_quota_and_uses_short_separate_cache(transport, monkeypatch):
    frame = finmind.fetch_dataset('TaiwanStockTickSnapshot', date(2026,9,11), token='test-secret', data_id='2330')
    assert transport.calls[0]['params'] == {'data_id':'2330'}
    finmind.fetch_dataset('TaiwanStockTickSnapshot', date(2026,9,11), token='test-secret', data_id='2330')
    assert len(transport.calls) == 1
    future=time.time()+11
    monkeypatch.setattr('app.finmind.time.time',lambda:future)
    finmind.fetch_dataset('TaiwanStockTickSnapshot', date(2026,9,11), token='test-secret', data_id='2330')
    assert len(transport.calls) == 2
    assert get_rate_limiter().get_stats().requests_in_window == 2
