from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, RawPrice, Stock, TradingCalendar
from skills.price_coverage import incomplete_dates, require_market_coverage
from skills import data_quality, daily_pick


@pytest.fixture
def market_session(monkeypatch):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for sid, market in [('2330', 'TWSE'), ('6488', 'TPEX')]:
            session.add(Stock(stock_id=sid, market=market, security_type='stock', is_listed=True))
            session.add(RawPrice(stock_id=sid, trading_date=date(2026, 9, 22), close=100, volume=1000))
        session.add_all([TradingCalendar(trading_date=date(2026, 9, d), is_open=True) for d in (22, 23)])
        session.commit()
        monkeypatch.setattr('app.market_calendar.get_latest_trading_day', lambda _: date(2026, 9, 23))
        yield session
    engine.dispose()


def test_both_missing_or_one_market_missing_fail_until_repaired(market_session):
    s = market_session
    with pytest.raises(ValueError, match='2026-09-23'):
        require_market_coverage(s)
    s.add(RawPrice(stock_id='6488', trading_date=date(2026, 9, 23), close=101, volume=1000))
    s.commit()
    with pytest.raises(ValueError, match='2026-09-23'):
        require_market_coverage(s)
    s.add(RawPrice(stock_id='2330', trading_date=date(2026, 9, 23), close=101, volume=1000))
    s.commit()
    assert require_market_coverage(s)['market_coverage_target'] == '2026-09-23'


def test_market_shortfall_cannot_hide_behind_large_other_market():
    counts = pd.DataFrame([(date(2026, 9, d), m, n) for d, m, n in
        [(22, 'TWSE', 1000), (22, 'TPEX', 800), (23, 'TWSE', 1000), (23, 'TPEX', 700)]],
        columns=['trading_date', 'market', 'rows_count'])
    assert incomplete_dates(counts) == [date(2026, 9, 23)]
    assert incomplete_dates(pd.DataFrame(), required_dates=[date(2026, 9, 23)]) == [date(2026, 9, 23)]


@pytest.mark.parametrize('mode', ['strict', 'research', 'dev'])
def test_partial_market_stops_quality_before_expensive_checks(market_session, monkeypatch, mode):
    finished = []
    monkeypatch.setattr(data_quality, 'start_job', lambda *a, **k: 'test')
    monkeypatch.setattr(data_quality, 'finish_job', lambda *a, **k: finished.append((a, k)))
    monkeypatch.setattr(data_quality, 'check_data_quality', lambda *a: pytest.fail('partial data proceeded'))
    with pytest.raises(ValueError, match='行情不完整'):
        data_quality.run(SimpleNamespace(data_quality_mode=mode), market_session)
    assert finished[-1][0][2] == 'failed'
    assert finished[-1][1]['logs']['market_inputs_blocked']


def test_direct_pick_cannot_publish_with_partial_market(market_session, monkeypatch):
    monkeypatch.setattr(daily_pick, 'start_job', lambda *a, **k: 'test')
    finished = []
    monkeypatch.setattr(daily_pick, 'finish_job', lambda *a, **k: finished.append(a))
    monkeypatch.setattr(daily_pick, '_load_latest_model', lambda *a: pytest.fail('model loaded'))
    with pytest.raises(ValueError, match='行情不完整'):
        daily_pick.run(SimpleNamespace(), market_session)
    assert finished[-1][2] == 'failed'


def test_unpublished_index_blocks_build_preflight(market_session, monkeypatch):
    from skills.market_input_gate import require_market_inputs
    from app.finmind import FinMindError
    monkeypatch.setattr('skills.market_input_gate.require_market_coverage',
                        lambda _: {'market_coverage_target': '2026-09-23'})
    def missing(*args):
        raise FinMindError('TAIEX 尚未更新')
    monkeypatch.setattr(daily_pick, '_load_market_price_df', missing)
    with pytest.raises(FinMindError, match='TAIEX'):
        require_market_inputs(SimpleNamespace(market_filter_enabled=True), market_session)


def test_recent_index_does_not_cache_partial_publication_for_a_day(monkeypatch):
    monkeypatch.setattr('app.config.load_config', lambda: SimpleNamespace(finmind_token='', finmind_requests_per_hour=5400))
    captured = []
    today = date.today()
    def fetch(*args, **kwargs):
        captured.append(kwargs['cache_ttl'])
        return pd.DataFrame({'date': [str(today)], 'stock_id': ['TAIEX'], 'price': [100.]})
    monkeypatch.setattr('app.finmind.fetch_dataset', fetch)
    daily_pick._load_market_price_df(None, today, 200)
    assert captured == [300]


def test_blocked_job_remains_visible_after_runner_rollback(market_session):
    from app.models import Job
    with pytest.raises(ValueError, match='行情不完整'):
        data_quality.run(SimpleNamespace(data_quality_mode='dev'), market_session)
    market_session.rollback()
    job = market_session.query(Job).filter_by(job_name='data_quality_check').one()
    assert job.status == 'failed'
    assert job.logs_json['market_inputs_blocked'] is True
