"""有價格的日期只代表資料存在，不用缺行情推定假日。"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, RawPrice, TradingCalendar
from skills.ingest_trading_calendar import (
    _apply_official_closures,
    _seed_rows,
    seed_calendar,
)


@pytest.fixture
def session():
    # SQLite in-memory；MySQL-specific upsert 不能用，所以下面 fixture 用直接 add
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[RawPrice.__table__, TradingCalendar.__table__])
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _add_price(session, sid, d):
    session.add(RawPrice(stock_id=sid, trading_date=d, open=100, high=101, low=99, close=100, volume=1000))


class TestActualTradingDates:
    def test_returns_dates_with_prices(self, session):
        _add_price(session, "2330", date(2026, 5, 8))
        _add_price(session, "2330", date(2026, 5, 11))
        _add_price(session, "1101", date(2026, 5, 11))  # 同日多檔
        session.commit()

        from skills.ingest_trading_calendar import _actual_trading_dates
        actual = _actual_trading_dates(session, date(2026, 5, 1), date(2026, 5, 31))
        assert actual == {date(2026, 5, 8), date(2026, 5, 11)}

    def test_empty_when_no_prices(self, session):
        from skills.ingest_trading_calendar import _actual_trading_dates
        actual = _actual_trading_dates(session, date(2026, 1, 1), date(2026, 12, 31))
        assert actual == set()


def test_official_typhoon_closure_overrides_stale_provider_and_seed():
    stale = [{"trading_date": date(2026, 10, 23), "is_open": True,
              "session_type": "FULL", "note": "FinMind TaiwanStockTradingDate"}]
    corrected = _apply_official_closures(stale)
    assert corrected[0]["is_open"] is False
    assert corrected[0]["session_type"] == "HOLIDAY"
    assert "https://www.twse.com.tw/" in corrected[0]["note"]
    assert stale[0]["is_open"] is True
    assert _apply_official_closures(corrected) == corrected
    assert _seed_rows(date(2026, 10, 23), date(2026, 10, 23))[0]["is_open"] is False


def test_official_closure_does_not_guess_other_missing_price_days():
    rows = [{"trading_date": date(2026, 10, 22), "is_open": True,
             "session_type": "FULL", "note": "FinMind TaiwanStockTradingDate"}]
    assert _apply_official_closures(rows) == rows
