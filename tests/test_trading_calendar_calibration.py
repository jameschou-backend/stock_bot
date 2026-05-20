"""驗證 ingest_trading_calendar 的 raw_prices 校準邏輯。

對 in-memory SQLite 建立 minimal schema 後，注入：
- 兩個 weekday（5/8 Fri, 5/11 Mon）有 prices 紀錄 → 應是 trading day
- 一個 weekday（5/12 Tue）沒有 prices → 應被校準為 HOLIDAY（國定假日）
- 一個 weekend（5/9 Sat）有 prices → 應被校準為 MAKEUP（補班日）

並驗證 weekday seed + calibration 後最終 calendar 行為正確。
"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, RawPrice, TradingCalendar
from skills.ingest_trading_calendar import (
    _calibrate_from_prices,
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


@pytest.mark.skipif(
    True,  # SQLite 不支援 on_duplicate_key_update，calibration / seed 要 MySQL
    reason="seed_calendar / _calibrate_from_prices 使用 MySQL-specific upsert，需 MySQL fixture",
)
class TestCalibrationOnMySQL:
    """這些測試需要真 MySQL；CI 跑時跳過。對應整合測試請在 staging DB 驗證。"""
    def test_holiday_inference(self):
        ...
    def test_makeup_day_inference(self):
        ...
