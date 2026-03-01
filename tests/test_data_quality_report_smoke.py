from __future__ import annotations

from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, RawInstitutional, RawPrice, Stock, TradingCalendar
from scripts.data_quality_report import _generate_report_in_session


class _Cfg:
    data_quality_mode = "research"
    dq_coverage_ratio_prices = 0.7
    dq_coverage_ratio_institutional = 0.5
    dq_coverage_ratio_margin = 0.5


def test_data_quality_report_smoke():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    asof = date(2026, 2, 10)
    with Session(engine) as session:
        session.add_all(
            [
                Stock(stock_id="2330", security_type="stock", is_listed=True),
                TradingCalendar(trading_date=date(2026, 2, 9), is_open=True, session_type="FULL"),
                TradingCalendar(trading_date=date(2026, 2, 10), is_open=True, session_type="FULL"),
                RawPrice(stock_id="2330", trading_date=date(2026, 2, 9), close=100, open=100, high=101, low=99, volume=1000),
                RawPrice(stock_id="2330", trading_date=date(2026, 2, 10), close=101, open=101, high=102, low=100, volume=1200),
                RawInstitutional(stock_id="2330", trading_date=date(2026, 2, 10), foreign_net=10),
            ]
        )
        session.flush()

        report = _generate_report_in_session(days=30, asof=asof, config=_Cfg(), session=session)
        assert report["asof"] == asof.isoformat()
        assert "datasets" in report
        assert any(ds["dataset"] == "raw_prices" for ds in report["datasets"])
