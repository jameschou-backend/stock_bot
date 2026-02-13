from __future__ import annotations

from datetime import date

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.models import Base, RawInstitutional, RawPrice, Stock
from skills import data_quality


class _Cfg:
    tz = "Asia/Taipei"
    dq_max_stale_calendar_days = 999
    dq_max_lag_trading_days = 999
    dq_min_stocks_prices = 1
    dq_min_stocks_institutional = 1
    dq_min_stocks_margin = 1
    dq_coverage_ratio_prices = 0.0
    dq_coverage_ratio_institutional = 0.0
    dq_coverage_ratio_margin = 0.0


def _create_report_table(session: Session) -> None:
    session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS data_quality_reports (
                report_date DATE NOT NULL,
                table_name VARCHAR(64) NOT NULL,
                expected_rows BIGINT NULL,
                actual_rows BIGINT NOT NULL,
                missing_ratio DOUBLE NULL,
                max_trading_date DATE NULL,
                notes TEXT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (report_date, table_name)
            )
            """
        )
    )
    session.flush()


def test_data_quality_run_persists_reports(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        _create_report_table(session)
        today = date.today()
        session.add_all(
            [
                Stock(stock_id="2330", security_type="stock", is_listed=True),
                Stock(stock_id="2317", security_type="stock", is_listed=True),
                RawPrice(stock_id="2330", trading_date=today, close=100, volume=1000),
                RawInstitutional(stock_id="2330", trading_date=today, foreign_net=10),
            ]
        )
        session.flush()

        monkeypatch.setattr(data_quality, "start_job", lambda *_args, **_kwargs: "job-1")
        monkeypatch.setattr(data_quality, "finish_job", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            data_quality,
            "_check_institutional_benchmark",
            lambda *_args, **_kwargs: (data_quality.BENCHMARK_MIN_ROWS, None),
        )

        logs = data_quality.run(_Cfg(), session)
        assert logs["data_quality_report_rows"] == 7

        rows = session.execute(
            text(
                """
                SELECT table_name, actual_rows
                FROM data_quality_reports
                WHERE report_date = :report_date
                ORDER BY table_name
                """
            ),
            {"report_date": today},
        ).fetchall()
        assert len(rows) == 7
        assert any(row[0] == "raw_prices" and row[1] == 1 for row in rows)


def test_data_quality_report_upsert_overwrites_same_key(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        _create_report_table(session)
        today = date.today()
        session.add_all(
            [
                Stock(stock_id="2330", security_type="stock", is_listed=True),
                Stock(stock_id="2317", security_type="stock", is_listed=True),
                RawPrice(stock_id="2330", trading_date=today, close=100, volume=1000),
                RawInstitutional(stock_id="2330", trading_date=today, foreign_net=10),
            ]
        )
        session.flush()

        monkeypatch.setattr(data_quality, "start_job", lambda *_args, **_kwargs: "job-1")
        monkeypatch.setattr(data_quality, "finish_job", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            data_quality,
            "_check_institutional_benchmark",
            lambda *_args, **_kwargs: (data_quality.BENCHMARK_MIN_ROWS, None),
        )

        data_quality.run(_Cfg(), session)
        session.add(RawPrice(stock_id="2317", trading_date=today, close=200, volume=2000))
        session.flush()
        data_quality.run(_Cfg(), session)

        count_same_key = session.execute(
            text(
                """
                SELECT COUNT(*) FROM data_quality_reports
                WHERE report_date = :report_date AND table_name = 'raw_prices'
                """
            ),
            {"report_date": today},
        ).scalar()
        assert int(count_same_key or 0) == 1

        actual_rows = session.execute(
            text(
                """
                SELECT actual_rows FROM data_quality_reports
                WHERE report_date = :report_date AND table_name = 'raw_prices'
                """
            ),
            {"report_date": today},
        ).scalar()
        assert int(actual_rows or 0) == 2
