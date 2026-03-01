from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, StockStatusHistory
from skills import tradability_filter


def test_tradability_filter_excludes_non_tradable_statuses():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                StockStatusHistory(id=1, stock_id="1111", effective_date=date(2026, 1, 1), status_type="delisted"),
                StockStatusHistory(id=2, stock_id="2222", effective_date=date(2026, 1, 1), status_type="halt"),
                StockStatusHistory(id=3, stock_id="3333", effective_date=date(2026, 1, 1), status_type="listed"),
            ]
        )
        session.flush()

        universe = pd.DataFrame({"stock_id": ["1111", "2222", "3333", "4444"]})
        filtered, stats = tradability_filter.filter_universe(
            session,
            universe,
            asof_date=date(2026, 2, 1),
            return_stats=True,
        )

        ids = set(filtered["stock_id"].tolist())
        assert "1111" not in ids
        assert "2222" not in ids
        assert "3333" in ids
        assert "4444" in ids  # 缺狀態時允許通過（但需記錄）
        assert stats["excluded_count"] == 2
        assert stats["missing_status_count"] == 1


def test_is_tradable_returns_reason():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            StockStatusHistory(id=1, stock_id="9999", effective_date=date(2026, 1, 1), status_type="suspend")
        )
        session.flush()
        ok, reasons = tradability_filter.is_tradable(session, "9999", date(2026, 2, 1))
        assert ok is False
        assert "SUSPEND" in reasons
