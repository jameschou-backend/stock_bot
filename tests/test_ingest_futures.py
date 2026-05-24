"""Stage 11.0 期貨 ingest 結構測試（不執行 real ingest，避免動 FinMind quota）。"""
from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest


def test_futures_models_importable():
    from app.models import RawFuturesDaily, RawFuturesInst
    assert hasattr(RawFuturesDaily, "__tablename__")
    assert RawFuturesDaily.__tablename__ == "raw_futures_daily"
    assert RawFuturesInst.__tablename__ == "raw_futures_inst"


def test_futures_daily_columns():
    from app.models import RawFuturesDaily
    cols = {c.name for c in RawFuturesDaily.__table__.columns}
    expected = {"contract_id", "trading_date", "contract_month",
                "open", "high", "low", "close", "volume",
                "open_interest", "settlement_price"}
    assert expected.issubset(cols)


def test_futures_inst_columns():
    from app.models import RawFuturesInst
    cols = {c.name for c in RawFuturesInst.__table__.columns}
    expected = {"contract_id", "trading_date",
                "foreign_long_oi", "foreign_short_oi", "foreign_net_oi",
                "trust_long_oi", "trust_short_oi", "trust_net_oi",
                "dealer_long_oi", "dealer_short_oi", "dealer_net_oi"}
    assert expected.issubset(cols)


def test_ingest_futures_module():
    mod = importlib.import_module("skills.ingest_futures")
    assert hasattr(mod, "run")
    assert hasattr(mod, "_normalize")
    assert hasattr(mod, "_select_near_month")
    assert mod.DEFAULT_CONTRACT == "TX"


def test_ingest_futures_inst_module():
    mod = importlib.import_module("skills.ingest_futures_inst")
    assert hasattr(mod, "run")
    assert hasattr(mod, "_normalize")
    assert mod.DEFAULT_CONTRACT == "TX"


def test_select_near_month_picks_correct_contract():
    from datetime import date
    from skills.ingest_futures import _select_near_month
    # 假 dataset：同 trading_date 三個 contract（近月 / 次月 / 季月）
    df = pd.DataFrame([
        {"contract_date": "202604", "close": 17000, "date": "2026-04-15"},
        {"contract_date": "202606", "close": 17200, "date": "2026-04-15"},
        {"contract_date": "202609", "close": 17500, "date": "2026-04-15"},
    ])
    out = _select_near_month(df, date(2026, 4, 15))
    assert len(out) == 1
    assert out.iloc[0]["contract_date"] == "202604"  # 當月


def test_select_near_month_skips_expired():
    from datetime import date
    from skills.ingest_futures import _select_near_month
    # 都過期 → fallback 最晚
    df = pd.DataFrame([
        {"contract_date": "202601", "close": 16000, "date": "2026-04-15"},
        {"contract_date": "202602", "close": 16500, "date": "2026-04-15"},
    ])
    out = _select_near_month(df, date(2026, 4, 15))
    assert len(out) == 1
    # 沒有當月以後，取最晚
    assert out.iloc[0]["contract_date"] == "202602"


def test_normalize_futures_daily_handles_nan():
    from skills.ingest_futures import _normalize
    df = pd.DataFrame([
        {"date": "2026-04-15", "contract_date": "202604",
         "open": 17000, "max": 17100, "min": 16950, "close": 17050,
         "volume": 12345, "open_interest": 56789, "settlement_price": 17050},
    ])
    out = _normalize(df)
    assert "contract_id" in out.columns
    assert out.iloc[0]["contract_id"] == "TX"
    assert out.iloc[0]["high"] == 17100  # max → high renamed


def test_normalize_futures_inst_handles_multi_dealer():
    """自營商有 (避險) + (自行買賣) 兩 row，應該 aggregate."""
    from skills.ingest_futures_inst import _normalize
    df = pd.DataFrame([
        {"date": "2026-04-15", "futures_id": "TX",
         "institutional_investors": "外資",
         "long_open_interest_volume": 50000, "short_open_interest_volume": 30000},
        {"date": "2026-04-15", "futures_id": "TX",
         "institutional_investors": "投信",
         "long_open_interest_volume": 5000, "short_open_interest_volume": 6000},
        {"date": "2026-04-15", "futures_id": "TX",
         "institutional_investors": "自營商(避險)",
         "long_open_interest_volume": 8000, "short_open_interest_volume": 4000},
        {"date": "2026-04-15", "futures_id": "TX",
         "institutional_investors": "自營商(自行買賣)",
         "long_open_interest_volume": 2000, "short_open_interest_volume": 1500},
    ])
    out = _normalize(df)
    assert len(out) == 1
    assert out.iloc[0]["foreign_long_oi"] == 50000
    assert out.iloc[0]["foreign_net_oi"] == 20000  # 50000-30000
    # 自營商兩 row 合計
    assert out.iloc[0]["dealer_long_oi"] == 10000  # 8000+2000
    assert out.iloc[0]["dealer_short_oi"] == 5500  # 4000+1500
    assert out.iloc[0]["dealer_net_oi"] == 4500


def test_normalize_futures_inst_filters_non_tx_contract():
    """非 TX contract 應該被 filter 掉."""
    from skills.ingest_futures_inst import _normalize
    df = pd.DataFrame([
        {"date": "2026-04-15", "futures_id": "MTX",
         "institutional_investors": "外資",
         "long_open_interest_volume": 10000, "short_open_interest_volume": 5000},
    ])
    out = _normalize(df)
    assert out.empty
