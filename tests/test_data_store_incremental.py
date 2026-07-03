"""data_store 增量重建等價性測試（2026-07-03 健檢效能審計發現 5）。

鐵律：增量 append 產出的 parquet 必須與全量重建 byte-identical（含 dtype 與列序）。
任何不符（歷史列數變動、schema 變動）必須 fallback 全量重建。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import skills.data_store as ds
from app.models import Base, Label, RawPrice
from skills.feature_store import FeatureStore


@pytest.fixture()
def session():
    # StaticPool：pd.read_sql 走 engine 會另開連線，:memory: 需共用同一連線
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture(autouse=True)
def _patch_paths(tmp_path, monkeypatch):
    """cache 路徑導向 tmp，並清 process 級 memoization。"""
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ds, "PRICES_PARQUET", tmp_path / "prices.parquet")
    monkeypatch.setattr(ds, "FEATURES_PARQUET", tmp_path / "features.parquet")
    monkeypatch.setattr(ds, "LABELS_PARQUET", tmp_path / "labels.parquet")
    ds.reset_ensure_memo()
    yield
    ds.reset_ensure_memo()


def _add_prices(session, rows):
    for sid, td, close, vol in rows:
        session.add(RawPrice(stock_id=sid, trading_date=td,
                             open=close, high=close, low=close, close=close, volume=vol))
    session.commit()


def _add_labels(session, rows):
    for sid, td, ret in rows:
        session.add(Label(stock_id=sid, trading_date=td, future_ret_h=ret))
    session.commit()


def _src_stats(session, table):
    from sqlalchemy import text
    mx = str(session.execute(text(f"SELECT max(trading_date) FROM {table}")).scalar() or "")
    n = int(session.execute(text(f"SELECT count(*) FROM {table}")).scalar() or 0)
    return mx, n


# ── prices ────────────────────────────────────────────────────────────────────


def test_prices_incremental_equals_full_rebuild(session, tmp_path):
    """來源只 append 新日期 → 走增量，結果須與全量重建完全相等（值/dtype/列序）。"""
    _add_prices(session, [
        ("1101", date(2026, 1, 2), 10.0, 1000),
        ("2330", date(2026, 1, 2), 500.0, 2000),
        ("1101", date(2026, 1, 3), 10.5, 1100),
        ("2330", date(2026, 1, 3), 505.0, 2100),
    ])
    ds._build_prices(session)  # 首次：全量
    assert (tmp_path / "prices.parquet").exists()

    # append 一個新交易日
    _add_prices(session, [
        ("1101", date(2026, 1, 6), 10.7, 1200),
        ("2330", date(2026, 1, 6), 510.0, 2200),
    ])
    mx, n = _src_stats(session, "raw_prices")
    assert ds._try_incremental_prices(session, mx, n) is True  # 走增量
    inc_result = pd.read_parquet(tmp_path / "prices.parquet")

    # 對照組：刪 cache 全量重建
    (tmp_path / "prices.parquet").unlink()
    ds._build_prices(session)
    full_result = pd.read_parquet(tmp_path / "prices.parquet")

    pd.testing.assert_frame_equal(inc_result, full_result, check_dtype=True, check_exact=True)


def test_prices_history_mutation_falls_back_to_full(session, tmp_path):
    """歷史列數變動（回補/刪除）→ 增量條件不符，須走全量。"""
    _add_prices(session, [
        ("1101", date(2026, 1, 2), 10.0, 1000),
        ("1101", date(2026, 1, 3), 10.5, 1100),
    ])
    ds._build_prices(session)

    # 同時 append 新日期 + 回補歷史列（總列數差 != 新日期列數）
    _add_prices(session, [
        ("2330", date(2026, 1, 2), 500.0, 2000),   # 歷史回補
        ("1101", date(2026, 1, 6), 10.7, 1200),    # 新日期
    ])
    mx, n = _src_stats(session, "raw_prices")
    assert ds._try_incremental_prices(session, mx, n) is False

    # _build_prices 整體仍須產出正確全量結果
    ds._build_prices(session, src_max=mx, src_rows=n)
    result = pd.read_parquet(tmp_path / "prices.parquet")
    assert len(result) == 4


def test_prices_no_new_date_no_incremental(session):
    """來源無新日期 → 不走增量（_ensure 層根本不會觸發重建，此處防禦性驗證）。"""
    _add_prices(session, [("1101", date(2026, 1, 2), 10.0, 1000)])
    ds._build_prices(session)
    mx, n = _src_stats(session, "raw_prices")
    assert ds._try_incremental_prices(session, mx, n) is False


# ── labels ────────────────────────────────────────────────────────────────────


def test_labels_incremental_equals_full_rebuild(session, tmp_path):
    _add_labels(session, [
        ("1101", date(2026, 1, 2), 0.01),
        ("2330", date(2026, 1, 2), -0.02),
    ])
    ds._build_labels(session)

    _add_labels(session, [
        ("1101", date(2026, 1, 3), 0.03),
        ("2330", date(2026, 1, 3), 0.04),
    ])
    mx, n = _src_stats(session, "labels")
    assert ds._try_incremental_labels(session, mx, n) is True
    inc_result = pd.read_parquet(tmp_path / "labels.parquet")

    (tmp_path / "labels.parquet").unlink()
    ds._build_labels(session)
    full_result = pd.read_parquet(tmp_path / "labels.parquet")

    pd.testing.assert_frame_equal(inc_result, full_result, check_dtype=True, check_exact=True)


# ── features（FeatureStore 來源）─────────────────────────────────────────────


def _make_feat_df(dates, sids, base=1.0):
    rows = []
    for td in dates:
        for sid in sids:
            rows.append({"stock_id": sid, "trading_date": td,
                         "f1": base + hash((sid, td)) % 7 * 0.1, "f2": base * 2})
    return pd.DataFrame(rows)


def test_features_incremental_equals_full_rebuild(tmp_path, monkeypatch):
    store_dir = tmp_path / "fs"
    monkeypatch.setenv("FEATURE_STORE_DIR", str(store_dir))
    monkeypatch.setattr("skills.feature_store._STORE_DIR", store_dir)

    fs = FeatureStore()
    fs.write(_make_feat_df([date(2026, 1, 2), date(2026, 1, 3)], ["1101", "2330"]))
    ds._build_features(None)  # 首次：全量（FeatureStore 路徑不需 db_session）

    fs.write(_make_feat_df([date(2026, 1, 6)], ["1101", "2330"]))
    mx, n = str(fs.get_max_date()), fs.row_count()
    assert ds._try_incremental_features(mx, n) is True
    inc_result = pd.read_parquet(tmp_path / "features.parquet")

    (tmp_path / "features.parquet").unlink()
    ds._build_features(None)
    full_result = pd.read_parquet(tmp_path / "features.parquet")

    pd.testing.assert_frame_equal(inc_result, full_result, check_dtype=True, check_exact=True)


def test_features_schema_evolution_falls_back_to_full(tmp_path, monkeypatch):
    """FeatureStore 新增欄位（schema 演進）→ 增量欄位不一致，須走全量。"""
    store_dir = tmp_path / "fs"
    monkeypatch.setenv("FEATURE_STORE_DIR", str(store_dir))
    monkeypatch.setattr("skills.feature_store._STORE_DIR", store_dir)

    fs = FeatureStore()
    fs.write(_make_feat_df([date(2026, 1, 2)], ["1101", "2330"]))
    ds._build_features(None)

    new_df = _make_feat_df([date(2026, 1, 3)], ["1101", "2330"])
    new_df["f3_new"] = 9.9  # 新欄位
    fs.write(new_df)
    mx, n = str(fs.get_max_date()), fs.row_count()
    assert ds._try_incremental_features(mx, n) is False

    ds._build_features(None)  # 全量重建含新欄位
    result = pd.read_parquet(tmp_path / "features.parquet")
    assert "f3_new" in result.columns
    assert len(result) == 4
