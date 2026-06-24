"""DATA_STORE_FREEZE 凍結模式回歸測試。

背景（2026-06-22）：回測對照實驗期間，data_store 的 lazy parquet cache
若被併發 daily pipeline / cron 觸發重建，會讓橫跨重建的 run 讀到不一致快照
（實測同指令 Sharpe 1.035 vs 0.805）。DATA_STORE_FREEZE=1 凍結 cache、跳過
重建，確保對照實驗可重現。此測試鎖定該行為。
"""
from pathlib import Path

import pandas as pd
import pytest

import skills.data_store as ds


def _make_cache(tmp_path: Path, monkeypatch) -> None:
    """在 tmp_path 建三個假 cache parquet 並 monkeypatch 路徑常數。"""
    for name, const in (
        ("prices", "PRICES_PARQUET"),
        ("features", "FEATURES_PARQUET"),
        ("labels", "LABELS_PARQUET"),
    ):
        p = tmp_path / f"{name}.parquet"
        pd.DataFrame({"trading_date": ["2026-05-20"]}).to_parquet(p)
        monkeypatch.setattr(ds, const, p)
    monkeypatch.setattr(ds, "CACHE_DIR", tmp_path)


def _forbid_rebuild(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("凍結模式不應觸發重建")
    monkeypatch.setattr(ds, "_build_prices", _boom)
    monkeypatch.setattr(ds, "_build_features", _boom)
    monkeypatch.setattr(ds, "_build_labels", _boom)


def test_freeze_skips_rebuild_when_cache_present(tmp_path, monkeypatch):
    """freeze=1 且 cache 齊全 → 不重建、不碰 db_session（傳 None 也安全）。"""
    _make_cache(tmp_path, monkeypatch)
    _forbid_rebuild(monkeypatch)
    monkeypatch.setenv("DATA_STORE_FREEZE", "1")
    ds._ensure(None)  # 不應 raise，也不應呼叫 _build_* 或 db_session


@pytest.mark.parametrize("val", ["1", "true", "on", "YES"])
def test_freeze_truthy_values(tmp_path, monkeypatch, val):
    _make_cache(tmp_path, monkeypatch)
    _forbid_rebuild(monkeypatch)
    monkeypatch.setenv("DATA_STORE_FREEZE", val)
    ds._ensure(None)


def test_freeze_raises_when_cache_missing(tmp_path, monkeypatch):
    """freeze=1 但缺 cache → 明確 RuntimeError（非 silent fallback）。"""
    _make_cache(tmp_path, monkeypatch)
    (tmp_path / "features.parquet").unlink()
    monkeypatch.setenv("DATA_STORE_FREEZE", "1")
    with pytest.raises(RuntimeError, match="DATA_STORE_FREEZE"):
        ds._ensure(None)


@pytest.mark.parametrize("val", ["", "0", "false", "off", "no"])
def test_freeze_off_values_do_not_skip(tmp_path, monkeypatch, val):
    """falsy / 未設值 → 不進凍結分支：以「缺 cache 不報 freeze 錯」反證。
    （正常路徑碰 db_session=None 會以其他方式失敗，但不會是 freeze 的 RuntimeError）。"""
    _make_cache(tmp_path, monkeypatch)
    (tmp_path / "features.parquet").unlink()  # 缺檔；若誤入凍結分支會報 freeze RuntimeError
    if val:
        monkeypatch.setenv("DATA_STORE_FREEZE", val)
    else:
        monkeypatch.delenv("DATA_STORE_FREEZE", raising=False)
    _forbid_rebuild(monkeypatch)  # 正常路徑會嘗試重建 → 觸發 AssertionError（證明沒走 freeze）
    with pytest.raises(Exception) as ei:
        ds._ensure(None)
    assert "DATA_STORE_FREEZE" not in str(ei.value)
