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


# ── staleness / memoization（2026-07-03 健檢 P1-4）────────────────────────────


@pytest.fixture(autouse=True)
def _reset_memo():
    """每個測試前後清 process 級 memoization，避免測試間互相污染。"""
    ds.reset_ensure_memo()
    yield
    ds.reset_ensure_memo()


class _FakeSession:
    """依 SQL 內容回傳 max(trading_date) / count(*) 的假 session。"""

    def __init__(self, px_max, px_rows, lbl_max, lbl_rows):
        self._vals = {
            ("raw_prices", "max"): px_max,
            ("raw_prices", "count"): px_rows,
            ("labels", "max"): lbl_max,
            ("labels", "count"): lbl_rows,
        }

    def execute(self, stmt):
        s = str(stmt).lower()
        table = "raw_prices" if "raw_prices" in s else "labels"
        kind = "count" if "count(" in s else "max"
        val = self._vals[(table, kind)]

        class _R:
            def scalar(self_inner):
                return val

        return _R()


class _FakeFeatureStore:
    """features 分支永遠判定不 stale（本組測試只驗 prices/labels 邏輯）。"""

    def get_max_date(self):
        return None

    def row_count(self):
        return None


def _patch_fs(monkeypatch):
    import skills.feature_store as fs_mod

    monkeypatch.setattr(fs_mod, "FeatureStore", _FakeFeatureStore)


def _record_rebuilds(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ds, "_build_prices", lambda *_a, **_k: calls.append("prices"))
    monkeypatch.setattr(ds, "_build_features", lambda *_a, **_k: calls.append("features"))
    monkeypatch.setattr(ds, "_build_labels", lambda *_a, **_k: calls.append("labels"))
    return calls


def test_row_count_mismatch_triggers_rebuild(tmp_path, monkeypatch):
    """max_date 相同但列數不同（歷史內容重建）→ 必須重建 cache。

    舊行為只比 max_date：force_recompute / adj factor 重灌後 max_date 不變，
    cache 默默供應修正前的舊快照。
    """
    _make_cache(tmp_path, monkeypatch)  # cache: max=2026-05-20, rows=1
    _patch_fs(monkeypatch)
    monkeypatch.delenv("DATA_STORE_FREEZE", raising=False)
    calls = _record_rebuilds(monkeypatch)

    session = _FakeSession(px_max="2026-05-20", px_rows=999, lbl_max="2026-05-20", lbl_rows=1)
    ds._ensure(session)
    assert "prices" in calls, "列數不符（999 != 1）應觸發 prices 重建（P1-4 迴歸）"
    assert "labels" not in calls, "labels max/rows 皆相符，不應重建"


def test_matching_snapshot_no_rebuild(tmp_path, monkeypatch):
    """max_date 與列數皆相符 → 不重建。"""
    _make_cache(tmp_path, monkeypatch)
    _patch_fs(monkeypatch)
    monkeypatch.delenv("DATA_STORE_FREEZE", raising=False)
    calls = _record_rebuilds(monkeypatch)

    session = _FakeSession(px_max="2026-05-20", px_rows=1, lbl_max="2026-05-20", lbl_rows=1)
    ds._ensure(session)
    assert calls == []


def test_ensure_memoized_within_process(tmp_path, monkeypatch):
    """同 process 第二次 _ensure 不再檢查/重建：run 內保證單一快照。

    背景：rolling 回測逐 fold 呼叫 get_*，若 pipeline 併發更新 DB，
    同一 run 的前後 fold 可能讀到兩個不同快照。
    """
    _make_cache(tmp_path, monkeypatch)
    _patch_fs(monkeypatch)
    monkeypatch.delenv("DATA_STORE_FREEZE", raising=False)
    calls = _record_rebuilds(monkeypatch)

    fresh = _FakeSession(px_max="2026-05-20", px_rows=1, lbl_max="2026-05-20", lbl_rows=1)
    ds._ensure(fresh)
    assert calls == []

    # 來源變 stale（模擬 pipeline 中途更新 DB）→ 同 process 內不得重建
    stale = _FakeSession(px_max="2026-07-03", px_rows=999, lbl_max="2026-07-03", lbl_rows=999)
    ds._ensure(stale)
    assert calls == [], "memoization 應鎖定首次快照，run 中途不得換資料"

    # reset 後恢復檢查
    ds.reset_ensure_memo()
    ds._ensure(stale)
    assert "prices" in calls and "labels" in calls
