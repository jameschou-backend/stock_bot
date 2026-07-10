"""官方 adj factor 切換腳本（scripts/cutover_official_adj_factors.py）單元測試。

只測純函數（展開/clip 記錄/diff 統計/颱風修復清單），不碰 DB——
DB 寫入路徑僅在 --apply 下執行且由人工決定，dry-run 統計邏輯是這裡的驗收對象。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.cutover_official_adj_factors import (
    clip_and_record,
    compute_cutover_diff,
    expand_official_factors,
    find_market_closure_fix_events,
)

D = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]


def _factors(stock_id, dates, values):
    return pd.DataFrame({
        "stock_id": stock_id, "trading_date": dates, "adj_factor": values,
    })


def _days(mapping):
    """{stock_id: [dates]} → raw_days DataFrame。"""
    rows = [(sid, d) for sid, ds in mapping.items() for d in ds]
    return pd.DataFrame(rows, columns=["stock_id", "trading_date"])


# ──────────────────────────────────────────────
# expand_official_factors
# ──────────────────────────────────────────────

class TestExpand:
    def test_internal_gap_ffill(self):
        """factor 缺日（官方序列漏日）→ 沿用前值，不產生假跳動。"""
        f = _factors("1101", [D[0], D[1], D[3], D[4]], [0.9, 0.9, 1.0, 1.0])
        days = _days({"1101": D})   # raw 有 1/4，factor 缺
        out, n_gap = expand_official_factors(f, days)
        assert n_gap == 1
        s = out.set_index("trading_date")["adj_factor"]
        assert s[pd.Timestamp(D[2])] == pytest.approx(0.9)   # 缺日 ffill 前值

    def test_leading_gap_bfill_and_trailing_ffill(self):
        """窗口前 raw 交易日 → bfill 最早已知 factor；窗口後新交易日 → ffill。"""
        f = _factors("1101", [D[1], D[2]], [0.8, 1.0])
        days = _days({"1101": D})   # raw 比 factor 多頭尾各一天以上
        out, _ = expand_official_factors(f, days)
        s = out.set_index("trading_date")["adj_factor"]
        assert s[pd.Timestamp(D[0])] == pytest.approx(0.8)   # leading bfill
        assert s[pd.Timestamp(D[3])] == pytest.approx(1.0)   # trailing ffill
        assert s[pd.Timestamp(D[4])] == pytest.approx(1.0)
        assert len(out) == len(D)

    def test_stocks_without_events_not_written(self):
        """無事件股票不展開（消費端視同 1.0，與 populate 語義一致）。"""
        f = _factors("1101", [D[0]], [1.0])
        days = _days({"1101": D, "2330": D})
        out, _ = expand_official_factors(f, days)
        assert set(out["stock_id"]) == {"1101"}


# ──────────────────────────────────────────────
# clip_and_record
# ──────────────────────────────────────────────

class TestClipAndRecord:
    def test_no_clip_passthrough(self):
        df = _factors("1101", D, [0.5, 0.5, 1.0, 1.0, 1.0])
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        out, payload = clip_and_record(df)
        assert payload == {}
        np.testing.assert_allclose(out["adj_factor"], df["adj_factor"])

    def test_deep_split_stock_clipped_and_recorded(self):
        """一拆二十 × 歷年配息 → 累積 factor < 0.1，clip 並記錄（5314 pattern）。"""
        df = pd.concat([
            _factors("5314", D, [0.04, 0.04, 0.05, 1.0, 1.0]),
            _factors("1101", D, [0.9] * 5),
        ])
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        out, payload = clip_and_record(df)
        assert set(payload) == {"5314"}
        assert payload["5314"]["clipped_days"] == 3
        assert payload["5314"]["min_ratio"] == pytest.approx(0.04)
        assert out[out.stock_id == "5314"]["adj_factor"].min() == pytest.approx(0.1)
        # 未觸 clip 股票不受影響
        np.testing.assert_allclose(out[out.stock_id == "1101"]["adj_factor"], 0.9)


# ──────────────────────────────────────────────
# compute_cutover_diff（dry-run 統計核心）
# ──────────────────────────────────────────────

class TestCutoverDiff:
    def test_changed_rows_and_buckets(self):
        old = _factors("1101", D, [0.90, 0.90, 1.0, 1.0, 1.0])
        new = _factors("1101", D, [0.95, 0.95, 1.0, 1.0, 1.0])   # 前兩日變 ~5.6%
        stats = compute_cutover_diff(new, old)
        assert stats["n_common_rows"] == 5
        assert stats["n_changed_rows"] == 2
        assert stats["changed_pct_of_common"] == pytest.approx(0.4)
        assert stats["change_magnitude_buckets"]["5%~20%"] == 2
        assert stats["change_magnitude_quantiles"]["max"] == pytest.approx(
            0.95 / 0.90 - 1.0)
        assert stats["top_changed_stocks"][0]["stock_id"] == "1101"

    def test_identical_frames_zero_changed(self):
        old = _factors("1101", D, [0.9] * 5)
        stats = compute_cutover_diff(old, old)
        assert stats["n_changed_rows"] == 0
        assert stats["change_magnitude_quantiles"] == {}
        assert stats["top_changed_stocks"] == []

    def test_dropped_stock_flat_vs_nonflat(self):
        """移除股票分流：factor 全平坦（=1.0）無資訊損失；非平坦列入損失清單。"""
        old = pd.concat([
            _factors("1101", D, [1.0] * 5),                  # 平坦 → 損失無害
            _factors("2330", D, [0.8, 0.8, 1.0, 1.0, 1.0]),  # 非平坦 → 資訊損失
            _factors("3008", D, [0.9] * 5),                  # 留存股
        ])
        new = _factors("3008", D, [0.9] * 5)
        stats = compute_cutover_diff(new, old)
        assert stats["n_stocks_dropped"] == 2
        assert stats["stocks_dropped_nonflat"] == ["2330"]
        assert stats["n_stocks_dropped_nonflat"] == 1
        assert stats["n_rows_only_in_old"] == 10

    def test_added_stock_and_rows(self):
        old = _factors("1101", D, [1.0] * 5)
        new = pd.concat([
            _factors("1101", D, [1.0] * 5),
            _factors("9910", D, [0.5] * 5),   # DB 沒有的股票（官方補缺口）
        ])
        stats = compute_cutover_diff(new, old)
        assert stats["stocks_added"] == ["9910"]
        assert stats["n_rows_only_in_new"] == 5
        assert stats["n_changed_rows"] == 0

    def test_post_freeze_change_counted(self):
        """凍結日之後的變動單獨計數（切換主要動機：凍結後新事件調整）。"""
        old = _factors("1101", D, [1.0] * 5)
        new = _factors("1101", D, [0.95, 0.95, 0.95, 0.95, 1.0])  # 1/8 除息事件
        stats = compute_cutover_diff(new, old, freeze_date=date(2024, 1, 4))
        assert stats["n_changed_rows"] == 4
        assert stats["n_changed_rows_post_freeze"] == 1   # 只有 1/5 在凍結日後且變動
        assert stats["n_stocks_changed_post_freeze"] == 1


# ──────────────────────────────────────────────
# find_market_closure_fix_events（颱風修復清單）
# ──────────────────────────────────────────────

class TestMarketClosureFix:
    MARKET_DAYS = {date(2024, 7, 22), date(2024, 7, 23), date(2024, 7, 26)}

    def _events(self, rows):
        return pd.DataFrame(rows, columns=["stock_id", "event_date"])

    def test_typhoon_day_events_detected(self):
        """2024-07-24/25 凱米颱風停市（週三/週四）：事件日為平日但無交易 → 修復清單。"""
        ev = self._events([
            ("1101", date(2024, 7, 24)),   # 颱風停市日
            ("2330", date(2024, 7, 25)),   # 颱風停市日
            ("3008", date(2024, 7, 23)),   # 正常交易日 → 不列
            ("2603", date(2024, 7, 27)),   # 週六 → 不列（非停市，本來就休市）
        ])
        out = find_market_closure_fix_events(ev, self.MARKET_DAYS)
        assert sorted(out["stock_id"]) == ["1101", "2330"]

    def test_empty_events(self):
        out = find_market_closure_fix_events(
            pd.DataFrame(columns=["stock_id", "event_date"]), self.MARKET_DAYS)
        assert out.empty
