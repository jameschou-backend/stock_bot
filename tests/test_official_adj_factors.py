"""官方 adj factor 引擎（skills/official_adj_factors.py）單元測試。

覆蓋：
1. 事件比率 → 累積 factor 數學（task fixture：一檔股票兩次除息 + 一次減資）
2. 四個官方 endpoint parser（真實 payload 節錄 fixture）
3. 對帳分類邏輯（matched / 缺事件 / 比率差 / 減資 / 現金增資 / clip / 停牌對應）
4. data_quality adj factor 新鮮度檢查
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from skills import official_adj_factors as oaf


# ──────────────────────────────────────────────
# 1. 事件比率 → 累積 factor
# ──────────────────────────────────────────────

class TestComputeStockFactorSeries:
    # task fixture：兩次除息（r=0.95）+ 一次減資（r=2.0）
    DATES = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10, 11, 12, 15)]
    # d3=1/4 除息、d6=1/9 除息、d9=1/12 減資恢復買賣
    EVENTS = [(date(2024, 1, 4), 0.95), (date(2024, 1, 9), 0.95), (date(2024, 1, 12), 2.0)]

    def test_two_dividends_one_reduction(self):
        ev_dates, ratios = zip(*self.EVENTS)
        f = oaf.compute_stock_factor_series(list(ev_dates), list(ratios), self.DATES)
        # 最近段（>= 最後事件日）= 1.0
        np.testing.assert_allclose(f[8:], [1.0, 1.0])
        # 減資前、第二次除息後：×2.0（減資彌補虧損方向 factor > 1）
        np.testing.assert_allclose(f[5:8], [2.0, 2.0, 2.0])
        # 兩次除息之間：×0.95
        np.testing.assert_allclose(f[2:5], [1.9, 1.9, 1.9])
        # 最早段：再 ×0.95
        np.testing.assert_allclose(f[:2], [1.805, 1.805])

    def test_event_day_belongs_to_after_segment(self):
        """除權息日當天開盤即以參考價為基準 → 事件日 factor 屬事件後段。"""
        f = oaf.compute_stock_factor_series([date(2024, 1, 8)], [0.9], self.DATES)
        assert f[4] == pytest.approx(1.0)   # 1/8 事件日當天
        assert f[3] == pytest.approx(0.9)   # 1/5 事件前

    def test_event_between_trading_days_ffill(self):
        """事件日落在非交易日（停牌）：更早日期照樣累乘，步階天然 ffill。"""
        f = oaf.compute_stock_factor_series([date(2024, 1, 6)], [0.5], self.DATES)  # 週六
        np.testing.assert_allclose(f[:4], 0.5)   # <= 1/5
        np.testing.assert_allclose(f[4:], 1.0)   # >= 1/8
        # 事件間缺日沿用前值（等值步階，無單日假跳動）
        assert len(set(np.round(f[:4], 12))) == 1

    def test_same_day_events_multiply(self):
        f = oaf.compute_stock_factor_series(
            [date(2024, 1, 8), date(2024, 1, 8)], [0.9, 0.5], self.DATES
        )
        assert f[0] == pytest.approx(0.45)

    def test_no_events_all_ones(self):
        f = oaf.compute_stock_factor_series([], [], self.DATES)
        np.testing.assert_allclose(f, 1.0)

    def test_unsorted_events_ok(self):
        ev_dates, ratios = zip(*reversed(self.EVENTS))
        f = oaf.compute_stock_factor_series(list(ev_dates), list(ratios), self.DATES)
        assert f[0] == pytest.approx(1.805)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            oaf.compute_stock_factor_series([date(2024, 1, 4)], [], self.DATES)
        with pytest.raises(ValueError):
            oaf.compute_stock_factor_series([date(2024, 1, 4)], [-0.5], self.DATES)


class TestBuildFactorFrame:
    def test_only_stocks_with_events(self):
        events = oaf.events_to_dataframe([
            oaf.AdjEvent("1101", date(2024, 1, 8), "TWSE", oaf.SOURCE_EX_RIGHTS,
                         "息", prev_close=100.0, ref_price=95.0),
        ])
        td = pd.DataFrame({
            "stock_id": ["1101"] * 4 + ["2330"] * 4,
            "trading_date": [date(2024, 1, d) for d in (4, 5, 8, 9)] * 2,
        })
        out = oaf.build_factor_frame(events, td)
        assert set(out["stock_id"]) == {"1101"}  # 無事件股票不輸出（消費端視同 1.0）
        f = out.set_index("trading_date")["adj_factor"]
        assert f[date(2024, 1, 5)] == pytest.approx(0.95)
        assert f[date(2024, 1, 8)] == pytest.approx(1.0)

    def test_empty_events(self):
        out = oaf.build_factor_frame(
            pd.DataFrame(columns=oaf.EVENT_COLUMNS),
            pd.DataFrame({"stock_id": ["1101"], "trading_date": [date(2024, 1, 4)]}),
        )
        assert out.empty


# ──────────────────────────────────────────────
# 2. Parsers（真實 payload 節錄）
# ──────────────────────────────────────────────

TWSE_EX_RIGHTS_PAYLOAD = {
    "stat": "OK",
    "fields": ["資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價",
               "權值+息值", "權/息", "漲停價格", "跌停價格", "開盤競價基準",
               "減除股利參考價", "詳細資料", "最近一次申報資料 季別/日期",
               "最近一次申報每股 (單位)淨值", "最近一次申報每股 (單位)盈餘"],
    "data": [
        ["113年01月04日", "2454", "聯發科", "953.00", "928.40", "24.600000", "息",
         "1,020.00", "836.00", "928.00", "928.40", "2454,20240104", "", "244.52", "15.17"],
        ["113年01月17日", "006203", "元大MSCI台灣", "64.50", "62.70", "1.800000", "息",
         "68.95", "56.45", "62.70", "62.70", "006203,20240117", "", "", ""],
        ["113年01月10日", "6442", "光聖", "67.20", "65.94", "1.254762", "權",
         "73.90", "59.40", "67.20", "67.20", "6442,20240110", "", "90.51", "7.82"],
    ],
}

TWSE_REDUCTION_PAYLOAD = {
    "stat": "OK",
    "fields": ["恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價",
               "漲停價格", "跌停價格", "開盤競價基準", "除權參考價", "減資原因", "詳細資料"],
    "data": [
        ["113/01/22", "3432", "台端", "10.65", "19.69", "21.65", "17.75", "19.70",
         "--", "彌補虧損", "3432  ,20240110"],
    ],
}

TPEX_EX_RIGHTS_PAYLOAD = {
    "date": "20240101~20240131",
    "tables": [{
        "totalCount": 2,
        "fields": ["除權息日期", "代號", "名稱", "除權息前收盤價", "除權息參考價", "權值",
                   "息值", "權值+息值", "權/息", "漲停價", "跌停價", "開始交易基準價",
                   "減除股利參考價", "現金股利", "每仟股無償配股", "現金增資股數",
                   "現金增資認購價", "公開承銷股數", "員工認購股數", "原股東認購股數",
                   "按持股比例仟股認購"],
        "data": [
            ["113/01/03", "6629", "泰金-KY           ", "55.00", "53.50", "0.000000",
             "1.500000", "1.500000", "除息", "58.80", "48.15", "53.50", "53.50",
             "1.50000000", "0.00000000", "0", "0.00", "0", "0", "0", "0.00000000"],
            # 現金增資 only 除權：參考價 38.72 ≠ 開始交易基準價 39.25（=前收盤）
            ["113/01/10", "1799", "易威              ", "39.25", "38.72", "0.528412",
             "0.000000", "0.528412", "除權", "43.15", "34.85", "39.25", "39.25",
             "0.00000000", "0.00000000", "6000000", "28.60", "600000", "600000",
             "4800000", "41.76517920"],
        ],
    }],
}

TPEX_REDUCTION_PAYLOAD = {
    "date": "20240101~20241231",
    "tables": [{
        "totalCount": 1,
        "fields": ["恢復買賣日期", "股票代號", "名稱", "最後交易日之收盤價格",
                   "減資恢復買賣開始日參考價格", "漲停價格", "跌停價格", "開始交易基準價",
                   "除權參考價", "減資原因", "詳細資料"],
        "data": [
            ["1130205", "3064", "泰偉", "10.65", "35.50", "39.05", "31.95", "35.50",
             "0.00", "彌補虧損", "<table><tr><th>股票代號</th></tr></table>"],
        ],
    }],
}


class TestParsers:
    def test_twse_ex_rights(self):
        events = oaf.parse_twse_ex_rights(TWSE_EX_RIGHTS_PAYLOAD)
        assert len(events) == 3
        mtk = events[0]
        assert (mtk.stock_id, mtk.event_date, mtk.event_type) == ("2454", date(2024, 1, 4), "息")
        assert mtk.ratio == pytest.approx(928.40 / 953.00)
        # 開盤競價基準只差 tick rounding（0.4 元）→ 不觸發現金增資旗標
        assert mtk.cash_increase_suspected is False

    def test_twse_ex_rights_chinese_date_format(self):
        assert oaf._parse_roc_date("113年01月04日") == date(2024, 1, 4)
        assert oaf._parse_roc_date("113/01/22") == date(2024, 1, 22)
        assert oaf._parse_roc_date("1130205") == date(2024, 2, 5)

    def test_twse_ex_rights_stat_no_data_is_empty(self):
        assert oaf.parse_twse_ex_rights({"stat": "很抱歉，沒有符合條件的資料!"}) == []

    def test_twse_transient_error_stat_raises(self):
        """TWSE 偶發回無意義錯誤 stat（實測：2026-04 chunk 回「查詢開始日期小於
        92年5月5日」）——必須 raise 讓上游重試，靜默當空結果會漏抓整月事件。"""
        from app.twse_client import TWSEError
        with pytest.raises(TWSEError):
            oaf.parse_twse_ex_rights({"stat": "查詢開始日期小於92年5月5日，請重新查詢!"})
        with pytest.raises(TWSEError):
            oaf.parse_twse_capital_reduction({"stat": "查詢結束日期小於查詢開始日期，請重新查詢!"})

    def test_twse_capital_reduction(self):
        events = oaf.parse_twse_capital_reduction(TWSE_REDUCTION_PAYLOAD)
        assert len(events) == 1
        ev = events[0]
        assert (ev.stock_id, ev.event_date, ev.event_type) == ("3432", date(2024, 1, 22), "減資")
        assert ev.ratio == pytest.approx(19.69 / 10.65)  # 彌補虧損減資 → 比率 > 1
        assert ev.reason == "彌補虧損"

    def test_tpex_ex_rights(self):
        events = oaf.parse_tpex_ex_rights(TPEX_EX_RIGHTS_PAYLOAD)
        assert len(events) == 2
        assert events[0].stock_id == "6629"
        assert events[0].event_type == "息"     # 「除息」normalize 成「息」
        assert events[0].ratio == pytest.approx(53.50 / 55.00)
        cash = events[1]
        assert cash.event_type == "權"
        assert cash.cash_increase_suspected is True   # 現金增資股數 > 0
        assert cash.ratio == pytest.approx(38.72 / 39.25)
        assert cash.ratio_opening == pytest.approx(1.0)  # 開始交易基準價 = 前收盤

    def test_tpex_capital_reduction(self):
        events = oaf.parse_tpex_capital_reduction(TPEX_REDUCTION_PAYLOAD)
        assert len(events) == 1
        ev = events[0]
        assert (ev.stock_id, ev.event_date) == ("3064", date(2024, 2, 5))
        assert ev.ratio == pytest.approx(35.50 / 10.65)

    def test_events_to_dataframe_filters_non_4digit_and_bad_ratio(self):
        events = oaf.parse_twse_ex_rights(TWSE_EX_RIGHTS_PAYLOAD)
        events.append(oaf.AdjEvent("9999", date(2024, 1, 4), "TWSE", oaf.SOURCE_EX_RIGHTS,
                                   "息", prev_close=None, ref_price=50.0))  # 無前收盤 → skip
        df = oaf.events_to_dataframe(events)
        assert "006203" not in set(df["stock_id"])   # 6 碼 ETF 濾掉
        assert "9999" not in set(df["stock_id"])     # 無效比率濾掉
        assert set(df["stock_id"]) == {"2454", "6442"}


# ──────────────────────────────────────────────
# 3. 對帳分類
# ──────────────────────────────────────────────

def _mk_event(stock_id, event_date, ratio, *, source=oaf.SOURCE_EX_RIGHTS,
              ratio_opening=None, cash=False):
    """直接組 events_df 一列（prev_close=100 基準）。"""
    return {
        "stock_id": stock_id, "event_date": event_date, "market": "TWSE",
        "source": source, "event_type": "減資" if source == oaf.SOURCE_CAPITAL_REDUCTION else "息",
        "prev_close": 100.0, "ref_price": 100.0 * ratio,
        "opening_ref": (100.0 * ratio_opening) if ratio_opening is not None else None,
        "ratio": ratio, "ratio_opening": ratio_opening,
        "cash_increase_suspected": cash, "reason": "", "payload_json": "{}",
    }


def _mk_snapshot(stock_id, dates, factors):
    return pd.DataFrame({
        "stock_id": stock_id, "trading_date": dates, "adj_factor": factors,
    })


DATES_5 = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]


class TestReconcile:
    def test_perfect_match(self):
        events = pd.DataFrame([_mk_event("1111", date(2024, 1, 4), 0.95)])
        snap = _mk_snapshot("1111", DATES_5, [0.95, 0.95, 1.0, 1.0, 1.0])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(1.0)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_MATCHED
        assert res.extra_jumps.empty

    def test_missing_in_snapshot(self):
        events = pd.DataFrame([_mk_event("2222", date(2024, 1, 4), 0.90)])
        snap = _mk_snapshot("2222", DATES_5, [1.0] * 5)  # 快照完全沒調整
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(0.0)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_MISSING_IN_SNAPSHOT

    def test_missing_capital_reduction_classified(self):
        events = pd.DataFrame([
            _mk_event("3333", date(2024, 1, 4), 2.0, source=oaf.SOURCE_CAPITAL_REDUCTION),
        ])
        snap = _mk_snapshot("3333", DATES_5, [1.0] * 5)
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_MISSING_REDUCTION

    def test_cash_increase_matched_via_opening_ref(self):
        """FinMind 未還原現金增資認購權：參考價比率 0.95 不合、開盤競價基準比率 1.0 合。"""
        events = pd.DataFrame([
            _mk_event("4444", date(2024, 1, 4), 0.95, ratio_opening=1.0, cash=True),
        ])
        snap = _mk_snapshot("4444", DATES_5, [1.0] * 5)
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(1.0)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_MATCHED_OPENING_REF
        assert res.summary["n_matched_via_opening_ref"] == 1

    def test_ratio_diff_cash_increase_classified(self):
        events = pd.DataFrame([
            _mk_event("5555", date(2024, 1, 4), 0.90, ratio_opening=0.98, cash=True),
        ])
        snap = _mk_snapshot("5555", DATES_5, [0.95, 0.95, 1.0, 1.0, 1.0])  # 快照 0.95 兩邊都不合
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_RATIO_DIFF_CASH_INCREASE

    def test_suspension_maps_to_next_snapshot_date(self):
        """減資停牌：事件日不在快照交易日 → 對應到恢復買賣後第一個快照日。"""
        events = pd.DataFrame([
            _mk_event("6666", date(2024, 1, 5), 2.0, source=oaf.SOURCE_CAPITAL_REDUCTION),
        ])
        # 1/5 停牌無快照列；1/8 恢復，之前 factor 2.0 → 之後 1.0
        snap = _mk_snapshot("6666",
                            [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 8)],
                            [2.0, 2.0, 2.0, 1.0])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(1.0)
        assert res.event_results.iloc[0]["mapped_date"] == date(2024, 1, 8)

    def test_clip_touched_excluded_from_denominator(self):
        events = pd.DataFrame([
            _mk_event("7777", date(2024, 1, 4), 0.95),
            _mk_event("1111", date(2024, 1, 4), 0.95),
        ])
        snap = pd.concat([
            _mk_snapshot("7777", DATES_5, [1.0] * 5),
            _mk_snapshot("1111", DATES_5, [0.95, 0.95, 1.0, 1.0, 1.0]),
        ])
        res = oaf.reconcile_events_vs_snapshot(events, snap, clip_touched={"7777"})
        assert res.summary["n_eligible"] == 1   # clip 股不進分母
        assert res.match_rate == pytest.approx(1.0)
        clip_row = res.event_results[res.event_results["stock_id"] == "7777"].iloc[0]
        assert clip_row["reason"] == oaf.REASON_CLIP_TOUCHED

    def test_stock_without_snapshot(self):
        events = pd.DataFrame([_mk_event("8888", date(2024, 1, 4), 0.95)])
        snap = _mk_snapshot("1111", DATES_5, [1.0] * 5)
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.summary["n_eligible"] == 0
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_STOCK_NO_SNAPSHOT

    def test_event_before_snapshot_coverage(self):
        events = pd.DataFrame([_mk_event("1111", date(2024, 1, 2), 0.95)])  # 第一個快照日
        snap = _mk_snapshot("1111", DATES_5, [1.0] * 5)
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_OUT_OF_COVERAGE

    def test_extra_snapshot_jump_reported(self):
        events = pd.DataFrame([_mk_event("1111", date(2024, 1, 4), 0.95)])
        snap = pd.concat([
            _mk_snapshot("1111", DATES_5, [0.95, 0.95, 1.0, 1.0, 1.0]),
            _mk_snapshot("9998", DATES_5, [0.9, 0.9, 0.9, 1.0, 1.0]),  # 官方無事件的跳動
        ])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.summary["n_extra_snapshot_jumps"] == 1
        assert res.extra_jumps.iloc[0]["stock_id"] == "9998"
        assert res.extra_jumps.iloc[0]["trading_date"] == date(2024, 1, 5)

    def test_same_gap_multiple_events_combined(self):
        """停牌期間除息 + 減資落在同一 gap：官方比率相乘後比對。"""
        events = pd.DataFrame([
            _mk_event("1111", date(2024, 1, 4), 0.95),
            _mk_event("1111", date(2024, 1, 5), 2.0, source=oaf.SOURCE_CAPITAL_REDUCTION),
        ])
        snap = _mk_snapshot("1111",
                            [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 8)],
                            [1.9, 1.9, 1.0])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(1.0)
        np.testing.assert_allclose(res.event_results["r_official_gap"].to_numpy(), 1.9)


# ──────────────────────────────────────────────
# 4. data_quality adj factor 新鮮度
# ──────────────────────────────────────────────

class TestAdjFactorFreshness:
    @staticmethod
    def _session():
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from app.models import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return Session(engine)

    def test_warns_when_adj_factor_stale(self):
        from app.models import PriceAdjustFactor, RawPrice
        from skills import data_quality
        with self._session() as session:
            session.add(RawPrice(stock_id="2330", trading_date=date(2026, 7, 9),
                                 close=100, volume=1000))
            session.add(PriceAdjustFactor(stock_id="2330", trading_date=date(2026, 6, 23),
                                          adj_factor=1.0))
            session.flush()
            issues, metrics = data_quality._check_adj_factor_freshness(session, max_lag_days=7)
        assert len(issues) == 1
        assert issues[0].category == "adj_factor_stale"
        assert issues[0].severity == "warning"   # Phase 1：切換生產前不 fail
        assert metrics["adj_factor_lag_vs_prices_days"] == 16

    def test_no_issue_within_threshold(self):
        from app.models import PriceAdjustFactor, RawPrice
        from skills import data_quality
        with self._session() as session:
            session.add(RawPrice(stock_id="2330", trading_date=date(2026, 7, 9),
                                 close=100, volume=1000))
            session.add(PriceAdjustFactor(stock_id="2330", trading_date=date(2026, 7, 7),
                                          adj_factor=1.0))
            session.flush()
            issues, metrics = data_quality._check_adj_factor_freshness(session, max_lag_days=7)
        assert issues == []
        assert metrics["adj_factor_lag_vs_prices_days"] == 2

    def test_empty_table_no_issue(self):
        from app.models import RawPrice
        from skills import data_quality
        with self._session() as session:
            session.add(RawPrice(stock_id="2330", trading_date=date(2026, 7, 9),
                                 close=100, volume=1000))
            session.flush()
            issues, metrics = data_quality._check_adj_factor_freshness(session, max_lag_days=7)
        assert issues == []
        assert metrics["adj_factor_max_date"] is None
