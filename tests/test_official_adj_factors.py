"""官方 adj factor 引擎（skills/official_adj_factors.py）單元測試。

覆蓋：
1. 事件比率 → 累積 factor 數學（task fixture：一檔股票兩次除息 + 一次減資）
2. 四個官方 endpoint parser（真實 payload 節錄 fixture）
3. 對帳分類邏輯（matched / 缺事件 / 比率差 / 減資 / 現金增資 / clip / 停牌對應）
4. data_quality adj factor 新鮮度檢查
"""
from __future__ import annotations

import json
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

    def test_split_ratio_math_yageo_pattern(self):
        """股票分割比率數學（2327 國巨 pattern）：除息 0.97 → 一拆四 0.25。

        分割日之後 factor=1.0；除息後、分割前段 factor=0.25（分割向過去累乘）；
        除息前段 factor=0.97×0.25=0.2425。adj_close = close × factor 使
        分割前 546 元收盤還原為 546×0.25=136.5，與分割後價格軸連續。
        """
        events = [(date(2024, 1, 4), 0.97), (date(2024, 1, 10), 0.25)]
        ev_dates, ratios = zip(*events)
        f = oaf.compute_stock_factor_series(list(ev_dates), list(ratios), self.DATES)
        np.testing.assert_allclose(f[6:], 1.0)            # 1/10 分割日（含）之後
        np.testing.assert_allclose(f[2:6], 0.25)          # 除息後、分割前
        np.testing.assert_allclose(f[:2], 0.97 * 0.25)    # 最早段
        # 一拆二十（極端分割）與除息疊加不觸發任何 clip/例外
        f20 = oaf.compute_stock_factor_series(
            [date(2024, 1, 8), date(2024, 1, 12)], [0.05, 0.9], self.DATES)
        assert f20[0] == pytest.approx(0.045)


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


# Phase 2：面額變更（股票分割/併股）真實 payload 節錄（2026-07-10 實測）
TWSE_PAR_VALUE_PAYLOAD = {
    "stat": "OK",
    "fields": ["恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價",
               "漲停價格", "跌停價格", "開盤競價基準", "詳細資料"],
    "data": [
        # 台境 一拆二（面額 10 → 5）
        ["113/11/11", "8476", "台境", "58.80", "29.40", "32.30", "26.50", "29.40",
         "8476,20241031,20241111"],
        # 國巨 一拆四（面額 10 → 2.5），Phase 1 已知缺口
        ["114/08/25", "2327", "國巨", "546.00", "136.50", "150.00", "123.00", "136.50",
         "2327,20250814,20250825"],
        # 康霈 千元股分割（含千分位逗號價格）
        ["114/07/21", "6919", "康霈*", "1,215.00", "121.50", "133.50", "109.50", "121.50",
         "6919,20250714,20250721"],
    ],
}

TPEX_PAR_VALUE_PAYLOAD = {
    "date": "20240101~20260709",
    "tables": [{
        "totalCount": 2,
        "fields": ["恢復買賣日期", "證券代號", "證券名稱", "最後交易日之收盤價格",
                   "恢復買賣開始參考價", "漲停價格", "跌停價格", "開始交易基準價", "詳細資料"],
        "data": [
            # 智通 一拆二：參考價 84.75 ≠ 開始交易基準價 84.80（tick rounding）
            ["1130909", "8932", "智通*           ", "169.50", "84.75", "93.20", "76.30",
             "84.80",
             "<table><tr><th>證券代號/證券名稱:</th><td>8932&nbsp/&nbsp智通*</td></tr>"
             "<tr><th>停止買賣日期:</th><td>113/08/29</td></tr>"
             "<tr><th>恢復買賣日期:</th><td>113/09/09</td></tr>"
             "<tr><th>變更股票面額換股率:</th><td>2.00000000</td></tr>"
             "<tr><th>變更前股票面額:</th><td>10.00</td></tr>"
             "<tr><th>變更後股票面額:</th><td>5.00</td></tr></table>"],
            # 世紀 一拆二十（面額 10 → 0.5，ratio=0.05 需通過 RATIO_LO sanity）
            ["1140331", "5314", "世紀*           ", "1390.00", "69.50", "76.40", "62.60",
             "69.50",
             "<table><tr><th>變更股票面額換股率:</th><td>20.00000000</td></tr>"
             "<tr><th>變更前股票面額:</th><td>10.00</td></tr>"
             "<tr><th>變更後股票面額:</th><td>0.50</td></tr></table>"],
        ],
    }],
    "stat": "ok",
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

    def test_twse_payload_without_stat_key_raises(self):
        """無 stat 鍵的 dict（error object / schema 變更）不可放行——放行會靜默
        回 0 事件並被 checkpoint 固化，整月事件永久漏抓。"""
        from app.twse_client import TWSEError
        with pytest.raises(TWSEError):
            oaf.parse_twse_ex_rights({"error": "internal error"})
        with pytest.raises(TWSEError):
            oaf.parse_twse_capital_reduction({"data": []})  # 有 data 但無 stat 亦不放行

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

    def test_tpex_error_payload_raises(self):
        """TPEx HTTP 200 的 JSON error object（無 tables 鍵）/ 非 ok stat /
        schema 變更 → 必須 raise TWSEError，不可靜默當 0 事件。"""
        from app.twse_client import TWSEError
        with pytest.raises(TWSEError):
            oaf.parse_tpex_ex_rights({"error": "查詢失敗", "code": 500})  # 無 tables
        with pytest.raises(TWSEError):
            oaf.parse_tpex_capital_reduction({"date": "20240101~20240131"})  # 無 tables
        with pytest.raises(TWSEError):
            oaf.parse_tpex_ex_rights({"stat": "error", "tables": []})  # 非 ok stat
        with pytest.raises(TWSEError):
            oaf.parse_tpex_ex_rights({"tables": {"data": []}})  # tables 非 list
        with pytest.raises(TWSEError):
            oaf.parse_tpex_ex_rights({"tables": [["not", "a", "dict"]]})  # 元素非 dict

    def test_tpex_stat_ok_lowercase_and_empty_tables_is_empty(self):
        """實測 TPEx stat 為小寫 'ok'；stat ok + tables 空 = 合法空月。"""
        assert oaf.parse_tpex_ex_rights({"stat": "ok", "tables": []}) == []
        # 節錄 fixture 無 stat 鍵但有 tables：仍視為有效（tables 為 schema 錨點）
        assert oaf.parse_tpex_capital_reduction({"tables": []}) == []

    def test_twse_par_value_change(self):
        events = oaf.parse_twse_par_value_change(TWSE_PAR_VALUE_PAYLOAD)
        assert len(events) == 3
        yageo = events[1]
        assert (yageo.stock_id, yageo.event_date) == ("2327", date(2025, 8, 25))
        assert yageo.event_type == "面額變更"
        assert yageo.source == oaf.SOURCE_PAR_VALUE_CHANGE
        assert yageo.ratio == pytest.approx(0.25)   # 國巨一拆四：136.50 / 546.00
        assert events[0].ratio == pytest.approx(0.5)  # 台境一拆二
        assert events[2].prev_close == pytest.approx(1215.0)  # 千分位逗號

    def test_twse_par_value_change_empty_and_error_stat(self):
        from app.twse_client import TWSEError
        assert oaf.parse_twse_par_value_change({"stat": "OK", "data": []}) == []
        assert oaf.parse_twse_par_value_change({"stat": "查無資料"}) == []
        with pytest.raises(TWSEError):
            oaf.parse_twse_par_value_change({"stat": "查詢結束日期小於查詢開始日期"})
        with pytest.raises(TWSEError):
            oaf.parse_twse_par_value_change({"error": "no stat"})

    def test_tpex_par_value_change(self):
        events = oaf.parse_tpex_par_value_change(TPEX_PAR_VALUE_PAYLOAD)
        assert len(events) == 2
        ev = events[0]
        assert (ev.stock_id, ev.event_date, ev.event_type) == ("8932", date(2024, 9, 9), "面額變更")
        assert ev.ratio == pytest.approx(84.75 / 169.50)           # 主口徑：參考價
        assert ev.ratio_opening == pytest.approx(84.80 / 169.50)   # 基準價僅 tick rounding
        assert ev.payload["split_rate"] == pytest.approx(2.0)      # 詳細資料 HTML 抽出
        assert ev.payload["par_before"] == pytest.approx(10.0)
        assert ev.payload["par_after"] == pytest.approx(5.0)
        # 一拆二十 ratio=0.05 需通過 RATIO_LO sanity（0.02），不可被 events_to_dataframe 濾掉
        deep = events[1]
        assert deep.ratio == pytest.approx(0.05)
        df = oaf.events_to_dataframe(events)
        assert set(df["stock_id"]) == {"8932", "5314"}

    def test_tpex_par_value_change_error_payload_raises(self):
        from app.twse_client import TWSEError
        assert oaf.parse_tpex_par_value_change({"stat": "ok", "tables": []}) == []
        with pytest.raises(TWSEError):
            oaf.parse_tpex_par_value_change({"error": "查詢失敗"})

    def test_events_to_dataframe_filters_non_4digit_and_bad_ratio(self):
        events = oaf.parse_twse_ex_rights(TWSE_EX_RIGHTS_PAYLOAD)
        events.append(oaf.AdjEvent("9999", date(2024, 1, 4), "TWSE", oaf.SOURCE_EX_RIGHTS,
                                   "息", prev_close=None, ref_price=50.0))  # 無前收盤 → skip
        df = oaf.events_to_dataframe(events)
        assert "006203" not in set(df["stock_id"])   # 6 碼 ETF 濾掉
        assert "9999" not in set(df["stock_id"])     # 無效比率濾掉
        assert set(df["stock_id"]) == {"2454", "6442"}

    def test_events_to_dataframe_dedupes_server_duplicate_rows(self):
        """TWSE 偶發同 payload 回完全相同列 ×2（實測：TWTAUU 2016 5906 減資）
        → 同鍵同價重複列必為 server 重複，只保留一筆；
        同日不同 source/價（真實多事件）不可誤刪。"""
        dup = oaf.AdjEvent("5906", date(2016, 7, 14), "TWSE", oaf.SOURCE_CAPITAL_REDUCTION,
                           "減資", prev_close=1.77, ref_price=5.68)
        distinct = oaf.AdjEvent("5906", date(2016, 7, 14), "TWSE", oaf.SOURCE_EX_RIGHTS,
                                "息", prev_close=5.68, ref_price=5.40)
        df = oaf.events_to_dataframe([dup, dup, dup, distinct])
        assert len(df) == 2
        assert sorted(df["source"]) == [oaf.SOURCE_CAPITAL_REDUCTION, oaf.SOURCE_EX_RIGHTS]

    def test_validate_events_in_range(self):
        """TWSE CDN 毒快取第二形態：stat=OK 但內容屬於別的查詢窗
        （實測：2016-04 chunk 拿到 2019-07 資料）→ 必須 raise 讓上游重抓。"""
        from app.twse_client import TWSEError
        ok = oaf.AdjEvent("1101", date(2024, 1, 10), "TWSE", oaf.SOURCE_EX_RIGHTS,
                          "息", prev_close=100.0, ref_price=95.0)
        outside = oaf.AdjEvent("2603", date(2019, 7, 1), "TWSE", oaf.SOURCE_EX_RIGHTS,
                               "息", prev_close=100.0, ref_price=95.0)
        oaf.validate_events_in_range([ok], date(2024, 1, 1), date(2024, 1, 31))  # 不 raise
        oaf.validate_events_in_range([], date(2024, 1, 1), date(2024, 1, 31))    # 空結果合法
        with pytest.raises(TWSEError, match="窗外事件"):
            oaf.validate_events_in_range([ok, outside],
                                         date(2024, 1, 1), date(2024, 1, 31))


# ──────────────────────────────────────────────
# 2.5 checkpoint 續跑（scripts/build_official_adj_factors.py）
# ──────────────────────────────────────────────

class TestCheckpointResume:
    @staticmethod
    def _fake_client(calls: list):
        """回空結果的假 client：記錄每次 HTTP 抓取。"""
        from types import SimpleNamespace

        def _mk(kind: str):
            def _fetch(start, end, cache_bust=False):
                calls.append(kind)
                if kind.startswith("twse"):
                    return {"stat": "OK", "data": []}
                return {"stat": "ok", "tables": []}
            return _fetch

        return SimpleNamespace(
            fetch_twse_ex_rights_raw=_mk("twse_ex_rights"),
            fetch_twse_capital_reduction_raw=_mk("twse_capital_reduction"),
            fetch_twse_par_value_change_raw=_mk("twse_par_value_change"),
            fetch_tpex_ex_rights_raw=_mk("tpex_ex_rights"),
            fetch_tpex_capital_reduction_raw=_mk("tpex_capital_reduction"),
            fetch_tpex_par_value_change_raw=_mk("tpex_par_value_change"),
        )

    def test_corrupted_checkpoint_refetched_not_crash(self, tmp_path):
        """截斷/損壞 checkpoint（JSONDecodeError）→ 自動重抓，不 crash、不需人工刪檔。"""
        from scripts.build_official_adj_factors import fetch_all_events

        calls: list = []
        client = self._fake_client(calls)
        window = (date(2024, 1, 1), date(2024, 1, 31))

        # 第一輪：6 個來源各 1 chunk，全部 HTTP 抓取 + 寫 checkpoint
        fetch_all_events(client, *window, ckpt_dir=tmp_path, resume=True)
        assert len(calls) == 6
        ckpts = sorted(tmp_path.glob("*.json"))
        assert len(ckpts) == 6
        assert not list(tmp_path.glob("*.tmp.*"))  # 原子寫入不留 tmp

        # 損壞其中一個（模擬寫入中斷留下截斷 JSON）
        ckpts[0].write_text('{"stat": "OK", "data": [', encoding="utf-8")

        calls.clear()
        fetch_all_events(client, *window, ckpt_dir=tmp_path, resume=True)
        assert len(calls) == 1  # 只重抓損壞的那個 chunk，其餘走 checkpoint
        json.loads(ckpts[0].read_text(encoding="utf-8"))  # 已被有效內容覆寫

    def test_error_payload_checkpoint_refetched(self, tmp_path):
        """舊版存下的錯誤 payload checkpoint（parse 拋 TWSEError）→ 自動重抓。"""
        from scripts.build_official_adj_factors import fetch_all_events

        calls: list = []
        client = self._fake_client(calls)
        window = (date(2024, 1, 1), date(2024, 1, 31))
        fetch_all_events(client, *window, ckpt_dir=tmp_path, resume=True)

        bad = sorted(tmp_path.glob("tpex_ex_rights_*.json"))[0]
        bad.write_text(json.dumps({"error": "查詢失敗"}), encoding="utf-8")  # 無 tables 鍵

        calls.clear()
        fetch_all_events(client, *window, ckpt_dir=tmp_path, resume=True)
        assert calls == ["tpex_ex_rights"]

    def test_poisoned_window_checkpoint_refetched(self, tmp_path):
        """毒快取第二形態固化的 checkpoint（stat=OK 但資料屬別的查詢窗）
        → resume 時窗驗證失敗、自動重抓，不把窗外事件帶進 events。"""
        from scripts.build_official_adj_factors import fetch_all_events

        calls: list = []
        client = self._fake_client(calls)
        window = (date(2024, 1, 1), date(2024, 1, 31))
        fetch_all_events(client, *window, ckpt_dir=tmp_path, resume=True)

        # 模擬 2016-04 chunk 拿到 2019-07 資料的實測 pattern：
        # 2024-01 chunk checkpoint 裡塞 2024-05 的除權息列
        poisoned = sorted(tmp_path.glob("twse_ex_rights_*.json"))[0]
        poisoned.write_text(json.dumps({
            "stat": "OK",
            "data": [["113年05月10日", "2603", "長榮", "155.00", "85.00", "70.0", "息",
                      "93.50", "76.50", "85.00", "85.00", "x", "", "", ""]],
        }, ensure_ascii=False), encoding="utf-8")

        calls.clear()
        df = fetch_all_events(client, *window, ckpt_dir=tmp_path, resume=True)
        assert calls == ["twse_ex_rights"]   # 只重抓毒 chunk
        assert df.empty                       # 假 client 回空，窗外事件未殘留


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

    def test_matched_via_next_snapshot_gap(self):
        """FinMind 晚一交易日調整（1538/4806 pattern）：官方事件 gap 平坦、
        次一 gap 命中 → matched_via_next_snapshot_gap，且對應 extra jump 被吸收。"""
        events = pd.DataFrame([
            _mk_event("1538", date(2024, 1, 4), 2.0, source=oaf.SOURCE_CAPITAL_REDUCTION),
        ])
        # 官方恢復買賣日 1/4 的 gap 平坦；快照把跳動記在 1/5
        snap = _mk_snapshot("1538", DATES_5, [2.0, 2.0, 2.0, 1.0, 1.0])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(1.0)
        row = res.event_results.iloc[0]
        assert row["reason"] == oaf.REASON_MATCHED_NEXT_GAP
        assert row["mapped_date"] == date(2024, 1, 5)
        assert row["r_snap"] == pytest.approx(2.0)
        assert res.summary["n_matched_via_next_gap"] == 1
        assert res.extra_jumps.empty   # 1/5 的快照跳動已被官方事件吸收

    def test_next_gap_not_used_when_mapped_gap_has_jump(self):
        """mapped gap 本身有跳動但比率不合 → 不走次一 gap 備援（那是不同事件），
        即使次一 gap 的比率恰好命中官方比率。"""
        events = pd.DataFrame([_mk_event("1111", date(2024, 1, 4), 0.80)])
        # gap(1/4)=0.9/1.0=0.9（有跳動、不合 0.8）；gap(1/5)=1.0/1.25=0.8（恰好命中）
        snap = _mk_snapshot("1111", DATES_5, [0.9, 0.9, 1.0, 1.25, 1.25])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        row = res.event_results.iloc[0]
        assert not bool(row["matched"])
        assert row["reason"] == oaf.REASON_RATIO_DIFF_OTHER
        assert res.summary["n_matched_via_next_gap"] == 0

    def test_market_closed_missing_excluded_from_denominator(self):
        """颱風停市日（平日但全市場無交易）快照漏調整 → 已證實快照缺陷，
        分類 missing_in_snapshot_market_closed 且排除於 match rate 分母。"""
        market_days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 8)]
        # 1/5（週五）全市場停市；官方除息事件掛在 1/5，快照全程平坦
        events = pd.DataFrame([
            _mk_event("9101", date(2024, 1, 5), 0.90),
            _mk_event("1111", date(2024, 1, 3), 0.95),   # 正常可對帳事件
        ])
        snap = pd.concat([
            _mk_snapshot("9101", market_days, [1.0] * 4),
            _mk_snapshot("1111", market_days, [0.95, 1.0, 1.0, 1.0]),
        ])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        closed = res.event_results[res.event_results["stock_id"] == "9101"].iloc[0]
        assert closed["reason"] == oaf.REASON_MISSING_MARKET_CLOSED
        assert res.summary["n_missing_market_closed"] == 1
        assert res.summary["n_eligible"] == 1            # 停市缺陷不進分母
        assert res.match_rate == pytest.approx(1.0)

    def test_par_value_change_mismatch_classified(self):
        """面額變更事件缺失/比率差有獨立分類（與除權息/減資區隔）。"""
        ev_missing = pd.DataFrame([
            _mk_event("2327", date(2024, 1, 4), 0.25, source=oaf.SOURCE_PAR_VALUE_CHANGE),
        ])
        snap_flat = _mk_snapshot("2327", DATES_5, [1.0] * 5)
        res = oaf.reconcile_events_vs_snapshot(ev_missing, snap_flat)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_MISSING_PAR_VALUE

        ev_diff = pd.DataFrame([
            _mk_event("2327", date(2024, 1, 4), 0.25, source=oaf.SOURCE_PAR_VALUE_CHANGE),
        ])
        snap_diff = _mk_snapshot("2327", DATES_5, [0.5, 0.5, 1.0, 1.0, 1.0])
        res = oaf.reconcile_events_vs_snapshot(ev_diff, snap_diff)
        assert res.event_results.iloc[0]["reason"] == oaf.REASON_RATIO_DIFF_PAR_VALUE

    def test_par_value_change_matched(self):
        """國巨一拆四 pattern：快照 r_snap=0.25 與官方 ratio=0.25 match。"""
        events = pd.DataFrame([
            _mk_event("2327", date(2024, 1, 4), 0.25, source=oaf.SOURCE_PAR_VALUE_CHANGE),
        ])
        snap = _mk_snapshot("2327", DATES_5, [0.25, 0.25, 1.0, 1.0, 1.0])
        res = oaf.reconcile_events_vs_snapshot(events, snap)
        assert res.match_rate == pytest.approx(1.0)

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

    def test_check_failure_logs_and_records_issue_not_silent(self, caplog):
        """守望機制本身壞掉（query 拋例外）→ logger warning 留痕 + QualityIssue，
        不再被 except 沉默吞掉。"""
        import logging
        from skills import data_quality

        class BrokenSession:
            def query(self, *args, **kwargs):
                raise RuntimeError("schema drifted")

        with caplog.at_level(logging.WARNING, logger="skills.data_quality"):
            issues, metrics = data_quality._check_adj_factor_freshness(
                BrokenSession(), max_lag_days=7
            )
        assert len(issues) == 1
        assert issues[0].category == "adj_factor_freshness_check_failed"
        assert issues[0].severity == "warning"
        assert "schema drifted" in metrics["adj_factor_freshness_check_error"]
        assert any("新鮮度檢查本身失敗" in r.message for r in caplog.records)
