"""公開申購抽籤掃描（scripts/ipo_lottery_scan.py）單元測試。

全部使用 fixture payload，不打真網路、不連 DB。
"""
from __future__ import annotations

from datetime import date

import pytest

from scripts.ipo_lottery_scan import (
    Announcement,
    build_scan_items,
    classify_status,
    compute_metrics,
    parse_announcements,
    parse_decimal,
    roc_to_date,
)

# TWSE publicForm 實際 schema（含「取消公開抽籤 」尾端空白，忠於線上格式）
_FIELDS = [
    "序號", "抽籤日期", "證券名稱", "證券代號", "發行市場",
    "申購開始日", "申購結束日", "承銷股數", "實際承銷股數",
    "承銷價(元)", "實際承銷價(元)", "撥券日期(上市、上櫃日期)",
    "主辦券商", "申購股數", "總承銷金額(元)", "總合格件", "中籤率(%)",
    "取消公開抽籤 ",
]


def _row(
    name: str,
    stock_id: str,
    market: str,
    draw: str = "115/07/22",
    start: str = "115/07/16",
    end: str = "115/07/20",
    price: str = "80",
    actual_price: str = "未訂出",
    shares: str = "1,000",
    win_rate: str = "0",
    cancelled: str = "",
) -> list[str]:
    return [
        "1", draw, name, stock_id, market, start, end,
        "1,588,000", "1,588,000", price, actual_price, "115/07/28",
        "測試券商", shares, "未訂出", "0", win_rate, cancelled,
    ]


@pytest.fixture()
def payload() -> dict:
    return {
        "stat": "OK",
        "date": 2026,
        "title": "公開申購公告-抽籤日程表",
        "fields": _FIELDS,
        "data": [
            # 進行中 IPO：實際承銷價未訂出 → effective_price 退回公告承銷價 80
            _row("測試IPO", "7001", "初上市", start="115/07/10", end="115/07/14"),
            # 已抽籤上櫃增資：實際承銷價 17.2、中籤率 6.04%
            _row(
                "測試增資", "1586", "上櫃增資",
                draw="115/07/09", start="115/07/03", end="115/07/07",
                price="17.2", actual_price="17.2", win_rate="6.04",
            ),
            # 公債 → parse 階段排除
            _row("115央債", "A151GA", "中央登錄公債"),
            # 取消案件 → 保留但 cancelled=True（build_scan_items 排除）
            _row("取消生技", "7814", "初上櫃", actual_price="---", cancelled="取消申購"),
            # 可轉債（五碼代號，DB 市價一律 n/a）
            _row("測試可轉債", "15821", "上櫃增資", price="100"),
        ],
    }


# ──────────────────────────────────────────────
# 解析
# ──────────────────────────────────────────────
class TestParsing:
    def test_roc_to_date(self):
        assert roc_to_date("115/07/23") == date(2026, 7, 23)
        assert roc_to_date(" 115/01/05 ") == date(2026, 1, 5)

    @pytest.mark.parametrize("raw", ["", "未訂出", "---", "115/07", "abc/07/01"])
    def test_roc_to_date_invalid(self, raw):
        assert roc_to_date(raw) is None

    def test_parse_decimal(self):
        assert parse_decimal("1,588,000") == 1_588_000.0
        assert parse_decimal("17.2") == 17.2
        assert parse_decimal("0") == 0.0

    @pytest.mark.parametrize("raw", ["", "未訂出", "---", "-", "N/A"])
    def test_parse_decimal_missing(self, raw):
        assert parse_decimal(raw) is None

    def test_parse_announcements(self, payload):
        anns = parse_announcements(payload)
        # 公債被排除，其餘 4 筆保留
        assert [a.stock_id for a in anns] == ["7001", "1586", "7814", "15821"]

        ipo = anns[0]
        assert ipo.market_type == "初上市"
        assert ipo.sub_start == date(2026, 7, 10)
        assert ipo.sub_end == date(2026, 7, 14)
        assert ipo.underwriting_price == 80.0
        assert ipo.actual_price is None  # 未訂出
        assert ipo.effective_price == 80.0  # 退回公告承銷價
        assert ipo.shares_per_unit == 1000.0
        assert ipo.win_rate is None  # 公告 0 → 未抽籤

        drawn = anns[1]
        assert drawn.actual_price == 17.2
        assert drawn.effective_price == 17.2
        assert drawn.win_rate == pytest.approx(0.0604)

        assert anns[2].cancelled is True
        assert anns[3].cancelled is False

    def test_parse_announcements_schema_change_fails_loud(self, payload):
        payload["fields"] = [f for f in payload["fields"] if f != "中籤率(%)"]
        with pytest.raises(ValueError, match="schema 變更"):
            parse_announcements(payload)


# ──────────────────────────────────────────────
# 折價與期望值
# ──────────────────────────────────────────────
class TestMetrics:
    def test_discount_calc(self):
        m = compute_metrics(underwriting_price=80.0, market_price=100.0, shares_per_unit=1000, win_rate=None)
        assert m.discount == pytest.approx(0.25)  # 100/80 - 1
        assert m.discount_amount == pytest.approx(20_000.0)  # (100-80)×1000

    def test_expected_value_with_win_rate(self):
        # EV = 20000 × 1.5% − 20 = 280
        m = compute_metrics(80.0, 100.0, 1000, win_rate=0.015, fee=20.0)
        assert m.expected_value == pytest.approx(280.0)

    def test_expected_value_negative_when_rate_too_low(self):
        # EV = 20000 × 0.05% − 20 = -10 → 手續費吃掉折價
        m = compute_metrics(80.0, 100.0, 1000, win_rate=0.0005, fee=20.0)
        assert m.expected_value == pytest.approx(-10.0)

    def test_breakeven_when_win_rate_missing(self):
        m = compute_metrics(80.0, 100.0, 1000, win_rate=None, fee=20.0)
        assert m.expected_value is None
        assert m.breakeven_win_rate == pytest.approx(20.0 / 20_000.0)  # 0.1%
        assert m.breakeven_discount == pytest.approx(20.0 / 80_000.0)  # 0.025%

    def test_negative_discount(self):
        # 市價低於承銷價：折價為負、無保本中籤率（抽中必虧）
        m = compute_metrics(100.0, 90.0, 1000, win_rate=0.01, fee=20.0)
        assert m.discount == pytest.approx(-0.10)
        assert m.discount_amount == pytest.approx(-10_000.0)
        assert m.expected_value == pytest.approx(-120.0)
        assert m.breakeven_win_rate is None

    def test_market_price_missing(self):
        m = compute_metrics(80.0, None, 1000, win_rate=0.01)
        assert m.discount is None
        assert m.discount_amount is None
        assert m.expected_value is None
        assert m.breakeven_win_rate is None
        # 保本折價門檻只需承銷價與股數，仍可算
        assert m.breakeven_discount == pytest.approx(20.0 / 80_000.0)

    def test_underwriting_price_missing(self):
        m = compute_metrics(None, 100.0, 1000, win_rate=0.01)
        assert m.discount is None
        assert m.breakeven_discount is None

    def test_negative_fee_rejected(self):
        with pytest.raises(ValueError):
            compute_metrics(80.0, 100.0, 1000, win_rate=0.01, fee=-1.0)


# ──────────────────────────────────────────────
# 狀態分類
# ──────────────────────────────────────────────
class TestStatus:
    def _ann(self, start: date, end: date) -> Announcement:
        return Announcement(
            stock_id="7001", name="x", market_type="初上市",
            draw_date=None, sub_start=start, sub_end=end, allotment_date=None,
            underwriter="", underwriting_price=None, actual_price=None,
            shares_per_unit=None, win_rate=None, cancelled=False,
        )

    def test_classify(self):
        ann = self._ann(date(2026, 7, 10), date(2026, 7, 14))
        assert classify_status(ann, date(2026, 7, 9)) == "upcoming"
        assert classify_status(ann, date(2026, 7, 10)) == "ongoing"
        assert classify_status(ann, date(2026, 7, 14)) == "ongoing"
        assert classify_status(ann, date(2026, 7, 15)) == "closed"


# ──────────────────────────────────────────────
# 掃描結果組合（狀態過濾 / min-discount / n-a 標記）
# ──────────────────────────────────────────────
class TestBuildScanItems:
    TODAY = date(2026, 7, 10)

    def test_default_excludes_closed_and_cancelled(self, payload):
        anns = parse_announcements(payload)
        items = build_scan_items(anns, market_prices={}, today=self.TODAY)
        ids = {it["stock_id"] for it in items}
        assert "1586" not in ids  # 已截止
        assert "7814" not in ids  # 取消
        assert {"7001", "15821"} <= ids

    def test_include_closed_shows_drawn_win_rate(self, payload):
        anns = parse_announcements(payload)
        items = build_scan_items(
            anns, market_prices={"1586": (18.9, date(2026, 7, 9))}, today=self.TODAY, include_closed=True
        )
        drawn = next(it for it in items if it["stock_id"] == "1586")
        assert drawn["status"] == "closed"
        assert drawn["win_rate"] == pytest.approx(0.0604)
        # EV = (18.9-17.2)×1000 × 0.0604 − 20 = 82.68
        assert drawn["expected_value_twd"] == pytest.approx(82.68, abs=0.01)

    def test_market_price_from_db_and_na(self, payload):
        anns = parse_announcements(payload)
        items = build_scan_items(
            anns, market_prices={"7001": (100.0, date(2026, 7, 9))}, today=self.TODAY
        )
        by_id = {it["stock_id"]: it for it in items}
        ipo = by_id["7001"]
        assert ipo["market_price"] == 100.0
        assert ipo["market_price_date"] == "2026-07-09"
        assert ipo["discount"] == pytest.approx(0.25)
        assert ipo["expected_value_twd"] is None  # 未抽籤 → EV 缺
        assert ipo["breakeven_win_rate"] == pytest.approx(0.001)
        # DB 沒有的（可轉債五碼）→ n/a
        cb = by_id["15821"]
        assert cb["market_price"] is None
        assert cb["discount"] is None

    def test_min_discount_filter(self, payload):
        anns = parse_announcements(payload)
        prices = {"7001": (100.0, date(2026, 7, 9))}  # 折價 25%
        items = build_scan_items(anns, prices, today=self.TODAY, min_discount=0.1)
        assert [it["stock_id"] for it in items] == ["7001"]
        # 門檻高於 25% → 全部濾掉；無市價者（折價未知）也不得通過
        assert build_scan_items(anns, prices, today=self.TODAY, min_discount=0.3) == []

    def test_sorted_by_discount_desc(self, payload):
        anns = parse_announcements(payload)
        prices = {
            "7001": (88.0, date(2026, 7, 9)),  # +10%
            "15821": None,  # 不會出現：僅四碼可查價；模擬仍無價
        }
        prices.pop("15821")
        items = build_scan_items(anns, prices, today=self.TODAY)
        discounts = [it["discount"] for it in items]
        assert discounts[0] == pytest.approx(0.10)
        assert discounts[-1] is None  # None 排最後
