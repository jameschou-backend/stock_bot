"""TWSE/TPEx client parser 單元測試。

所有測試使用 fixture JSON（不打網路），確保：
1. 民國年轉換正確
2. 千分位逗號、空字串、'-'/'--' 等異常值的處理
3. 上市 vs 上櫃格式差異被正確 normalize
4. TPEx 已知的拼字錯字（ShortConvering）被相容處理
"""
from __future__ import annotations

from datetime import date

import pytest

from app.twse_client import (
    TWSEClient,
    TWSEError,
    normalize_key,
    roc_date_to_west,
    safe_float,
    safe_int,
    strip_commas,
)


# ──────────────────────────────────────────────
# Helper 函式
# ──────────────────────────────────────────────

class TestROCDateConversion:
    @pytest.mark.parametrize(
        "roc,expected",
        [
            ("1150515", date(2026, 5, 15)),
            ("1140101", date(2025, 1, 1)),
            ("1141231", date(2025, 12, 31)),
            ("115/05/15", date(2026, 5, 15)),
            ("114/1/1", date(2025, 1, 1)),
            ("110/2/29", None),  # 民國 110 = 2021，不是閏年，會 raise
        ],
    )
    def test_valid_and_invalid(self, roc, expected):
        if expected is None:
            with pytest.raises(TWSEError):
                roc_date_to_west(roc)
        else:
            assert roc_date_to_west(roc) == expected

    @pytest.mark.parametrize("invalid", ["", "  ", "ABC", "115", "1150515X", "abc/def/ghi"])
    def test_garbage_raises(self, invalid):
        with pytest.raises(TWSEError):
            roc_date_to_west(invalid)


class TestSafeNumber:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("123", 123.0),
            ("123.45", 123.45),
            ("1,234,567", 1234567.0),
            ("  123.45  ", 123.45),
            ("", None),
            ("-", None),
            ("--", None),
            ("X", None),
            ("not_a_number", None),
            (None, None),
        ],
    )
    def test_safe_float(self, raw, expected):
        assert safe_float(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("123", 123),
            ("1,234", 1234),
            ("123.7", 123),  # truncate
            ("", None),
            ("-", None),
        ],
    )
    def test_safe_int(self, raw, expected):
        assert safe_int(raw) == expected

    def test_strip_commas_handles_none(self):
        assert strip_commas(None) == ""
        assert strip_commas("1,234") == "1234"


class TestNormalizeKey:
    """TPEx 3insti JSON key 有空白/大小寫不一致 bug，client 用 normalize_key 做容忍解析。"""

    def test_strips_whitespace_and_case(self):
        assert normalize_key("  Foreign Investors-Buy  ") == "foreigninvestorsbuy"
        assert normalize_key("Dealers -TotalSell") == "dealerstotalsell"

    def test_handles_punctuation(self):
        assert normalize_key("a-b_c.d") == "abcd"


# ──────────────────────────────────────────────
# TWSE OpenAPI parsers
# ──────────────────────────────────────────────

class TestTWSEOAPIStockDay:
    def test_basic_parse(self):
        payload = [
            {
                "Date": "1150515",
                "Code": "2330",
                "Name": "台積電",
                "OpeningPrice": "850.00",
                "HighestPrice": "855.00",
                "LowestPrice": "848.00",
                "ClosingPrice": "852.00",
                "Change": "+2.00",
                "TradeVolume": "12345678",
                "TradeValue": "10500000000",
                "Transaction": "15000",
            }
        ]
        rows = TWSEClient._parse_twse_oapi_stock_day(payload)
        assert len(rows) == 1
        r = rows[0]
        assert r["stock_id"] == "2330"
        assert r["trading_date"] == date(2026, 5, 15)
        assert r["open"] == 850.0
        assert r["close"] == 852.0
        assert r["volume"] == 12345678
        assert r["market"] == "TWSE"

    def test_skips_malformed_row(self):
        payload = [
            {"Code": "2330", "Date": "1150515", "OpeningPrice": "850", "ClosingPrice": "852",
             "HighestPrice": "855", "LowestPrice": "848", "TradeVolume": "100",
             "TradeValue": "10000", "Transaction": "10", "Change": "0"},
            {"Code": "BAD", "Date": "GARBAGE"},  # 日期解析會失敗
        ]
        rows = TWSEClient._parse_twse_oapi_stock_day(payload)
        assert len(rows) == 1
        assert rows[0]["stock_id"] == "2330"

    def test_rejects_non_list(self):
        with pytest.raises(TWSEError):
            TWSEClient._parse_twse_oapi_stock_day({"not": "a list"})


class TestTWSEOAPIMargin:
    """TWSE OpenAPI MI_MARGN 用「中文 key」，與 PER/OHLCV 的英文 key 不同。"""

    def test_chinese_keys(self):
        payload = [
            {
                "股票代號": "2330",
                "股票名稱": "台積電",
                "融資買進": "1,000",
                "融資賣出": "500",
                "融資現金償還": "0",
                "融資前日餘額": "10,000",
                "融資今日餘額": "10,500",
                "融資限額": "100,000",
                "融券買進": "100",
                "融券賣出": "200",
                "融券現券償還": "0",
                "融券前日餘額": "1,000",
                "融券今日餘額": "1,100",
                "融券限額": "10,000",
                "資券互抵": "50",
                "註記": " ",
            }
        ]
        rows = TWSEClient._parse_twse_oapi_margin(payload)
        assert len(rows) == 1
        assert rows[0]["stock_id"] == "2330"
        assert rows[0]["margin_purchase_buy"] == 1000
        assert rows[0]["margin_purchase_today_balance"] == 10500
        assert rows[0]["short_sale_today_balance"] == 1100
        assert rows[0]["note"] == ""


class TestTWSEOAPIPer:
    def test_handles_empty_pe_for_loss_stocks(self):
        payload = [
            {"Date": "1150515", "Code": "1101", "Name": "台泥",
             "PEratio": "", "PBratio": "0.85", "DividendYield": "5.50"},
            {"Date": "1150515", "Code": "2330", "Name": "台積電",
             "PEratio": "25.30", "PBratio": "8.50", "DividendYield": "1.20"},
        ]
        rows = TWSEClient._parse_twse_oapi_per(payload)
        assert len(rows) == 2
        assert rows[0]["per"] is None  # 空字串 = 缺值
        assert rows[0]["pbr"] == 0.85
        assert rows[1]["per"] == 25.3


# ──────────────────────────────────────────────
# TPEx OpenAPI parsers
# ──────────────────────────────────────────────

class TestTPExOAPIQuotes:
    def test_basic_parse(self):
        payload = [
            {
                "Date": "1150515",
                "SecuritiesCompanyCode": "6488",
                "CompanyName": "環球晶",
                "Open": "650",
                "High": "655",
                "Low": "648",
                "Close": "652",
                "Change": " +2",
                "TradingShares": "1000000",
                "TransactionAmount": "650000000",
                "TransactionNumber": "1500",
            }
        ]
        rows = TWSEClient._parse_tpex_oapi_quotes(payload)
        assert len(rows) == 1
        r = rows[0]
        assert r["stock_id"] == "6488"
        assert r["trading_date"] == date(2026, 5, 15)
        assert r["close"] == 652.0
        assert r["market"] == "TPEx"


class TestTPExOAPIMargin:
    """TPEx OpenAPI margin 有官方拼字錯字 ShortConvering，必須兼容。"""

    def test_typo_short_convering(self):
        payload = [
            {
                "Date": "1150515",
                "SecuritiesCompanyCode": "6488",
                "CompanyName": "環球晶",
                "MarginPurchaseBalancePreviousDay": "1000",
                "MarginPurchase": "100",
                "MarginSales": "50",
                "CashRedemption": "0",
                "MarginPurchaseBalance": "1050",
                "MarginPurchaseQuota": "10000",
                "ShortSaleBalancePreviousDay": "200",
                "ShortSale": "20",
                "ShortConvering": "10",  # 官方錯字
                "StockRedemption": "0",
                "ShortSaleBalance": "210",
                "ShortSaleQuota": "2000",
                "Offsetting": "5",
                "Note": "",
            }
        ]
        rows = TWSEClient._parse_tpex_oapi_margin(payload)
        assert len(rows) == 1
        # ShortConvering 應被映射成 short_sale_buy（=融券買進=回補）
        assert rows[0]["short_sale_buy"] == 10
        assert rows[0]["short_sale_today_balance"] == 210

    def test_fallback_to_correct_spelling(self):
        """若官方未來修正拼字為 ShortCovering，應仍可正確解析。"""
        payload = [
            {
                "Date": "1150515",
                "SecuritiesCompanyCode": "6488",
                "CompanyName": "X",
                "MarginPurchase": "0",
                "MarginSales": "0",
                "ShortSale": "0",
                "ShortCovering": "99",  # 正確拼字
            }
        ]
        rows = TWSEClient._parse_tpex_oapi_margin(payload)
        assert rows[0]["short_sale_buy"] == 99


class TestTPExOAPIPer:
    def test_dividend_per_share_extra_field(self):
        """TPEx PER 比 TWSE 多一個 DividendPerShare 欄位。"""
        payload = [
            {
                "Date": "1150515",
                "SecuritiesCompanyCode": "6488",
                "CompanyName": "環球晶",
                "PriceEarningRatio": "15.5",
                "DividendPerShare": "10.0",
                "YieldRatio": "1.5",
                "PriceBookRatio": "3.2",
            }
        ]
        rows = TWSEClient._parse_tpex_oapi_per(payload)
        assert rows[0]["dividend_per_share"] == 10.0
        assert rows[0]["dividend_yield"] == 1.5
        assert rows[0]["per"] == 15.5


# ──────────────────────────────────────────────
# Legacy parsers
# ──────────────────────────────────────────────

class TestTWSELegacyStockDay:
    def test_basic_parse_with_comma_numbers(self):
        payload = {
            "stat": "OK",
            "date": "20260515",
            "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
                       "開盤價", "最高價", "最低價", "收盤價", "漲跌價差"],
            "data": [
                ["2330", "台積電", "12,345,678", "15,000", "10,500,000,000",
                 "850.00", "855.00", "848.00", "852.00", "+2.00"],
            ],
        }
        rows = TWSEClient._parse_twse_legacy_stock_day(payload, date(2026, 5, 15))
        assert len(rows) == 1
        r = rows[0]
        assert r["stock_id"] == "2330"
        assert r["volume"] == 12345678
        assert r["amount"] == 10500000000.0
        assert r["open"] == 850.0
        assert r["close"] == 852.0

    def test_non_ok_stat_returns_empty(self):
        payload = {"stat": "很抱歉，無此資料。", "data": []}
        assert TWSEClient._parse_twse_legacy_stock_day(payload, date(2026, 5, 15)) == []


class TestTWSELegacyT86:
    def test_aggregates_foreign_investor_subtypes(self):
        """T86 有「外陸資（不含外資自營商）」與「外資自營商」兩組，client 應加總成 foreign_*."""
        payload = {
            "data": [
                # 0: code, 1: name
                # 2-4: 外陸資（不含外資自營商）buy/sell/net
                # 5-7: 外資自營商 buy/sell/net
                # 8-10: 投信 buy/sell/net
                # 11: 自營商買賣超
                # 12-14: 自營商（自行買賣）buy/sell/net
                # 15-17: 自營商（避險）buy/sell/net
                # 18: 三大法人買賣超
                ["2330", "台積電",
                 "1,000,000", "500,000", "500,000",   # 外陸資 buy/sell/net
                 "10,000", "5,000", "5,000",          # 外資自營商
                 "200,000", "100,000", "100,000",     # 投信
                 "50,000",                            # 自營商買賣超
                 "30,000", "20,000", "10,000",        # 自營(自行)
                 "60,000", "20,000", "40,000",        # 自營(避險)
                 "655,000"],                          # 三大法人
            ],
        }
        rows = TWSEClient._parse_twse_legacy_t86(payload, date(2026, 5, 15))
        assert len(rows) == 1
        r = rows[0]
        assert r["foreign_buy"] == 1_010_000   # 外陸資 + 外資自營商 buy
        assert r["foreign_sell"] == 505_000
        assert r["foreign_net"] == 505_000     # 外陸資 net + 外資自營商 net
        assert r["trust_net"] == 100_000
        assert r["dealer_self_net"] == 10_000
        assert r["dealer_hedging_net"] == 40_000
        assert r["dealer_net"] == 50_000
        assert r["total_net"] == 655_000

    def test_skips_row_with_too_few_columns(self):
        payload = {"data": [["1101", "台泥"]]}  # 欄位數不足
        assert TWSEClient._parse_twse_legacy_t86(payload, date(2026, 5, 15)) == []


class TestTWSELegacyMarginFilters:
    """TWSE Legacy MI_MARGN 同一個 table 內混合「市場彙總列」與「個股 daily」，
    client 必須只保留 stock_id 開頭為數字的列。"""

    def test_excludes_summary_rows(self):
        payload = {
            "tables": [
                {
                    "data": [
                        ["合計", "", "100", "50", "0", "1000", "1050", "10000",
                         "10", "20", "0", "100", "110", "1000", "5", " "],
                        ["2330", "台積電", "1,000", "500", "0", "10,000", "10,500", "100,000",
                         "100", "200", "0", "1,000", "1,100", "10,000", "50", ""],
                    ]
                }
            ]
        }
        rows = TWSEClient._parse_twse_legacy_margin(payload, date(2026, 5, 15))
        assert len(rows) == 1
        assert rows[0]["stock_id"] == "2330"


class TestTPExLegacyQuotes:
    def test_basic_parse(self):
        payload = {
            "tables": [
                {
                    "data": [
                        # [代號, 名稱, 收盤, 漲跌, 開盤, 最高, 最低, 均價, 成交股數, 成交金額, 成交筆數]
                        ["6488", "環球晶", "652", "+2", "650", "655", "648", "651",
                         "1000000", "650000000", "1500"],
                    ]
                }
            ]
        }
        rows = TWSEClient._parse_tpex_legacy_quotes(payload, date(2026, 5, 15))
        assert len(rows) == 1
        assert rows[0]["stock_id"] == "6488"
        assert rows[0]["close"] == 652.0
        assert rows[0]["market"] == "TPEx"


class TestTWSELegacyPER:
    def test_handles_loss_stock_with_dash(self):
        """TWSE legacy 對虧損股的 PE 用 '-' 表示，與 OpenAPI 的空字串不同。"""
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "收盤價", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [
                ["1101", "台泥", "35.00", "5.50", "112", "-", "0.85", "113Q4"],
                ["2330", "台積電", "852.00", "1.20", "112", "25.30", "8.50", "113Q4"],
            ],
        }
        rows = TWSEClient._parse_twse_legacy_per(payload, date(2026, 5, 15))
        assert len(rows) == 2
        assert rows[0]["per"] is None  # '-' 視為 None
        assert rows[1]["per"] == 25.30
        assert rows[0]["dividend_yield"] == 5.50


# ──────────────────────────────────────────────
# 整合性：parser 必須在 type 錯誤時 raise TWSEError（非 silent return）
# ──────────────────────────────────────────────

class TestParserTypeChecking:
    @pytest.mark.parametrize("parser_name", [
        "_parse_twse_oapi_stock_day",
        "_parse_tpex_oapi_quotes",
        "_parse_twse_oapi_margin",
        "_parse_tpex_oapi_margin",
        "_parse_twse_oapi_per",
        "_parse_tpex_oapi_per",
    ])
    def test_oapi_parsers_require_list(self, parser_name):
        parser = getattr(TWSEClient, parser_name)
        with pytest.raises(TWSEError):
            parser({"not": "a list"})

    @pytest.mark.parametrize("parser_name", [
        "_parse_twse_legacy_stock_day",
        "_parse_tpex_legacy_quotes",
        "_parse_twse_legacy_t86",
        "_parse_tpex_legacy_3insti",
        "_parse_twse_legacy_margin",
        "_parse_tpex_legacy_margin",
        "_parse_twse_legacy_per",
        "_parse_tpex_legacy_per",
    ])
    def test_legacy_parsers_require_dict(self, parser_name):
        parser = getattr(TWSEClient, parser_name)
        with pytest.raises(TWSEError):
            parser(["not", "a", "dict"], date(2026, 5, 15))
