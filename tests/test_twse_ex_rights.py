"""驗證 TWSE 除權除息 parser 與 ingest_corporate_actions toggle。"""
from __future__ import annotations

import os

import pytest

from app.twse_client import TWSEClient, TWSEError
from skills import ingest_corporate_actions


class TestExRightsParser:
    def test_basic_parse(self):
        payload = {
            "stat": "OK",
            "fields": ["資料日期", "股票代號", "股票名稱", "除權息前收盤價",
                       "除權息參考價", "權值+息值", "權/息", "漲停價格"],
            "data": [
                # 台積電除息 5 元範例（虛構但 adj_factor 計算合理）
                ["1150520", "2330", "台積電", "1000.00", "995.00", "5.000000",
                 "息", "1094.00"],
                # 股票股利 10%（除權，ref_price = pre_close / 1.10）
                ["1150520", "1234", "假股", "110.00", "100.00", "10.000000",
                 "權", "121.00"],
            ],
        }
        rows = TWSEClient._parse_twse_legacy_ex_rights(payload)
        assert len(rows) == 2

        # 2330 息：adj_factor = 1000/995 ≈ 1.005
        r0 = rows[0]
        assert r0["stock_id"] == "2330"
        assert r0["action_type"] == "息"
        assert abs(r0["adj_factor"] - 1000 / 995) < 1e-6
        assert r0["pre_close"] == 1000.0
        assert r0["ref_price"] == 995.0

        # 1234 權：adj_factor = 110/100 = 1.10
        r1 = rows[1]
        assert r1["action_type"] == "權"
        assert abs(r1["adj_factor"] - 1.10) < 1e-6

    def test_handles_zero_ref_price(self):
        payload = {
            "stat": "OK",
            "fields": [],
            "data": [
                ["1150520", "2330", "台積電", "1000.00", "0", "0",
                 "息", "1094.00"],
            ],
        }
        rows = TWSEClient._parse_twse_legacy_ex_rights(payload)
        # ref_price=0 不該 div-by-zero，adj_factor 應 None
        assert rows[0]["adj_factor"] is None

    def test_non_ok_stat_returns_empty(self):
        assert TWSEClient._parse_twse_legacy_ex_rights({"stat": "無資料", "data": []}) == []

    def test_rejects_wrong_payload_type(self):
        with pytest.raises(TWSEError):
            TWSEClient._parse_twse_legacy_ex_rights(["not", "a", "dict"])


class TestCorporateActionsSourceToggle:
    @pytest.fixture(autouse=True)
    def _cleanup(self, monkeypatch):
        monkeypatch.delenv("INGEST_CORPORATE_ACTIONS_SOURCE", raising=False)

    def test_default_is_none(self, monkeypatch):
        monkeypatch.delenv("INGEST_CORPORATE_ACTIONS_SOURCE", raising=False)
        assert ingest_corporate_actions._resolve_source() == "none"

    def test_explicit_twse(self, monkeypatch):
        monkeypatch.setenv("INGEST_CORPORATE_ACTIONS_SOURCE", "twse")
        assert ingest_corporate_actions._resolve_source() == "twse"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("INGEST_CORPORATE_ACTIONS_SOURCE", "TWSE")
        assert ingest_corporate_actions._resolve_source() == "twse"

    def test_unknown_falls_back_to_none(self, monkeypatch):
        monkeypatch.setenv("INGEST_CORPORATE_ACTIONS_SOURCE", "mops")  # 還沒實作
        assert ingest_corporate_actions._resolve_source() == "none"
