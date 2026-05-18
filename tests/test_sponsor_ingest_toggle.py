"""驗證 SPONSOR_INGEST env var 開關行為。

設計目標：
- 預設（未設或 'on'）：執行全部 sponsor ingest（向後相容）
- 'off' / 'false' / '0' / 'no'：跳過 6 個 per-stock 重型 sponsor ingest

對應 pipelines/daily_pipeline.py 的 _should_run_sponsor_ingest()。
"""
from __future__ import annotations

import os

import pytest

from pipelines.daily_pipeline import _should_run_sponsor_ingest


class TestSponsorIngestToggle:
    @pytest.fixture(autouse=True)
    def _cleanup(self, monkeypatch):
        """每個測試都從乾淨的 env 開始。"""
        monkeypatch.delenv("SPONSOR_INGEST", raising=False)

    def test_default_is_on_when_env_unset(self, monkeypatch):
        """未設定環境變數時應該預設執行 sponsor ingest（向後相容）。"""
        monkeypatch.delenv("SPONSOR_INGEST", raising=False)
        assert _should_run_sponsor_ingest() is True

    def test_explicit_on(self, monkeypatch):
        monkeypatch.setenv("SPONSOR_INGEST", "on")
        assert _should_run_sponsor_ingest() is True

    @pytest.mark.parametrize("falsy", ["off", "false", "0", "no", "OFF", "False", "No"])
    def test_off_values_skip_sponsor(self, monkeypatch, falsy):
        """所有公認的「關閉」值都應跳過 sponsor ingest。"""
        monkeypatch.setenv("SPONSOR_INGEST", falsy)
        assert _should_run_sponsor_ingest() is False, f"value {falsy!r} should disable sponsor ingest"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("SPONSOR_INGEST", "  off  ")
        assert _should_run_sponsor_ingest() is False

    def test_unknown_value_falls_back_to_on(self, monkeypatch):
        """未知值（typo）應保留預設行為（on），避免誤判導致 silent skip。"""
        monkeypatch.setenv("SPONSOR_INGEST", "maybe")
        assert _should_run_sponsor_ingest() is True
