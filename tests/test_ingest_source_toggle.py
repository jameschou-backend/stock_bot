"""驗證 4 個 ingest 的 INGEST_*_SOURCE env var 開關行為。

設計目標：
- 預設（未設）：finmind（向後相容）
- 'twse'：切到 TWSE/TPEx 後端
- 'finmind'：顯式選 finmind
- 未知值（typo）：fall-back 'finmind' 並 log warning（不誤切後端）

對應四個 ingest:
- skills/ingest_prices.py        INGEST_PRICES_SOURCE
- skills/ingest_institutional.py INGEST_INSTITUTIONAL_SOURCE
- skills/ingest_margin_short.py  INGEST_MARGIN_SHORT_SOURCE
- skills/ingest_per.py           INGEST_PER_SOURCE
"""
from __future__ import annotations

import pytest

from skills import ingest_prices, ingest_institutional, ingest_margin_short, ingest_per


_INGEST_MODULES = [
    (ingest_prices, "INGEST_PRICES_SOURCE"),
    (ingest_institutional, "INGEST_INSTITUTIONAL_SOURCE"),
    (ingest_margin_short, "INGEST_MARGIN_SHORT_SOURCE"),
    (ingest_per, "INGEST_PER_SOURCE"),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """確保每個測試從乾淨 env 開始，避免測試互相干擾。"""
    for _, env_key in _INGEST_MODULES:
        monkeypatch.delenv(env_key, raising=False)


@pytest.mark.parametrize("module,env_key", _INGEST_MODULES)
class TestIngestSourceToggle:
    def test_default_is_finmind(self, module, env_key, monkeypatch):
        monkeypatch.delenv(env_key, raising=False)
        assert module._resolve_source() == "finmind", (
            f"{module.__name__} default must be 'finmind' for backward compatibility"
        )

    def test_explicit_finmind(self, module, env_key, monkeypatch):
        monkeypatch.setenv(env_key, "finmind")
        assert module._resolve_source() == "finmind"

    def test_explicit_twse(self, module, env_key, monkeypatch):
        monkeypatch.setenv(env_key, "twse")
        assert module._resolve_source() == "twse"

    def test_case_insensitive(self, module, env_key, monkeypatch):
        monkeypatch.setenv(env_key, "TWSE")
        assert module._resolve_source() == "twse"
        monkeypatch.setenv(env_key, "FinMind")
        assert module._resolve_source() == "finmind"

    def test_whitespace_stripped(self, module, env_key, monkeypatch):
        monkeypatch.setenv(env_key, "  twse  ")
        assert module._resolve_source() == "twse"

    def test_unknown_value_falls_back_to_finmind(self, module, env_key, monkeypatch):
        """typo 防呆：未知值應走預設 finmind（不靜默切到 twse 導致誤切後端）。"""
        monkeypatch.setenv(env_key, "twsee")  # 多打一個 e
        assert module._resolve_source() == "finmind"
        monkeypatch.setenv(env_key, "tw")
        assert module._resolve_source() == "finmind"
        monkeypatch.setenv(env_key, "")
        assert module._resolve_source() == "finmind"


class TestIndependentSwitches:
    """4 個開關必須互相獨立——可以只切其中一個，其他保留 finmind。"""

    def test_can_switch_only_prices(self, monkeypatch):
        monkeypatch.setenv("INGEST_PRICES_SOURCE", "twse")
        assert ingest_prices._resolve_source() == "twse"
        assert ingest_institutional._resolve_source() == "finmind"
        assert ingest_margin_short._resolve_source() == "finmind"
        assert ingest_per._resolve_source() == "finmind"

    def test_can_switch_only_institutional(self, monkeypatch):
        monkeypatch.setenv("INGEST_INSTITUTIONAL_SOURCE", "twse")
        assert ingest_prices._resolve_source() == "finmind"
        assert ingest_institutional._resolve_source() == "twse"
        assert ingest_margin_short._resolve_source() == "finmind"
        assert ingest_per._resolve_source() == "finmind"

    def test_can_switch_all_four(self, monkeypatch):
        for _, env_key in _INGEST_MODULES:
            monkeypatch.setenv(env_key, "twse")
        for module, _ in _INGEST_MODULES:
            assert module._resolve_source() == "twse"


class TestRunDispatch:
    """驗證 run() 真的會根據 _resolve_source() 選對應 _run_*."""

    @pytest.mark.parametrize("module", [m for m, _ in _INGEST_MODULES])
    def test_run_has_both_backends(self, module):
        assert hasattr(module, "_run_finmind"), f"{module.__name__} 缺 _run_finmind"
        assert hasattr(module, "_run_twse"), f"{module.__name__} 缺 _run_twse"
        assert hasattr(module, "run"), f"{module.__name__} 缺 run"
