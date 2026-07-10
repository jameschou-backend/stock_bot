"""處置/注意股 live-only 過濾測試（總體檢保留項 1）。

涵蓋：
1. skills/disposition_filter.py：名單解析語義（處置期有效性、四碼過濾、
   placeholder 跳過）、當日快取、fail-open（全部/部分來源失敗）。
2. skills/daily_pick.py::_apply_disposition_filter：剔除語義、index 標籤保留
  （P0-1 同款迴歸防護）、注意股僅記錄、config 開關、排名不受影響。

所有測試以 fixture/monkeypatch 密閉，不打真實網路。
"""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import disposition_filter as dispf
from skills import daily_pick


AS_OF = date(2026, 7, 10)  # 民國 115/07/10

# ── fixture payloads（欄位對齊 2026-07-10 實測 payload）──

TWSE_PUNISH_PAYLOAD = [
    {  # 處置中（迄日 115/07/16 >= as_of）→ 納入
        "Number": "1", "Date": "1150702", "Code": "1101", "Name": "台泥",
        "ReasonsOfDisposition": "連續三次",
        "DispositionPeriod": "115/07/03～115/07/16",
        "DispositionMeasures": "第一次處置", "Detail": "…", "LinkInformation": "…",
    },
    {  # 處置已結束（迄日 115/06/15 < as_of）→ 不納入
        "Number": "2", "Date": "1150601", "Code": "2330", "Name": "台積電",
        "ReasonsOfDisposition": "連續三次",
        "DispositionPeriod": "115/06/02～115/06/15",
        "DispositionMeasures": "第一次處置", "Detail": "…", "LinkInformation": "…",
    },
    {  # 權證（6 碼）→ 不納入集合（非四碼 stock_id）
        "Number": "3", "Date": "1150706", "Code": "059570", "Name": "強茂凱基5B購03",
        "ReasonsOfDisposition": "連續三次",
        "DispositionPeriod": "115/07/07～115/07/20",
        "DispositionMeasures": "第一次處置", "Detail": "…", "LinkInformation": "…",
    },
    {  # 處置期間無法解析 → 保守納入
        "Number": "4", "Date": "1150708", "Code": "5678", "Name": "解析失敗股",
        "ReasonsOfDisposition": "連續三次",
        "DispositionPeriod": "格式不明",
        "DispositionMeasures": "第一次處置", "Detail": "…", "LinkInformation": "…",
    },
]

TWSE_NOTICE_PAYLOAD = [
    {  # 無資料 placeholder（實測：Number="0"、Code=""）→ 跳過
        "Number": "0", "Code": "", "Name": "", "NumberOfAnnouncement": "0",
        "TradingInfoForAttention": "", "Date": "", "ClosingPrice": "0", "PE": "0",
    },
    {
        "Number": "1", "Code": "2330", "Name": "台積電", "NumberOfAnnouncement": "1",
        "TradingInfoForAttention": "第一款", "Date": "1150709",
        "ClosingPrice": "1000", "PE": "25",
    },
]

TPEX_DISPOSAL_PAYLOAD = [
    {  # 處置中 → 納入
        "Date": "1150708", "SecuritiesCompanyCode": "4707", "CompanyName": "磐亞",
        "DispositionPeriod": "1150709~1150722",
        "DispositionReasons": "最近10個營業日內有6個營業日", "DisposalCondition": "…",
    },
    {  # 可轉債（5 碼）→ 不納入集合
        "Date": "1150709", "SecuritiesCompanyCode": "61828", "CompanyName": "合晶八",
        "DispositionPeriod": "1150710~1150723",
        "DispositionReasons": "因連續3個營業日", "DisposalCondition": "…",
    },
]

TPEX_ATTENTION_PAYLOAD = [
    {
        "Date": "1150709", "SecuritiesCompanyCode": "3324", "CompanyName": "雙鴻",
        "TradingInformation": "第十一款", "ClosePrice": "920.00",
        "PriceEarningRatio": "26.55",
    },
]

_URL_PAYLOADS = {
    dispf.TWSE_OAPI_PUNISH: TWSE_PUNISH_PAYLOAD,
    dispf.TWSE_OAPI_NOTICE: TWSE_NOTICE_PAYLOAD,
    dispf.TPEX_OAPI_DISPOSAL: TPEX_DISPOSAL_PAYLOAD,
    dispf.TPEX_OAPI_ATTENTION: TPEX_ATTENTION_PAYLOAD,
}


def _patch_fetch_ok(monkeypatch, counter=None):
    def _fake_fetch(url, timeout):
        if counter is not None:
            counter[url] = counter.get(url, 0) + 1
        return _URL_PAYLOADS[url]

    monkeypatch.setattr(dispf, "_fetch_json", _fake_fetch)


# ──────────────────────────────────────────────
# 1. disposition_filter 模組
# ──────────────────────────────────────────────

class TestParsePeriod:
    def test_twse_slash_fullwidth_tilde(self):
        start, end = dispf._parse_period("115/07/03～115/07/16")
        assert start == date(2026, 7, 3)
        assert end == date(2026, 7, 16)

    def test_tpex_compact_ascii_tilde(self):
        start, end = dispf._parse_period("1150710~1150723")
        assert start == date(2026, 7, 10)
        assert end == date(2026, 7, 23)

    @pytest.mark.parametrize("raw", ["", None, "格式不明", "115/07/03", "a~b"])
    def test_unparseable_returns_none(self, raw):
        assert dispf._parse_period(raw) == (None, None)


class TestFetchDispositionLists:
    def test_basic_semantics(self, monkeypatch, tmp_path):
        """處置期有效性 + 四碼過濾 + placeholder 跳過。"""
        _patch_fetch_ok(monkeypatch)
        result = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)

        # 1101 處置中、5678 期間解析失敗保守納入、4707 TPEx 處置中；
        # 2330 已過期、059570/61828 非四碼 → 排除
        assert result["disposition"] == {"1101", "5678", "4707"}
        # 注意股：TWSE 2330 + TPEx 3324（placeholder 空 Code 跳過）
        assert result["attention"] == {"2330", "3324"}

    def test_cache_written_and_reused(self, monkeypatch, tmp_path):
        """全來源成功 → 寫當日快取；第二次呼叫直接讀快取、不打網路。"""
        counter: dict = {}
        _patch_fetch_ok(monkeypatch, counter)
        first = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)

        cache_path = tmp_path / f"{AS_OF.isoformat()}.json"
        assert cache_path.exists()
        assert sum(counter.values()) == 4  # 4 個 endpoint 各打一次

        # 第二次呼叫：若打網路直接爆炸
        def _boom(url, timeout):
            raise AssertionError("cache hit 時不應打網路")

        monkeypatch.setattr(dispf, "_fetch_json", _boom)
        second = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)
        assert second == first
        assert isinstance(second["disposition"], set)
        assert isinstance(second["attention"], set)

    def test_fail_open_all_endpoints_down(self, monkeypatch, tmp_path, caplog):
        """全來源失敗 → 空集合 + warning 留痕 + 不寫快取（不可讓 daily_pick 掛掉）。"""
        def _down(url, timeout):
            raise ConnectionError("network down")

        monkeypatch.setattr(dispf, "_fetch_json", _down)
        with caplog.at_level("WARNING", logger="skills.disposition_filter"):
            result = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)

        assert result == {"disposition": set(), "attention": set()}
        assert not (tmp_path / f"{AS_OF.isoformat()}.json").exists()
        assert any("fail-open" in rec.message for rec in caplog.records)

    def test_partial_failure_returns_partial_without_cache(
        self, monkeypatch, tmp_path, caplog
    ):
        """單一來源失敗 → 其餘名單照常回傳、不寫快取（同日重跑可重試）。"""
        def _tpex_down(url, timeout):
            if url == dispf.TPEX_OAPI_DISPOSAL:
                raise ConnectionError("tpex down")
            return _URL_PAYLOADS[url]

        monkeypatch.setattr(dispf, "_fetch_json", _tpex_down)
        with caplog.at_level("WARNING", logger="skills.disposition_filter"):
            result = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)

        # TPEx 處置缺席，但 TWSE 處置 + 雙邊注意股仍在
        assert result["disposition"] == {"1101", "5678"}
        assert result["attention"] == {"2330", "3324"}
        assert not (tmp_path / f"{AS_OF.isoformat()}.json").exists()

    def test_corrupt_cache_refetches(self, monkeypatch, tmp_path):
        """快取損壞 → warning + 重新抓取（不可 silent 失敗）。"""
        cache_path = tmp_path / f"{AS_OF.isoformat()}.json"
        cache_path.write_text("not json{", encoding="utf-8")
        _patch_fetch_ok(monkeypatch)
        result = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)
        assert result["disposition"] == {"1101", "5678", "4707"}

    def test_never_raises_contract(self, monkeypatch, tmp_path):
        """公開函式契約：即使內部炸出未預期例外也回空集合（daily_pick 依賴此契約）。"""
        def _bug(timeout):
            raise RuntimeError("unexpected bug")

        monkeypatch.setattr(dispf, "_fetch_all_records", _bug)
        result = dispf.fetch_disposition_lists(as_of=AS_OF, cache_dir=tmp_path)
        assert result == {"disposition": set(), "attention": set()}


# ──────────────────────────────────────────────
# 2. daily_pick._apply_disposition_filter hook
# ──────────────────────────────────────────────

def _build_candidates() -> pd.DataFrame:
    """模擬 _choose_pick_date 之後的進場候選：index 為原始特徵矩陣的列標籤（非 0..k-1）。"""
    return pd.DataFrame(
        {
            "stock_id": ["1101", "2330", "4707", "9999"],
            "trading_date": [AS_OF] * 4,
            "score": [0.9, 0.8, 0.7, 0.6],
        },
        index=[10, 11, 12, 13],  # 刻意非連續起點，重現 P0-1 觸發條件
    )


def _patch_lists(monkeypatch, disposition=None, attention=None):
    def _fake_lists(*args, **kwargs):
        return {
            "disposition": set(disposition or ()),
            "attention": set(attention or ()),
        }

    monkeypatch.setattr(
        daily_pick._disposition, "fetch_disposition_lists", _fake_lists
    )


class TestDailyPickHook:
    def test_excludes_disposition_and_preserves_index(self, monkeypatch):
        """剔除處置股且保留原 index 標籤（絕不可 merge 重置 index）。"""
        df = _build_candidates()
        stats: dict = {}
        _patch_lists(monkeypatch, disposition={"2330"}, attention={"1101"})

        out = daily_pick._apply_disposition_filter(
            df, SimpleNamespace(), stats
        )

        assert list(out["stock_id"]) == ["1101", "4707", "9999"]
        # index 標籤必須是原始標籤（10/12/13），feature_df.loc 對齊依賴此性質
        assert list(out.index) == [10, 12, 13]
        assert stats["disposition_excluded"] == 1
        assert stats["disposition_excluded_ids"] == ["2330"]
        # 注意股不剔除、只記錄
        assert stats["attention_flagged"] == ["1101"]
        assert "1101" in set(out["stock_id"])

    def test_ranking_unaffected(self, monkeypatch):
        """排名不受影響：存活列的 score 與相對順序與過濾前完全一致。"""
        df = _build_candidates()
        _patch_lists(monkeypatch, disposition={"4707"})

        out = daily_pick._apply_disposition_filter(df, SimpleNamespace(), {})

        surviving = df[df["stock_id"] != "4707"]
        pd.testing.assert_frame_equal(out, surviving)
        # topN 排序 = 原排序去掉被剔除者
        assert list(out.sort_values("score", ascending=False)["stock_id"]) == [
            "1101", "2330", "9999",
        ]

    def test_config_switch_disabled(self, monkeypatch):
        """enable_disposition_filter=False → 完全不打名單來源、df 原樣返回。"""
        df = _build_candidates()
        stats: dict = {}

        def _boom(*args, **kwargs):
            raise AssertionError("disabled 時不應抓名單")

        monkeypatch.setattr(
            daily_pick._disposition, "fetch_disposition_lists", _boom
        )
        out = daily_pick._apply_disposition_filter(
            df, SimpleNamespace(enable_disposition_filter=False), stats
        )
        assert out is df
        assert stats["disposition_filter"] == "disabled"

    def test_default_enabled(self, monkeypatch):
        """config 未帶開關 → 預設啟用。"""
        df = _build_candidates()
        stats: dict = {}
        _patch_lists(monkeypatch, disposition={"1101"})

        out = daily_pick._apply_disposition_filter(df, SimpleNamespace(), stats)
        assert "1101" not in set(out["stock_id"])
        assert stats["disposition_excluded"] == 1

    def test_fail_open_empty_sets_no_exclusion(self, monkeypatch):
        """fail-open：名單來源回空集合 → 不剔除任何候選。"""
        df = _build_candidates()
        stats: dict = {}
        _patch_lists(monkeypatch)  # 空集合

        out = daily_pick._apply_disposition_filter(df, SimpleNamespace(), stats)
        pd.testing.assert_frame_equal(out, df)
        assert stats["disposition_excluded"] == 0
        assert stats["attention_flagged"] == []

    def test_empty_candidates(self, monkeypatch):
        """空候選集：不打名單來源、回傳空 df + 歸零統計。"""
        df = pd.DataFrame(columns=["stock_id", "score"])
        stats: dict = {}

        def _boom(*args, **kwargs):
            raise AssertionError("空候選時不應抓名單")

        monkeypatch.setattr(
            daily_pick._disposition, "fetch_disposition_lists", _boom
        )
        out = daily_pick._apply_disposition_filter(df, SimpleNamespace(), stats)
        assert out.empty
        assert stats["disposition_excluded"] == 0

    def test_hook_wired_into_run(self):
        """佈線迴歸防護：run() 內必須實際呼叫 _apply_disposition_filter。"""
        import inspect

        src = inspect.getsource(daily_pick.run)
        assert "_apply_disposition_filter(" in src
