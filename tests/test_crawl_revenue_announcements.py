"""月營收公告日 forward 爬蟲測試：diff 邏輯與 append-only 語義。

全部用本地 fixture（CSV 文字常數 + tmp_path parquet），不打真網路。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts import crawl_revenue_announcements as crawler

# ── fixture：與 2026-07-10 實測 MOPS t21sc03 CSV 相同的 header 與列格式 ──

CSV_HEADER = (
    "出表日期,資料年月,公司代號,公司名稱,產業別,"
    "營業收入-當月營收,營業收入-上月營收,營業收入-去年當月營收,"
    "營業收入-上月比較增減(%),營業收入-去年同月增減(%),"
    "累計營業收入-當月累計營收,累計營業收入-去年累計營收,"
    "累計營業收入-前期比較增減(%),備註"
)


def _row(stock_id: str, ym: str = "115/6", revenue: str = "265297", name: str = "幸福") -> str:
    return (
        f'"115/07/10","{ym}","{stock_id}","{name}","水泥工業","{revenue}","335095","361327",'
        f'"-20.82","-26.57","1903969","2373296","-19.77","-"'
    )


def _csv(*rows: str) -> str:
    return "\r\n".join([CSV_HEADER, *rows]) + "\r\n"


def _fetched(*triples: tuple[str, tuple[int, int], int], source: str = "mops_sii") -> pd.DataFrame:
    """triples: (stock_id, (year, month), revenue) → 正規化 fetched frame。"""
    df = pd.DataFrame(
        {
            "stock_id": [t[0] for t in triples],
            "revenue_year": [t[1][0] for t in triples],
            "revenue_month": [t[1][1] for t in triples],
            "revenue": [t[2] for t in triples],
            "source": source,
        }
    )
    return crawler._coerce_fetched_dtypes(df)


# ── roc_ym_to_ad / report_months ──


def test_roc_ym_to_ad_variants():
    assert crawler.roc_ym_to_ad("115/6") == (2026, 6)
    assert crawler.roc_ym_to_ad("115/06") == (2026, 6)
    assert crawler.roc_ym_to_ad("11506") == (2026, 6)
    assert crawler.roc_ym_to_ad("9912") == (2010, 12)


@pytest.mark.parametrize("bad", ["", "115-6", "115/13", "115/0", "abc", "115/6/1"])
def test_roc_ym_to_ad_rejects_garbage(bad):
    with pytest.raises(ValueError):
        crawler.roc_ym_to_ad(bad)


def test_report_months_mid_window():
    assert crawler.report_months(date(2026, 7, 10)) == [(2026, 6), (2026, 5)]


def test_report_months_crosses_year_boundary():
    assert crawler.report_months(date(2026, 1, 5)) == [(2025, 12), (2025, 11)]
    assert crawler.report_months(date(2026, 2, 1), n=3) == [(2026, 1), (2025, 12), (2025, 11)]


# ── parse_mops_csv ──


def test_parse_mops_csv_normalizes_rows():
    text = _csv(
        _row("1108", revenue="265297"),
        _row("911616", name="某KY存託憑證"),  # 非四碼 → 依專案規範過濾
        _row("1109", revenue="-"),  # 當月營收無法解析 → drop
    )
    df = crawler.parse_mops_csv(text, source="mops_sii")
    assert list(df.columns) == ["stock_id", "revenue_year", "revenue_month", "revenue", "source"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["stock_id"] == "1108"
    assert (row["revenue_year"], row["revenue_month"]) == (2026, 6)
    assert row["revenue"] == 265297
    assert row["source"] == "mops_sii"
    assert df["revenue"].dtype == "int64"


def test_parse_mops_csv_header_only_is_empty_not_error():
    # 尚無公司申報的月份：MOPS 回只有 header 的 CSV（實測 2026-07-10 對 115_7 的行為）
    df = crawler.parse_mops_csv(CSV_HEADER + "\r\n", source="mops_otc")
    assert df.empty
    assert list(df.columns) == ["stock_id", "revenue_year", "revenue_month", "revenue", "source"]


def test_parse_mops_csv_bad_header_raises():
    with pytest.raises(ValueError, match="header"):
        crawler.parse_mops_csv("<html>error page</html>", source="mops_sii")


# ── diff_new_rows ──


def test_diff_all_new_on_empty_ledger():
    fetched = _fetched(("1108", (2026, 6), 100), ("1240", (2026, 6), 50))
    out = crawler.diff_new_rows(fetched, crawler.empty_ledger(), date(2026, 7, 10))
    assert len(out) == 2
    assert list(out.columns) == crawler.LEDGER_COLUMNS
    assert not out["is_revision"].any()
    assert (out["announcement_date"] == pd.Timestamp("2026-07-10")).all()


def test_diff_same_revenue_is_skipped_idempotent():
    fetched = _fetched(("1108", (2026, 6), 100))
    ledger = crawler.diff_new_rows(fetched, crawler.empty_ledger(), date(2026, 7, 9))
    out = crawler.diff_new_rows(fetched, ledger, date(2026, 7, 10))
    assert out.empty  # 同 key 同數字：同日重跑 / 隔日再看到都不 append


def test_diff_changed_revenue_appends_revision():
    ledger = crawler.diff_new_rows(
        _fetched(("1108", (2026, 6), 100)), crawler.empty_ledger(), date(2026, 7, 9)
    )
    out = crawler.diff_new_rows(_fetched(("1108", (2026, 6), 120)), ledger, date(2026, 7, 10))
    assert len(out) == 1
    assert bool(out.iloc[0]["is_revision"]) is True
    assert out.iloc[0]["revenue"] == 120


def test_diff_compares_against_latest_revision_not_first():
    ledger = crawler.diff_new_rows(
        _fetched(("1108", (2026, 6), 100)), crawler.empty_ledger(), date(2026, 7, 9)
    )
    rev = crawler.diff_new_rows(_fetched(("1108", (2026, 6), 120)), ledger, date(2026, 7, 10))
    ledger = pd.concat([ledger, rev], ignore_index=True)

    # 與「最新一列」相同 → 不 append
    assert crawler.diff_new_rows(_fetched(("1108", (2026, 6), 120)), ledger, date(2026, 7, 11)).empty
    # 數字又變回首版 → 相對最新列仍是變動 → 再記一列 revision
    out = crawler.diff_new_rows(_fetched(("1108", (2026, 6), 100)), ledger, date(2026, 7, 11))
    assert len(out) == 1
    assert bool(out.iloc[0]["is_revision"]) is True


def test_diff_same_key_different_month_is_new():
    ledger = crawler.diff_new_rows(
        _fetched(("1108", (2026, 5), 100)), crawler.empty_ledger(), date(2026, 6, 10)
    )
    out = crawler.diff_new_rows(_fetched(("1108", (2026, 6), 100)), ledger, date(2026, 7, 10))
    assert len(out) == 1
    assert bool(out.iloc[0]["is_revision"]) is False


def test_diff_empty_fetch_yields_nothing():
    out = crawler.diff_new_rows(
        crawler._coerce_fetched_dtypes(pd.DataFrame(columns=["stock_id", "revenue_year", "revenue_month", "revenue", "source"])),
        crawler.empty_ledger(),
        date(2026, 7, 10),
    )
    assert out.empty


# ── append_rows：append-only 語義 ──


def test_append_rows_appends_and_never_rewrites_existing(tmp_path):
    path = tmp_path / "announcements.parquet"
    batch1 = crawler.diff_new_rows(
        _fetched(("1108", (2026, 6), 100), ("1240", (2026, 6), 50)),
        crawler.empty_ledger(),
        date(2026, 7, 9),
    )
    assert crawler.append_rows(path, batch1) == 2
    frozen = crawler.load_ledger(path)  # 統一 ns 解析度再比對（parquet 存 ms）

    batch2 = crawler.diff_new_rows(
        _fetched(("1101", (2026, 6), 999), ("1108", (2026, 6), 120)),
        crawler.load_ledger(path),
        date(2026, 7, 10),
    )
    assert crawler.append_rows(path, batch2) == 4

    after = crawler.load_ledger(path)
    assert len(after) == 4
    # 既有列（前 2 列）逐 byte 不變
    pd.testing.assert_frame_equal(after.iloc[:2].reset_index(drop=True), frozen)
    # 新列在尾端：1101 新公告、1108 修正
    tail = after.iloc[2:].set_index("stock_id")
    assert bool(tail.loc["1101", "is_revision"]) is False
    assert bool(tail.loc["1108", "is_revision"]) is True
    assert tail.loc["1108", "revenue"] == 120
    # 首版 1108=100 原封不動仍在
    assert after.iloc[0]["revenue"] == 100


def test_append_rows_empty_batch_does_not_touch_file(tmp_path):
    path = tmp_path / "announcements.parquet"
    batch1 = crawler.diff_new_rows(_fetched(("1108", (2026, 6), 100)), crawler.empty_ledger(), date(2026, 7, 9))
    crawler.append_rows(path, batch1)
    before_bytes = path.read_bytes()
    assert crawler.append_rows(path, crawler.empty_ledger()) == 1
    assert path.read_bytes() == before_bytes  # 空 batch 完全不寫檔


def test_append_rows_empty_batch_missing_file_creates_nothing(tmp_path):
    path = tmp_path / "announcements.parquet"
    assert crawler.append_rows(path, crawler.empty_ledger()) == 0
    assert not path.exists()


def test_load_ledger_missing_file_returns_empty():
    df = crawler.load_ledger(crawler.ROOT / "nonexistent" / "nope.parquet")
    assert df.empty
    assert list(df.columns) == crawler.LEDGER_COLUMNS


def test_load_ledger_schema_mismatch_raises(tmp_path):
    path = tmp_path / "announcements.parquet"
    pd.DataFrame({"foo": [1]}).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="schema"):
        crawler.load_ledger(path)


# ── main：端對端（stub 網路層）與失敗處理 ──


def _stub_fetch(pages: dict):
    """pages: {(market, year, month): csv_text 或 Exception}"""

    def fake(session, market, ad_year, month, timeout=None):
        result = pages.get((market, ad_year, month), CSV_HEADER + "\r\n")
        if isinstance(result, Exception):
            raise result
        return result

    return fake


def test_main_end_to_end_diff_and_append(tmp_path, monkeypatch):
    path = tmp_path / "announcements.parquet"
    args = ["--output", str(path), "--sleep", "0"]

    # Day 1：sii 兩檔、otc 一檔申報（May 檔已完整，僅 1108 一檔簡化）
    monkeypatch.setattr(
        crawler,
        "fetch_market_csv",
        _stub_fetch(
            {
                ("sii", 2026, 6): _csv(_row("1108", revenue="100"), _row("1101", revenue="300")),
                ("otc", 2026, 6): _csv(_row("1240", revenue="50")),
                ("sii", 2026, 5): _csv(_row("1108", ym="115/5", revenue="90")),
            }
        ),
    )
    assert crawler.main(args + ["--as-of", "2026-07-09"]) == 0
    day1 = crawler.load_ledger(path)  # 統一 ns 解析度再比對（parquet 存 ms）
    assert len(day1) == 4
    assert not day1["is_revision"].any()
    assert set(day1["source"]) == {"mops_sii", "mops_otc"}

    # Day 2：新增一檔 + 1108 六月營收修正；其餘不變
    monkeypatch.setattr(
        crawler,
        "fetch_market_csv",
        _stub_fetch(
            {
                ("sii", 2026, 6): _csv(
                    _row("1108", revenue="120"),  # 修正公告
                    _row("1101", revenue="300"),  # 不變 → 不 append
                    _row("2330", revenue="777"),  # 新申報
                ),
                ("otc", 2026, 6): _csv(_row("1240", revenue="50")),
                ("sii", 2026, 5): _csv(_row("1108", ym="115/5", revenue="90")),
            }
        ),
    )
    assert crawler.main(args + ["--as-of", "2026-07-10"]) == 0
    day2 = crawler.load_ledger(path)
    assert len(day2) == 6
    # 既有 4 列原封不動
    pd.testing.assert_frame_equal(day2.iloc[:4].reset_index(drop=True), day1)
    new = day2.iloc[4:]
    by_id = new.set_index("stock_id")
    assert bool(by_id.loc["2330", "is_revision"]) is False
    assert bool(by_id.loc["1108", "is_revision"]) is True
    assert (new["announcement_date"] == pd.Timestamp("2026-07-10")).all()

    # 同日重跑：idempotent，不新增
    assert crawler.main(args + ["--as-of", "2026-07-10"]) == 0
    assert len(pd.read_parquet(path)) == 6


def test_main_fetch_failure_returns_nonzero_without_raising(tmp_path, monkeypatch, capsys):
    path = tmp_path / "announcements.parquet"

    def boom(session, market, ad_year, month, timeout=None):
        raise RuntimeError("MOPS 掛了")

    monkeypatch.setattr(crawler, "fetch_market_csv", boom)
    rc = crawler.main(["--output", str(path), "--sleep", "0", "--as-of", "2026-07-10"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and "MOPS 掛了" in err
    assert not path.exists()  # 全部失敗 → 不產生空 ledger


def test_main_partial_failure_appends_success_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    path = tmp_path / "announcements.parquet"
    monkeypatch.setattr(
        crawler,
        "fetch_market_csv",
        _stub_fetch(
            {
                ("sii", 2026, 6): RuntimeError("sii 掛了"),
                ("otc", 2026, 6): _csv(_row("1240", revenue="50")),
            }
        ),
    )
    rc = crawler.main(["--output", str(path), "--sleep", "0", "--as-of", "2026-07-10"])
    assert rc == 1  # 有失敗就非 0（供排程告警）
    assert "sii 掛了" in capsys.readouterr().err
    df = pd.read_parquet(path)  # 但成功的市場照常入帳（爬蟲壞=只丟當天資料）
    assert list(df["stock_id"]) == ["1240"]
