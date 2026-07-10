#!/usr/bin/env python
"""月營收「公告日」forward 爬蟲（PEAD 事件臂資料前置）。

背景（2026-07-10 總體檢提案 2）：歷史 per-company 營收公告日免費拿不到
（MOPS/data.gov.tw 只有依年月的數字，無公告時間戳，屬 TEJ/FinLab 付費庫等級），
時間換不回——只能從現在開始每日累積。本爬蟲每日跑一次，抓 MOPS 營收彙總 CSV，
與本地 ledger diff：「今天新出現的 (stock_id, 營收年月)」記 announcement_date=今天。

資料源（2026-07-10 實測驗證）：
- MOPS 彙總報表 t21sc03（上市/上櫃月營收統計表，含國內+KY），每日更新
  （出表日期當日、隨公司申報增量出現——申報窗 1~10 日內每天只列「已申報」公司）。
- 抓 CSV 版（非 HTML）：POST https://mopsov.twse.com.tw/server-java/FileDownLoad
  form: step=9, functionName=show_file2, filePath=/t21/{sii|otc}/,
  fileName=t21sc03_{民國年}_{月}.csv → UTF-8(BOM) CSV，欄位含
  出表日期/資料年月/公司代號/公司名稱/營業收入-當月營收…
- 尚無任何公司申報的月份回「只有 header 的空 CSV」（HTTP 200）→ 0 列，非錯誤。
- 注意：新版 mops.twse.com.tw 對 /nas/t21 路徑回 404，必須走 mopsov（舊站）。
- 不採 TWSE/TPEx OpenAPI t187ap05_L/mopsfin_t187ap05_O：實測該 dataset 只在
  截止日後整批更新（7/10 當天仍是 5 月資料、出表日 6/17），拿不到 per-company 公告日。

Ledger 語義（artifacts/revenue_announcements/announcements.parquet）：
- 欄位：stock_id, revenue_year, revenue_month, announcement_date, revenue, source, is_revision
  （revenue = 當月營收，單位千元；revenue_year/month 為西元；source = mops_sii / mops_otc）
- **append-only、絕不改寫既有列**（凍結首版防「修正公告」污染 point-in-time 序列）。
  同一 (stock_id, revenue_year, revenue_month) 再出現且數字 != ledger 最新一列
  → 另 append 一列 is_revision=True，原列不動。
- ⚠️ 首次建檔的整批列為「左截尾」：實際公告日可能早於 announcement_date
  （申報窗內已申報的公司全部記成建檔日）。下游使用時應排除
  announcement_date == min(announcement_date) 的首日批次。
- 每日 18:00 排程執行時，18:00 後才申報的公司會記成次日——announcement_date
  語義為「本系統首次可觀測日」，對 point-in-time 回測是保守（無 lookahead）方向。

失敗處理：任何錯誤 → stderr 訊息 + exit code 非 0，不拋未捕捉例外。
單一 market/月份抓取失敗不影響其他抓取（成功部分照常 append），但整體 exit 1。
爬蟲壞 = 只丟當天資料（隔日補上的公司 announcement_date 晚一天），不需告警。

每日 4 個 request（2 市場 × 2 個月），預設間隔 3 秒（MOPS 有 rate limit，禮貌抓取）。

用法：
    python scripts/crawl_revenue_announcements.py                # 每日排程用
    python scripts/crawl_revenue_announcements.py --as-of 2026-07-10 --sleep 0  # 測試
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

MOPS_FILE_DOWNLOAD_URL = "https://mopsov.twse.com.tw/server-java/FileDownLoad"
# 與 app/twse_client.py 同款 UA（本檔刻意不 import app 套件，爬蟲獨立於主系統存活）
USER_AGENT = "Mozilla/5.0 (compatible; stock_bot/1.0)"
MARKETS = ("sii", "otc")  # 上市 / 上櫃；興櫃（rotc）依專案規範排除 EMERGING

DEFAULT_OUTPUT = ROOT / "artifacts" / "revenue_announcements" / "announcements.parquet"
DEFAULT_SLEEP_SECONDS = 3.0
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_REPORT_MONTHS = 2  # 申報窗當月 + 前一個月（補漏申報/修正公告）
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0

KEY_COLUMNS = ["stock_id", "revenue_year", "revenue_month"]
LEDGER_COLUMNS = [
    "stock_id",
    "revenue_year",
    "revenue_month",
    "announcement_date",
    "revenue",
    "source",
    "is_revision",
]
# CSV header 必要欄位（缺任一 = 格式變更，fail loudly，禁止 silent fallback）
REQUIRED_CSV_FIELDS = ("資料年月", "公司代號", "營業收入-當月營收")

_STOCK_ID_RE = re.compile(r"^\d{4}$")  # 專案規範：stock_id 只允許四碼台股


def empty_ledger() -> pd.DataFrame:
    """回傳 schema/dtype 正確的空 ledger。"""
    return pd.DataFrame(
        {
            "stock_id": pd.Series(dtype="object"),
            "revenue_year": pd.Series(dtype="int32"),
            "revenue_month": pd.Series(dtype="int32"),
            "announcement_date": pd.Series(dtype="datetime64[ns]"),
            "revenue": pd.Series(dtype="int64"),
            "source": pd.Series(dtype="object"),
            "is_revision": pd.Series(dtype="bool"),
        }
    )


def roc_ym_to_ad(roc_ym: str) -> tuple[int, int]:
    """民國年月字串 → (西元年, 月)。

    接受兩種實測格式："115/6"、"115/06"（t21sc03 CSV）與 "11506"（OpenAPI 冗餘支援）。
    其他格式一律 ValueError（格式變更要炸出來，不可猜）。
    """
    raw = str(roc_ym).strip()
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"無法解析民國年月: {roc_ym!r}")
        roc_year, month = int(parts[0]), int(parts[1])
    elif raw.isdigit() and len(raw) in (4, 5, 6):  # 如 9906（民國 99 年 6 月）、11506；月固定佔末 2 碼
        roc_year, month = int(raw[:-2]), int(raw[-2:])
    else:
        raise ValueError(f"無法解析民國年月: {roc_ym!r}")
    if not (1 <= month <= 12) or roc_year < 1:
        raise ValueError(f"民國年月超出合理範圍: {roc_ym!r}")
    return roc_year + 1911, month


def report_months(as_of: date, n: int = DEFAULT_REPORT_MONTHS) -> list[tuple[int, int]]:
    """回傳 as_of 往前 n 個「已結束的月份」(西元年, 月)，新→舊。

    月營收申報截止為次月 10 日，故 as_of 當月只可能有「上個月」的申報；
    再往前一個月用來接住漏申報與修正公告。
    """
    if n < 1:
        raise ValueError(f"n 必須 >= 1: {n}")
    year, month = as_of.year, as_of.month
    out: list[tuple[int, int]] = []
    for _ in range(n):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        out.append((year, month))
    return out


def parse_mops_csv(text: str, source: str) -> pd.DataFrame:
    """解析 t21sc03 CSV → 正規化 DataFrame（stock_id/revenue_year/revenue_month/revenue/source）。

    - header 缺必要欄位 → ValueError（格式變更 fail loudly）
    - 只有 header 的空檔（該月尚無公司申報）→ 0 列，非錯誤
    - stock_id 過濾四碼（排除 KY 以外的 5+ 碼證券，如有）
    - 當月營收無法解析成數字的列 drop
    """
    header_line = text.splitlines()[0] if text.strip() else ""
    missing = [f for f in REQUIRED_CSV_FIELDS if f not in header_line]
    if missing:
        raise ValueError(f"MOPS CSV header 缺少必要欄位 {missing}（格式可能已變更）: {header_line[:200]!r}")

    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.empty:
        return _coerce_fetched_dtypes(pd.DataFrame(columns=KEY_COLUMNS + ["revenue", "source"]))

    out = pd.DataFrame(
        {
            "stock_id": df["公司代號"].astype(str).str.strip(),
            "_ym": df["資料年月"].astype(str).str.strip(),
            "revenue": pd.to_numeric(
                df["營業收入-當月營收"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ),
        }
    )
    out = out[out["stock_id"].str.match(_STOCK_ID_RE)]
    out = out.dropna(subset=["revenue"])
    if out.empty:
        return _coerce_fetched_dtypes(pd.DataFrame(columns=KEY_COLUMNS + ["revenue", "source"]))

    ym = out["_ym"].map(roc_ym_to_ad)
    out["revenue_year"] = [y for y, _ in ym]
    out["revenue_month"] = [m for _, m in ym]
    out["source"] = source
    out = out.drop(columns=["_ym"])
    return _coerce_fetched_dtypes(out[KEY_COLUMNS + ["revenue", "source"]])


def _coerce_fetched_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {
                "stock_id": pd.Series(dtype="object"),
                "revenue_year": pd.Series(dtype="int32"),
                "revenue_month": pd.Series(dtype="int32"),
                "revenue": pd.Series(dtype="int64"),
                "source": pd.Series(dtype="object"),
            }
        )
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["revenue_year"] = df["revenue_year"].astype("int32")
    df["revenue_month"] = df["revenue_month"].astype("int32")
    df["revenue"] = df["revenue"].astype("int64")
    df["source"] = df["source"].astype(str)
    return df.reset_index(drop=True)


def load_ledger(path: Path) -> pd.DataFrame:
    """讀取既有 ledger；不存在 → 空 ledger。schema 不符 → RuntimeError（不可 silent 修補）。"""
    if not path.exists():
        return empty_ledger()
    df = pd.read_parquet(path)
    if list(df.columns) != LEDGER_COLUMNS:
        raise RuntimeError(
            f"ledger schema 不符（預期 {LEDGER_COLUMNS}，實際 {list(df.columns)}）：{path}。"
            "拒絕讀寫，請人工檢查（append-only 檔案不可自動遷移）。"
        )
    # parquet round-trip 會把 datetime 存成 ms 解析度；統一轉回 ns，
    # 避免與新列（ns）concat / equals 比較時因解析度不同誤觸 append-only 防呆
    df["announcement_date"] = pd.to_datetime(df["announcement_date"]).astype("datetime64[ns]")
    return df


def diff_new_rows(fetched: pd.DataFrame, ledger: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """與 ledger diff，回傳今日應 append 的列。

    - key 不在 ledger → 新公告（is_revision=False, announcement_date=as_of）
    - key 已在 ledger 且 revenue != 該 key「最新一列」→ 修正公告（is_revision=True）
    - key 已在 ledger 且 revenue 相同 → 略過（同日重跑 idempotent）
    ledger 本身絕不修改。
    """
    fetched = fetched.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    if fetched.empty:
        return empty_ledger()

    if ledger.empty:
        picked = fetched.copy()
        picked["is_revision"] = False
    else:
        # append-only 檔案的列序 = 時間序 → groupby.tail(1) 即每個 key 的最新狀態
        latest = ledger.groupby(KEY_COLUMNS, sort=False).tail(1)
        latest = latest[KEY_COLUMNS + ["revenue"]].rename(columns={"revenue": "_ledger_revenue"})
        merged = fetched.merge(latest, on=KEY_COLUMNS, how="left")
        is_new = merged["_ledger_revenue"].isna()
        is_changed = (~is_new) & (merged["revenue"] != merged["_ledger_revenue"])
        picked = merged[is_new | is_changed].copy()
        picked["is_revision"] = is_changed[is_new | is_changed].to_numpy()
        picked = picked.drop(columns=["_ledger_revenue"])

    # 顯式 ns：pd.Timestamp(date) 廣播會產生 datetime64[s]，跨寫入解析度漂移
    picked["announcement_date"] = pd.Series(
        pd.Timestamp(as_of), index=picked.index, dtype="datetime64[ns]"
    )
    picked["is_revision"] = picked["is_revision"].astype(bool)
    return picked[LEDGER_COLUMNS].reset_index(drop=True)


def append_rows(path: Path, new_rows: pd.DataFrame) -> int:
    """append-only 寫入：既有列原封不動，新列接在後面，temp+os.replace 原子替換。

    回傳寫入後 ledger 總列數。new_rows 為空時不觸碰檔案。
    """
    existing = load_ledger(path)
    if new_rows.empty:
        return len(existing)

    new_rows = new_rows[LEDGER_COLUMNS].reset_index(drop=True)
    combined = new_rows if existing.empty else pd.concat([existing, new_rows], ignore_index=True)

    # append-only 不變量防呆：前 len(existing) 列必須與既有內容完全相同
    if not existing.empty:
        head = combined.iloc[: len(existing)].reset_index(drop=True)
        if not head.equals(existing.reset_index(drop=True)):
            raise RuntimeError("append-only 不變量被破壞：既有列在合併後發生變化，拒絕寫入")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        combined.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return len(combined)


def fetch_market_csv(
    session: requests.Session,
    market: str,
    ad_year: int,
    month: int,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """下載單一市場單一月份的 t21sc03 CSV（含國內+KY），回傳解碼後文字。

    5xx / 429 / 連線錯誤指數退避重試 RETRY_ATTEMPTS 次；其餘 4xx 直接失敗。
    """
    roc_year = ad_year - 1911
    payload = {
        "step": "9",
        "functionName": "show_file2",
        "filePath": f"/t21/{market}/",
        "fileName": f"t21sc03_{roc_year}_{month}.csv",
    }
    last_error: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = session.post(MOPS_FILE_DOWNLOAD_URL, data=payload, timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                last_error = RuntimeError(f"MOPS HTTP {resp.status_code}")
            else:
                resp.raise_for_status()
                return resp.content.decode("utf-8-sig")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    raise RuntimeError(
        f"MOPS 下載失敗 market={market} year={ad_year} month={month}: {last_error}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="月營收公告日 forward 爬蟲（append-only ledger）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="ledger parquet 路徑")
    parser.add_argument(
        "--as-of",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="announcement_date 戳記（預設今天；僅供測試/重演）",
    )
    parser.add_argument(
        "--report-months",
        type=int,
        default=DEFAULT_REPORT_MONTHS,
        help=f"往前抓幾個已結束月份（預設 {DEFAULT_REPORT_MONTHS}：申報窗當月+前月）",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="request 間隔秒數")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout 秒數")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """回傳 exit code；所有例外在此收斂為 stderr 訊息 + 非 0，不外拋。"""
    try:
        args = _parse_args(argv)
    except SystemExit as exc:  # argparse --help / 參數錯誤
        return int(exc.code or 0)

    try:
        as_of = args.as_of or date.today()
        months = report_months(as_of, args.report_months)

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        frames: list[pd.DataFrame] = []
        fetch_errors: list[str] = []
        first_request = True
        for ad_year, month in months:
            for market in MARKETS:
                if not first_request and args.sleep > 0:
                    time.sleep(args.sleep)
                first_request = False
                try:
                    text = fetch_market_csv(session, market, ad_year, month, timeout=args.timeout)
                    parsed = parse_mops_csv(text, source=f"mops_{market}")
                    frames.append(parsed)
                    print(f"[crawl-revenue] {market} {ad_year}-{month:02d}: {len(parsed)} 檔已申報")
                except Exception as exc:  # 單一抓取失敗不擋其他市場/月份
                    fetch_errors.append(f"{market} {ad_year}-{month:02d}: {exc}")

        if frames:
            fetched = pd.concat(frames, ignore_index=True)
        else:
            fetched = _coerce_fetched_dtypes(pd.DataFrame(columns=KEY_COLUMNS + ["revenue", "source"]))

        ledger = load_ledger(args.output)
        first_run = ledger.empty
        new_rows = diff_new_rows(fetched, ledger, as_of)
        total = append_rows(args.output, new_rows)

        n_revision = int(new_rows["is_revision"].sum()) if len(new_rows) else 0
        print(
            f"[crawl-revenue] as_of={as_of} 抓到 {len(fetched)} 列 → 新增 {len(new_rows)} 列"
            f"（其中修正公告 {n_revision}）；ledger 總計 {total} 列 → {args.output}"
        )
        if first_run and len(new_rows):
            print(
                "[crawl-revenue] ⚠️ 首次建檔：本批 announcement_date 為左截尾"
                "（實際公告日可能更早），下游應排除首日批次。"
            )

        if fetch_errors:
            for err in fetch_errors:
                print(f"[crawl-revenue] ERROR: {err}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"[crawl-revenue] FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
