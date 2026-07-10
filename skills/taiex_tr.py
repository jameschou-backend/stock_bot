"""TAIEX 發行量加權股價報酬指數（Total Return Index）對照臂。

回測策略含息 P&L（BACKTEST_ADJ_PRICE_PARQUET）對照的「機會成本基準」應為含息大盤，
即 TWSE 官方「發行量加權股價報酬指數」（TAIEX Total Return, 2003-01 起）。
本模組負責：fetch（TWSE 免費 endpoint）→ 快取（parquet, 增量更新）→
`vs_taiex_tr` 摘要計算（策略年化 vs TR 指數年化、超額）。

Endpoint 查證（2026-07-10，實測回應格式）：
- 歷史月批次（本模組採用）：
    GET https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U?date=YYYYMMDD&response=json
  一次回傳 date 所在月份全部交易日：
    {"stat": "OK", "fields": ["日　期", "發行量加權股價報酬指數"],
     "data": [["113/01/02", "38,475.17"], ...]}
  民國年日期、千分位逗號數值。同 legacy 路徑 www.twse.com.tw/indicesReport/MFI94U 等價。
  ⚠️ 特定 date 值會 deterministic 回傳錯誤 stat（server 端 per-date cache 汙染），
  故每月以多個 day-of-month probe（詳見 _NO_DATA_STAT_PREFIXES 註解）。
- OpenAPI 變體 https://openapi.twse.com.tw/v1/indicesReport/MFI94U 不接受 date 參數、
  僅回當月（[{"Date":"1150701","TAIEXTotalReturnIndex":"107781.17"}, ...]），
  無法回補歷史，故不採用。

網路失敗不阻斷回測：compute_vs_taiex_tr() 失敗回傳 None 並 log warning，
呼叫端（scripts/run_backtest.py）把 summary["vs_taiex_tr"] 設為 null。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# 只 import 純 helper（不建 client、不觸發任何網路/DB 副作用）
from app.twse_client import TWSEError, roc_date_to_west, safe_float

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

TAIEX_TR_ENDPOINT = "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U"
DEFAULT_CACHE_PATH = ROOT / "artifacts" / "benchmark" / "taiex_tr.parquet"
# TWSE 官方資料自 2003-01 起；再早的月份 stat != OK
TAIEX_TR_DATA_START = date(2003, 1, 1)
# legacy endpoint 禮貌性間隔（與 twse_client 慣例一致，避免 429）
DEFAULT_REQUEST_DELAY_SECS = 1.5
_HTTP_TIMEOUT_SECS = 15

# stat 非 OK 但屬「該日期查無資料」的已知訊息前綴（視為該 probe 無資料，非致命）。
# ⚠️ 實測（2026-07-10）：MFI94U 對「特定 date 值」會 deterministic 回傳錯誤 stat
# （如 date=20170203 → 「很抱歉」、date=20170210 → 「查詢日期小於92年1月」，
# 但 date=20170224 → OK 全月 18 列）——疑似 server 端 per-date cache 汙染。
# 因此單一 no-data 回應不可信，須換同月其他日期 probe；只有 stat=OK 可信。
_NO_DATA_STAT_PREFIXES = ("很抱歉", "查詢日期大於今日", "查詢日期小於")
# 每月依序 probe 的 day-of-month（任一回「OK 且屬本月」即得全月資料）。
# 汙染是 per-date-key 且散佈（實測 2017-06：key 01/26/28 回 2017-12 的 payload、
# key 15 回「很抱歉」，但 05/08/12/20/22/30 正常）→ 多備援 probe 日
_PROBE_DAYS = (1, 15, 28, 8, 22, 5, 26, 12)
_FETCH_MAX_RETRIES = 3          # 每個 probe 日的網路層重試次數
_FETCH_RETRY_BACKOFF_SECS = 3.0
_PROBE_INTERVAL_SECS = 1.0      # probe 日之間的禮貌間隔


class TaiexTRError(Exception):
    """TAIEX TR 指數 fetch / 解析失敗。"""


# ──────────────────────────────────────────────
# Fetch + parse
# ──────────────────────────────────────────────

def parse_mfi94u_payload(payload: dict) -> pd.DataFrame:
    """解析 MFI94U 月批次 JSON → DataFrame[date, tr_index]（純函數，可單元測試）。

    - stat == "OK"：解析 data 列（民國年日期 + 千分位數值）
    - stat 為已知 no-data 訊息（_NO_DATA_STAT_PREFIXES）：回傳空 DataFrame
      （單一 no-data 回應不可信——見常數註解的 per-date cache 汙染 quirk，
      是否視為「該月真的無資料」由 fetch_taiex_tr_month 的多日 probe 決定）
    - 其他 stat / 結構錯誤：raise TaiexTRError
    """
    if not isinstance(payload, dict):
        raise TaiexTRError(f"unexpected payload type: {type(payload)}")
    stat = str(payload.get("stat", ""))
    if stat != "OK":
        if stat.startswith(_NO_DATA_STAT_PREFIXES):
            return pd.DataFrame(columns=["date", "tr_index"])
        raise TaiexTRError(f"MFI94U stat != OK: {stat!r}")

    rows: List[Tuple[date, float]] = []
    for item in payload.get("data") or []:
        if not item or len(item) < 2:
            continue
        try:
            d = roc_date_to_west(str(item[0]))
        except TWSEError as exc:
            raise TaiexTRError(f"unparseable date in MFI94U row: {item!r}") from exc
        val = safe_float(item[1])
        if val is None or val <= 0:
            continue  # 缺值列跳過（不 silent 塞假值）
        rows.append((d, float(val)))
    df = pd.DataFrame(rows, columns=["date", "tr_index"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_taiex_tr_month(year: int, month: int, *, timeout: float = _HTTP_TIMEOUT_SECS) -> pd.DataFrame:
    """抓取單一月份的 TAIEX TR 指數（DataFrame[date, tr_index]）。

    因 per-date cache 汙染 quirk（見 _NO_DATA_STAT_PREFIXES 註解），對同月多個
    day-of-month 依序 probe（每個 probe 日對網路錯誤重試）；任一 probe 回 stat=OK
    即回傳全月資料。所有 probe 皆無 OK：
      - 當月（今日所在月）→ 回傳空 DataFrame（月初連假可能尚無資料，下次增量補齊）
      - 歷史月 → raise TaiexTRError（不允許 silent 缺月：快取覆蓋判定會誤認該月完整）
    """
    import requests  # 延遲 import：離線測試（注入 fetch_month_fn）不需要 requests

    last_reason = ""
    for probe_i, day in enumerate(_PROBE_DAYS):
        if probe_i > 0:
            time.sleep(_PROBE_INTERVAL_SECS)
        params = {"date": f"{year:04d}{month:02d}{day:02d}", "response": "json"}
        for attempt in range(_FETCH_MAX_RETRIES):
            if attempt > 0:
                time.sleep(_FETCH_RETRY_BACKOFF_SECS * attempt)
            try:
                resp = requests.get(
                    TAIEX_TR_ENDPOINT,
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; stock_bot/1.0)"},
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # 網路 / HTTP / JSON decode → 同 probe 日重試
                last_reason = f"day={day} attempt={attempt + 1}: {exc}"
                logger.warning("[taiex_tr] %s-%02d %s", year, month, last_reason)
                continue
            try:
                df = parse_mfi94u_payload(payload)
            except TaiexTRError as exc:
                # 未知 stat（該 date key 疑似汙染，回應 deterministic）→ 換下一個 probe 日
                last_reason = f"day={day}: {exc}"
                logger.warning("[taiex_tr] %s-%02d %s", year, month, last_reason)
                break
            if not df.empty:
                # 防禦：cache 汙染可能回傳「別的月份」的 payload（2026-07-10 首次
                # 回補實測 9 個月被 silent 跳過）——只接受屬於請求月份的列
                in_month = df[
                    df["date"].map(lambda d: (d.year, d.month) == (year, month))
                ].reset_index(drop=True)
                if not in_month.empty:
                    return in_month
                last_reason = f"day={day}: wrong-month payload（{df['date'].iloc[0]} 起）"
                logger.warning("[taiex_tr] %s-%02d %s", year, month, last_reason)
                break
            # stat 為已知 no-data 訊息：單一回應不可信（cache 汙染），換 probe 日
            last_reason = f"day={day}: no-data stat"
            break

    today = date.today()
    if (year, month) == (today.year, today.month):
        logger.info("[taiex_tr] %s-%02d 當月尚無資料（%s）", year, month, last_reason)
        return pd.DataFrame(columns=["date", "tr_index"])
    raise TaiexTRError(
        f"MFI94U {year}-{month:02d}: all probes {_PROBE_DAYS} failed ({last_reason})"
    )


# ──────────────────────────────────────────────
# 快取（parquet，增量更新）
# ──────────────────────────────────────────────

def load_taiex_tr(cache_path: Path | str = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """讀取快取；不存在時回傳空 DataFrame。"""
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "tr_index"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def _months_between(start: date, end: date) -> List[Tuple[int, int]]:
    """[start, end] 涵蓋的 (year, month) 清單（含頭尾月）。"""
    if start > end:
        return []
    months: List[Tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def update_taiex_tr_cache(
    start: date,
    end: date,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    *,
    fetch_month_fn: Optional[Callable[[int, int], pd.DataFrame]] = None,
    request_delay: float = DEFAULT_REQUEST_DELAY_SECS,
) -> pd.DataFrame:
    """確保快取涵蓋 [start, end]，只增量抓缺少的月份；回傳完整快取 DataFrame。

    月份覆蓋判定（presence-based）：範圍內「快取中一列都沒有」的月份須抓；
    快取 max 所在月（當初可能只抓到月中）永遠重抓。此判定可自動修復
    server 端 payload 汙染造成的歷史缺月（span-based 判定會誤認缺月已覆蓋）。
    重抓月與既有列以 date 去重（keep=last，新資料優先）。

    Args:
        start / end: 需要涵蓋的日期範圍（會 clip 到 TAIEX_TR_DATA_START）
        cache_path: parquet 快取路徑
        fetch_month_fn: 注入抓取函數（測試用；預設 fetch_taiex_tr_month）
        request_delay: 每次網路呼叫間隔秒數（禮貌限速；測試傳 0）
    """
    start = max(start, TAIEX_TR_DATA_START)
    if start > end:
        return load_taiex_tr(cache_path)
    fetch = fetch_month_fn or fetch_taiex_tr_month

    cached = load_taiex_tr(cache_path)
    if cached.empty:
        need_months = _months_between(start, end)
    else:
        have_months = {(d.year, d.month) for d in cached["date"]}
        cached_max: date = cached["date"].max()
        max_month = (cached_max.year, cached_max.month)
        need_months = [
            (y, m)
            for (y, m) in _months_between(start, end)
            # 缺月（含被汙染 payload 跳過的歷史月）→ 抓；max 月可能只到月中 → 重抓
            if (y, m) not in have_months or (y, m) == max_month
        ]

    if not need_months:
        return cached

    fetched_parts: List[pd.DataFrame] = []
    for i, (y, m) in enumerate(need_months):
        if i > 0 and request_delay > 0:
            time.sleep(request_delay)
        part = fetch(y, m)
        if part is not None and not part.empty:
            # 防禦（雙層）：只收屬於請求月份的列（fetch 層已擋，此處擋注入 fetcher）
            n0 = len(part)
            part = part[part["date"].map(lambda d: (d.year, d.month) == (y, m))]
            if len(part) < n0:
                logger.warning(
                    "[taiex_tr] %s-%02d payload 含 %d 列非本月資料，已剔除", y, m, n0 - len(part)
                )
            if not part.empty:
                fetched_parts.append(part.reset_index(drop=True))
    logger.info(
        "[taiex_tr] fetched %d/%d months (%s ~ %s)",
        len(fetched_parts), len(need_months), start, end,
    )

    # 排除空 frame 再 concat（避免 pandas empty-concat FutureWarning / dtype 汙染）
    non_empty = [df for df in (cached, *fetched_parts) if not df.empty]
    if not non_empty:
        raise TaiexTRError(f"TAIEX TR 快取與 fetch 皆無資料（{start} ~ {end}）")
    merged = pd.concat(non_empty, ignore_index=True) if len(non_empty) > 1 else non_empty[0].copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 原子寫入：tmp + os.replace，避免中斷留下半寫 parquet。
    # tmp 檔帶 pid 後綴（同 scripts/crawl_revenue_announcements.py 慣例）：
    # 兩個 run_backtest 並行（主臂/歸因臂對照）時固定 tmp 檔名會互踩——
    # 後開者 truncate 先開者正在寫的檔，可能把半寫 parquet 昇級成正式快取。
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        merged.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return merged


# ──────────────────────────────────────────────
# vs_taiex_tr 摘要
# ──────────────────────────────────────────────

def _annualized(total_return: float, years: float) -> float:
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (1 / max(years, 0.01)) - 1


def compute_vs_taiex_tr(
    equity_curve: List[Dict],
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    *,
    allow_fetch: bool = True,
    fetch_month_fn: Optional[Callable[[int, int], pd.DataFrame]] = None,
    request_delay: float = DEFAULT_REQUEST_DELAY_SECS,
) -> Optional[Dict]:
    """對回測 equity_curve 計算「vs 發行量加權股價報酬指數」摘要。

    任一步失敗（網路、快取不涵蓋、資料異常）→ log warning + 回傳 None（不阻斷回測）。

    Args:
        equity_curve: run_backtest 回傳的 [{"date": iso, "equity": float}, ...]
        allow_fetch: False 時只讀既有快取（離線模式）

    Returns:
        dict（見 keys）或 None：
          taiex_tr_total_return / taiex_tr_annualized_return
          strategy_total_return / strategy_annualized_return
          excess_total_return_vs_tr / excess_annualized_vs_tr
          window（start/end/tr_start_date/tr_end_date/years）、index_name、source
    """
    try:
        if not equity_curve or len(equity_curve) < 2:
            raise TaiexTRError("equity_curve 少於 2 點，無法計算對照")
        start_d = pd.Timestamp(equity_curve[0]["date"]).date()
        end_d = pd.Timestamp(equity_curve[-1]["date"]).date()
        eq_start = float(equity_curve[0]["equity"])
        eq_end = float(equity_curve[-1]["equity"])
        if eq_start <= 0:
            raise TaiexTRError(f"equity 起點非正值: {eq_start}")

        if allow_fetch:
            try:
                tr_df = update_taiex_tr_cache(
                    start_d, end_d, cache_path,
                    fetch_month_fn=fetch_month_fn, request_delay=request_delay,
                )
            except Exception as fetch_exc:
                # fetch 失敗但既有快取可能已涵蓋 → 降級用快取（明確 log，非 silent）
                logger.warning("[taiex_tr] fetch 失敗，降級使用既有快取: %s", fetch_exc)
                tr_df = load_taiex_tr(cache_path)
        else:
            tr_df = load_taiex_tr(cache_path)

        if tr_df.empty:
            raise TaiexTRError("TAIEX TR 快取為空且無法 fetch")

        ser = tr_df.set_index(pd.to_datetime(tr_df["date"]))["tr_index"].sort_index()
        # 起訖點皆取「當日或之前最近一筆」（backward asof，兩端同一口徑避免偏差）
        tr_start_val = ser.asof(pd.Timestamp(start_d))
        tr_end_val = ser.asof(pd.Timestamp(end_d))
        tr_start_date = ser.index.asof(pd.Timestamp(start_d))
        tr_end_date = ser.index.asof(pd.Timestamp(end_d))
        if pd.isna(tr_start_val) or pd.isna(tr_end_val) or tr_start_val <= 0:
            raise TaiexTRError(
                f"TR 指數不涵蓋回測窗（{start_d} ~ {end_d}；快取範圍 "
                f"{tr_df['date'].min()} ~ {tr_df['date'].max()}）"
            )
        # 快取過舊（例如離線且 end 遠超快取範圍）→ 對照失真，明確拒絕
        if (end_d - tr_end_date.date()).days > 45:
            raise TaiexTRError(
                f"TR 快取最新日 {tr_end_date.date()} 落後回測終點 {end_d} 超過 45 天"
            )

        years = max((end_d - start_d).days / 365.25, 0.01)
        tr_total = float(tr_end_val / tr_start_val - 1)
        strat_total = eq_end / eq_start - 1
        tr_annual = _annualized(tr_total, years)
        strat_annual = _annualized(strat_total, years)

        return {
            "index_name": "發行量加權股價報酬指數（TAIEX Total Return）",
            "source": f"TWSE MFI94U ({TAIEX_TR_ENDPOINT})",
            "window": {
                "start": start_d.isoformat(),
                "end": end_d.isoformat(),
                "tr_start_date": tr_start_date.date().isoformat(),
                "tr_end_date": tr_end_date.date().isoformat(),
                "years": round(years, 2),
            },
            "taiex_tr_total_return": round(tr_total, 4),
            "taiex_tr_annualized_return": round(tr_annual, 4),
            "strategy_total_return": round(strat_total, 4),
            "strategy_annualized_return": round(strat_annual, 4),
            "excess_total_return_vs_tr": round(strat_total - tr_total, 4),
            "excess_annualized_vs_tr": round(strat_annual - tr_annual, 4),
        }
    except Exception as exc:
        logger.warning("[taiex_tr] vs_taiex_tr 計算失敗（欄位將為 null，不阻斷回測）: %s", exc)
        return None
