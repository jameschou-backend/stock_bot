"""官方（TWSE/TPEx）除權息 + 減資 + 面額變更公告 → 自算還原因子引擎。

背景：FinMind 還原股價快照凍結於 2026-06-23（sponsor 過期），之後的新除權息事件
未調整，daily 重訓會逐步重演「髒 label 學錯」。本模組用官方免費公告自算事件比率
與累積 factor。Phase 1（除權息+減資）+ Phase 2（面額變更/股票分割、全窗對帳、
matched_via_next_snapshot_gap / market_closed 分類）；切換生產走
scripts/cutover_official_adj_factors.py（--dry-run 預設）。

Endpoint 清單（2026-07-10 實測可用）：

1. TWSE 除權除息計算結果表（TWT49U，歷史區間可查）
   GET https://www.twse.com.tw/rwd/zh/exRight/TWT49U
       ?startDate=YYYYMMDD&endDate=YYYYMMDD&response=json
   ⚠️ 參數是 ``startDate``（app/twse_client.fetch_ex_rights_history 用的 ``strDate``
   已被 server 拒絕，回「查詢結束日期小於查詢開始日期」）。
2. TWSE 減資恢復買賣參考價（TWTAUU；路徑 ``reducation`` 為官方拼字）
   GET https://www.twse.com.tw/rwd/zh/reducation/TWTAUU
       ?startDate=YYYYMMDD&endDate=YYYYMMDD&response=json
3. TPEx 除權除息計算結果表（新版 www endpoint，POST form；
   舊 /web/stock/exright/dailyquo/exDailyQ_result.php 已 302 轉到同一個服務）
   POST https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ
        form: startDate=YYYY/MM/DD&endDate=YYYY/MM/DD&response=json
4. TPEx 減資恢復買賣參考價
   POST https://www.tpex.org.tw/www/zh-tw/bulletin/revivt
        form: startDate=YYYY/MM/DD&endDate=YYYY/MM/DD&response=json
5. TWSE 變更股票面額恢復買賣參考價格（TWTB8U；Phase 2 新增，補股票分割/併股）
   GET https://www.twse.com.tw/rwd/zh/change/TWTB8U
       ?startDate=YYYYMMDD&endDate=YYYYMMDD&response=json
   （官方公式：恢復買賣參考價 = 停止買賣前收盤價 / 變更股票面額換股率；
   例：國巨 2327 2025-08-25 一拆四 546.00 → 136.50，ratio=0.25）
6. TPEx 面額變更恢復買賣參考價（Phase 2 新增）
   POST https://www.tpex.org.tw/www/zh-tw/bulletin/pvChgRslt
        form: startDate=YYYY/MM/DD&endDate=YYYY/MM/DD&response=json
   （換股率/變更前後面額在「詳細資料」HTML 內，parser 以 regex 抽出存 payload）

語義（與 FinMind adj/raw 一致，見 populate_adj_factors.py / build_features.apply_adj_factors）：

- ``adj_close = close × adj_factor``；最近日 factor = 1.0，越早 factor 越小
  （減資彌補虧損方向股價跳升，事件比率 > 1，較早 factor 反而較大）。
- 事件比率 ``r = 恢復買賣/除權息「參考價」÷ 事件前收盤``；
  除息/除權 r < 1，減資（彌補虧損）r > 1。
- 累積 factor：per-stock 由「最近日 = 1.0」向過去累乘事件比率；
  事件日當天屬於「事件後」段（除權息日開盤即為新價格基準）。
- 全交易日展開 = 事件步階函數天然 ffill（與 apply_adj_factors 的缺日 ffill 語義一致）。

⚠️ 現金增資疑義：現金增資認購權會使「除權息參考價」≠「開盤競價基準」
（例：TPEx 1799 2024-01-10 ref=38.72、開始交易基準價=39.25=前收盤）。
FinMind 快照對這類事件是否還原、用哪個價，由 reconcile 對帳實證分類，
不在 Phase 1 預先假設。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from app.twse_client import TWSEError, roc_date_to_west, safe_float

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Endpoints / constants
# ──────────────────────────────────────────────

TWSE_EX_RIGHTS_URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
TWSE_CAPITAL_REDUCTION_URL = "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU"
TWSE_PAR_VALUE_CHANGE_URL = "https://www.twse.com.tw/rwd/zh/change/TWTB8U"
TPEX_EX_RIGHTS_URL = "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ"
TPEX_CAPITAL_REDUCTION_URL = "https://www.tpex.org.tw/www/zh-tw/bulletin/revivt"
TPEX_PAR_VALUE_CHANGE_URL = "https://www.tpex.org.tw/www/zh-tw/bulletin/pvChgRslt"

SOURCE_EX_RIGHTS = "ex_rights"
SOURCE_CAPITAL_REDUCTION = "capital_reduction"
SOURCE_PAR_VALUE_CHANGE = "par_value_change"

#: FinMind 快照凍結日（sponsor 過期，此日之後 DB factor 是 daily pipeline 補 1.0）
SNAPSHOT_FREEZE_DATE = date(2026, 6, 23)

#: 事件比率 sanity 範圍（超出視為髒資料 skip；官方公告資料乾淨，範圍只防解析錯誤）。
#: 下界 0.02：一拆二十（面額 10 → 0.5，例：TPEx 5314 世紀 2025-03-31，ratio=0.05）
#: 需保留 tick rounding 餘裕，不可貼齊 0.05 邊界。
RATIO_LO, RATIO_HI = 0.02, 20.0

DEFAULT_DELAY: float = float(os.environ.get("OFFICIAL_ADJ_REQUEST_DELAY", "2.0"))
DEFAULT_TIMEOUT: float = float(os.environ.get("OFFICIAL_ADJ_TIMEOUT", "30"))
DEFAULT_MAX_RETRIES: int = int(os.environ.get("OFFICIAL_ADJ_MAX_RETRIES", "3"))
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504, 520, 521, 522, 524}
_USER_AGENT = "Mozilla/5.0 (compatible; stock_bot/1.0)"

_STOCK_ID_RE = re.compile(r"^\d{4}$")  # 專案規範：只允許四碼台股

EVENT_COLUMNS = [
    "stock_id", "event_date", "market", "source", "event_type",
    "prev_close", "ref_price", "opening_ref", "ratio", "ratio_opening",
    "cash_increase_suspected", "reason", "payload_json",
]


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class AdjEvent:
    """單一還原事件（除權/除息/減資）。

    ratio         = ref_price / prev_close（主口徑：官方參考價）
    ratio_opening = opening_ref / prev_close（備援口徑：開盤競價基準，
                    現金增資認購權疑義時與 ratio 不同）
    """

    stock_id: str
    event_date: date
    market: str            # 'TWSE' / 'TPEx'
    source: str            # SOURCE_EX_RIGHTS / SOURCE_CAPITAL_REDUCTION
    event_type: str        # '息' / '權' / '權息' / '減資'
    prev_close: Optional[float]
    ref_price: Optional[float]
    opening_ref: Optional[float] = None
    cash_increase_suspected: bool = False
    reason: str = ""       # 減資原因（彌補虧損/退還股款），除權息留空
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> Optional[float]:
        if self.prev_close and self.ref_price and self.prev_close > 0 and self.ref_price > 0:
            return self.ref_price / self.prev_close
        return None

    @property
    def ratio_opening(self) -> Optional[float]:
        if self.prev_close and self.opening_ref and self.prev_close > 0 and self.opening_ref > 0:
            return self.opening_ref / self.prev_close
        return None


def validate_events_in_range(events: Sequence[AdjEvent], start: date, end: date,
                             name: str = "") -> None:
    """驗證事件日期都落在請求窗 [start, end] 內。

    TWSE CDN 毒快取的第二形態（2026-07-10 實測）：回應 stat=OK 但內容是
    「別的查詢窗」的快取資料（例：請求 2016-04 chunk 拿到 2019-07 的除權息表）。
    不驗證會把整月錯誤事件固化進 checkpoint → 事件跨 chunk 重複、
    對帳徹底失真（match rate 97%→24%）。驗證失敗 raise TWSEError，
    由上游帶 cache-bust 重試繞開毒快取。
    """
    bad = [e for e in events if not (start <= e.event_date <= end)]
    if bad:
        raise TWSEError(
            f"{name} 回應含 {len(bad)}/{len(events)} 筆窗外事件"
            f"（例：{bad[0].stock_id} {bad[0].event_date}，請求窗 {start}~{end}），"
            "疑似 CDN 毒快取（需 cache-bust 重試，不可當有效結果）"
        )


def events_to_dataframe(events: Iterable[AdjEvent]) -> pd.DataFrame:
    """AdjEvent list → DataFrame（含比率欄）；只保留四碼 stock_id 與有效比率。

    去重：TWSE 偶發在同一 payload 內回完全相同的列兩次（實測：TWTAUU 2016
    5906 減資列 ×2），以 (stock_id, event_date, market, source, prev_close,
    ref_price) 為 key 保留第一筆——同鍵同價的重複列必為 server 重複列，
    不可能是兩個真實事件。"""
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for ev in events:
        if not _STOCK_ID_RE.match(str(ev.stock_id)):
            continue  # ETF（0050 為 4 碼會保留；006208 等 6 碼濾掉）
        r = ev.ratio
        if r is None or not (RATIO_LO <= r <= RATIO_HI):
            skipped += 1
            logger.warning(
                "[official_adj] skip event with invalid ratio: %s %s ratio=%s",
                ev.stock_id, ev.event_date, r,
            )
            continue
        rows.append({
            "stock_id": str(ev.stock_id),
            "event_date": ev.event_date,
            "market": ev.market,
            "source": ev.source,
            "event_type": ev.event_type,
            "prev_close": ev.prev_close,
            "ref_price": ev.ref_price,
            "opening_ref": ev.opening_ref,
            "ratio": r,
            "ratio_opening": ev.ratio_opening,
            "cash_increase_suspected": bool(ev.cash_increase_suspected),
            "reason": ev.reason,
            "payload_json": json.dumps(ev.payload, ensure_ascii=False),
        })
    if skipped:
        logger.warning("[official_adj] %d events skipped（無效比率）", skipped)
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if not df.empty:
        n_before = len(df)
        df = df.drop_duplicates(
            subset=["stock_id", "event_date", "market", "source",
                    "prev_close", "ref_price"],
            keep="first",
        )
        if len(df) < n_before:
            logger.warning("[official_adj] %d duplicate event rows dropped"
                           "（server 重複列）", n_before - len(df))
        df = df.sort_values(["stock_id", "event_date"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Parsers（純函數，方便 fixture 測試）
# ──────────────────────────────────────────────

def _parse_roc_date(s: str) -> date:
    """民國年日期解析，支援三種官方格式：

    - '113年01月04日'（TWT49U 資料日期）
    - '113/01/22'（TWTAUU / TPEx exDailyQ）
    - '1130205'（TPEx revivt 恢復買賣日期，7 碼 compact）
    """
    cleaned = (s or "").strip().replace("年", "/").replace("月", "/").replace("日", "")
    return roc_date_to_west(cleaned)


#: TWSE 「真的沒資料」的 stat 關鍵字（視為空結果）；其他非 OK stat 一律視為
#: 暫時性 server 錯誤 raise（實測 TWSE 偶發回「查詢開始日期小於92年5月5日」等
#: 無意義錯誤——若靜默當空結果會漏抓整個月的事件，必須重試）
_EMPTY_STAT_MARKERS = ("沒有符合條件", "查無資料")


def _stat_ok(payload: Any, name: str) -> bool:
    """TWSE payload 有效性判定：必須有 stat=OK 或明確「查無資料」才算有效。

    無 stat 鍵的 dict（error object / CDN 異常 / schema 變更）不可放行——
    放行會讓 parser 靜默回 0 事件並被 checkpoint 固化，整月除權息事件永久漏抓。
    """
    if not isinstance(payload, dict):
        raise TWSEError(f"expected dict for {name}, got {type(payload).__name__}")
    stat = payload.get("stat")
    if stat is None:
        raise TWSEError(
            f"{name} payload 無 stat 鍵（keys={sorted(map(str, payload))[:8]}），"
            "無法確認有效性（需重試，不可當空結果）"
        )
    stat_s = str(stat)
    if stat_s != "OK":
        if any(m in stat_s for m in _EMPTY_STAT_MARKERS):
            logger.info("[official_adj] %s stat=%s（視為空結果）", name, stat)
            return False
        raise TWSEError(f"{name} 回傳暫時性錯誤 stat={stat!r}（需重試，不可當空結果）")
    return True


def parse_twse_ex_rights(payload: Any) -> List[AdjEvent]:
    """TWSE TWT49U 除權除息計算結果表。

    fields：0:資料日期(113年01月04日) 1:股票代號 2:股票名稱 3:除權息前收盤價
    4:除權息參考價 5:權值+息值 6:權/息 7:漲停價格 8:跌停價格 9:開盤競價基準
    10:減除股利參考價 11:詳細資料 ...
    """
    if not _stat_ok(payload, "TWSE TWT49U"):
        return []
    out: List[AdjEvent] = []
    for r in payload.get("data") or []:
        try:
            if not r or len(r) < 10:
                continue
            prev_close = safe_float(r[3])
            ref_price = safe_float(r[4])
            opening_ref = safe_float(r[9])
            kind = (r[6] or "").strip() or "OTHER"
            cash_suspected = (
                ref_price is not None and opening_ref is not None
                and abs(opening_ref - ref_price) > max(0.011, 0.002 * ref_price)
            )
            out.append(AdjEvent(
                stock_id=str(r[1]).strip(),
                event_date=_parse_roc_date(r[0]),
                market="TWSE",
                source=SOURCE_EX_RIGHTS,
                event_type=kind,
                prev_close=prev_close,
                ref_price=ref_price,
                opening_ref=opening_ref,
                cash_increase_suspected=cash_suspected,
                payload={"value_amount": safe_float(r[5]), "kind": kind},
            ))
        except (IndexError, KeyError, TWSEError) as exc:
            logger.warning("TWSE TWT49U row skipped: %s row=%s", exc, r)
    return out


def parse_twse_capital_reduction(payload: Any) -> List[AdjEvent]:
    """TWSE TWTAUU 減資恢復買賣參考價格。

    fields：0:恢復買賣日期(113/01/22) 1:股票代號 2:名稱 3:停止買賣前收盤價格
    4:恢復買賣參考價 5:漲停 6:跌停 7:開盤競價基準 8:除權參考價 9:減資原因 10:詳細資料
    """
    if not _stat_ok(payload, "TWSE TWTAUU"):
        return []
    out: List[AdjEvent] = []
    for r in payload.get("data") or []:
        try:
            if not r or len(r) < 10:
                continue
            out.append(AdjEvent(
                stock_id=str(r[1]).strip(),
                event_date=_parse_roc_date(r[0]),
                market="TWSE",
                source=SOURCE_CAPITAL_REDUCTION,
                event_type="減資",
                prev_close=safe_float(r[3]),
                ref_price=safe_float(r[4]),
                opening_ref=safe_float(r[7]),
                reason=(r[9] or "").strip(),
                payload={"exright_ref": safe_float(r[8])},
            ))
        except (IndexError, KeyError, TWSEError) as exc:
            logger.warning("TWSE TWTAUU row skipped: %s row=%s", exc, r)
    return out


def _tpex_tables_rows(payload: Any, name: str) -> List[list]:
    """TPEx payload 取列（含 schema 驗證）。

    TPEx www 正常回應為 ``{"date": ..., "tables": [...], "stat": "ok"}``；
    HTTP 200 的 JSON error object（無 tables 鍵）或 schema 變更若靜默當 0 事件，
    會被 checkpoint 固化 → 整月除權息事件永久漏抓——必須 raise 讓上游重試。
    """
    if not isinstance(payload, dict):
        raise TWSEError(f"expected dict for {name}, got {type(payload).__name__}")
    stat = payload.get("stat")
    if stat is not None and str(stat).strip().lower() != "ok":
        stat_s = str(stat)
        if any(m in stat_s for m in _EMPTY_STAT_MARKERS):
            logger.info("[official_adj] %s stat=%s（視為空結果）", name, stat)
            return []
        raise TWSEError(f"{name} 回傳非 ok stat={stat!r}（需重試，不可當空結果）")
    if "tables" not in payload:
        raise TWSEError(
            f"{name} payload 無 tables 鍵（keys={sorted(map(str, payload))[:8]}），"
            "非預期 schema / error object（需重試，不可當空結果）"
        )
    tables = payload["tables"] or []
    if not isinstance(tables, list):
        raise TWSEError(f"{name} tables 非 list（{type(tables).__name__}），非預期 schema")
    rows: List[list] = []
    for table in tables:
        if not isinstance(table, dict):
            raise TWSEError(f"{name} tables 元素非 dict（{type(table).__name__}），非預期 schema")
        rows.extend(table.get("data") or [])
    return rows


def parse_tpex_ex_rights(payload: Any) -> List[AdjEvent]:
    """TPEx bulletin/exDailyQ 除權除息計算結果表。

    fields（21 欄）：0:除權息日期(113/01/03) 1:代號 2:名稱 3:除權息前收盤價
    4:除權息參考價 5:權值 6:息值 7:權值+息值 8:權/息(除息/除權/除權息)
    9:漲停價 10:跌停價 11:開始交易基準價 12:減除股利參考價 13:現金股利
    14:每仟股無償配股 15:現金增資股數 16:現金增資認購價 17:公開承銷股數
    18:員工認購股數 19:原股東認購股數 20:按持股比例仟股認購
    """
    out: List[AdjEvent] = []
    for r in _tpex_tables_rows(payload, "TPEx exDailyQ"):
        try:
            if not r or len(r) < 12:
                continue
            kind = (r[8] or "").strip().replace("除", "") or "OTHER"  # 除權息→權息
            cash_shares = safe_float(r[15]) if len(r) > 15 else None
            out.append(AdjEvent(
                stock_id=str(r[1]).strip(),
                event_date=_parse_roc_date(r[0]),
                market="TPEx",
                source=SOURCE_EX_RIGHTS,
                event_type=kind,
                prev_close=safe_float(r[3]),
                ref_price=safe_float(r[4]),
                opening_ref=safe_float(r[11]),
                cash_increase_suspected=bool(cash_shares and cash_shares > 0),
                payload={
                    "value_amount": safe_float(r[7]),
                    "cash_dividend": safe_float(r[13]) if len(r) > 13 else None,
                    "stock_dividend_per_1000": safe_float(r[14]) if len(r) > 14 else None,
                    "cash_increase_shares": cash_shares,
                },
            ))
        except (IndexError, KeyError, TWSEError) as exc:
            logger.warning("TPEx exDailyQ row skipped: %s row=%s", exc, r)
    return out


def parse_tpex_capital_reduction(payload: Any) -> List[AdjEvent]:
    """TPEx bulletin/revivt 減資恢復買賣參考價。

    fields（11 欄）：0:恢復買賣日期(1130205 compact) 1:股票代號 2:名稱
    3:最後交易日之收盤價格 4:減資恢復買賣開始日參考價格 5:漲停價格 6:跌停價格
    7:開始交易基準價 8:除權參考價 9:減資原因 10:詳細資料(HTML)
    """
    out: List[AdjEvent] = []
    for r in _tpex_tables_rows(payload, "TPEx revivt"):
        try:
            if not r or len(r) < 10:
                continue
            out.append(AdjEvent(
                stock_id=str(r[1]).strip(),
                event_date=_parse_roc_date(r[0]),
                market="TPEx",
                source=SOURCE_CAPITAL_REDUCTION,
                event_type="減資",
                prev_close=safe_float(r[3]),
                ref_price=safe_float(r[4]),
                opening_ref=safe_float(r[7]),
                reason=(r[9] or "").strip(),
                payload={"exright_ref": safe_float(r[8])},
            ))
        except (IndexError, KeyError, TWSEError) as exc:
            logger.warning("TPEx revivt row skipped: %s row=%s", exc, r)
    return out


def parse_twse_par_value_change(payload: Any) -> List[AdjEvent]:
    """TWSE TWTB8U 變更股票面額恢復買賣參考價格（股票分割/併股）。

    fields（9 欄）：0:恢復買賣日期(113/11/11) 1:股票代號 2:名稱 3:停止買賣前收盤價格
    4:恢復買賣參考價 5:漲停價格 6:跌停價格 7:開盤競價基準 8:詳細資料
    （官方公式：恢復買賣參考價 = 停止買賣前收盤價 / 變更股票面額換股率）
    """
    if not _stat_ok(payload, "TWSE TWTB8U"):
        return []
    out: List[AdjEvent] = []
    for r in payload.get("data") or []:
        try:
            if not r or len(r) < 8:
                continue
            out.append(AdjEvent(
                stock_id=str(r[1]).strip(),
                event_date=_parse_roc_date(r[0]),
                market="TWSE",
                source=SOURCE_PAR_VALUE_CHANGE,
                event_type="面額變更",
                prev_close=safe_float(r[3]),
                ref_price=safe_float(r[4]),
                opening_ref=safe_float(r[7]),
                payload={"detail": str(r[8]) if len(r) > 8 else None},
            ))
        except (IndexError, KeyError, TWSEError) as exc:
            logger.warning("TWSE TWTB8U row skipped: %s row=%s", exc, r)
    return out


#: TPEx pvChgRslt「詳細資料」HTML 內的換股率/面額欄位（<th>標籤:</th><td>值</td>）
_PV_RATE_RE = re.compile(r"變更股票面額換股率:?\s*</th>\s*<td>\s*([\d.,]+)")
_PV_PAR_BEFORE_RE = re.compile(r"變更前股票面額:?\s*</th>\s*<td>\s*([\d.,]+)")
_PV_PAR_AFTER_RE = re.compile(r"變更後股票面額:?\s*</th>\s*<td>\s*([\d.,]+)")


def parse_tpex_par_value_change(payload: Any) -> List[AdjEvent]:
    """TPEx bulletin/pvChgRslt 面額變更恢復買賣參考價（股票分割/併股）。

    fields（9 欄）：0:恢復買賣日期(1130909 compact) 1:證券代號 2:證券名稱
    3:最後交易日之收盤價格 4:恢復買賣開始參考價 5:漲停價格 6:跌停價格
    7:開始交易基準價 8:詳細資料(HTML，含換股率與變更前/後面額)

    注意：開始交易基準價會被 tick rounding（例 8932 參考價 84.75、基準價 84.80），
    主口徑用參考價；換股率抽進 payload 供交叉驗證。
    """
    out: List[AdjEvent] = []
    for r in _tpex_tables_rows(payload, "TPEx pvChgRslt"):
        try:
            if not r or len(r) < 8:
                continue
            detail = str(r[8]) if len(r) > 8 else ""

            def _extract(pattern: re.Pattern) -> Optional[float]:
                m = pattern.search(detail)
                return safe_float(m.group(1)) if m else None

            out.append(AdjEvent(
                stock_id=str(r[1]).strip(),
                event_date=_parse_roc_date(r[0]),
                market="TPEx",
                source=SOURCE_PAR_VALUE_CHANGE,
                event_type="面額變更",
                prev_close=safe_float(r[3]),
                ref_price=safe_float(r[4]),
                opening_ref=safe_float(r[7]),
                payload={
                    "split_rate": _extract(_PV_RATE_RE),
                    "par_before": _extract(_PV_PAR_BEFORE_RE),
                    "par_after": _extract(_PV_PAR_AFTER_RE),
                },
            ))
        except (IndexError, KeyError, TWSEError) as exc:
            logger.warning("TPEx pvChgRslt row skipped: %s row=%s", exc, r)
    return out


# ──────────────────────────────────────────────
# HTTP client（禮貌 rate limit + retry；TPEx 需 POST，故不直接用 TWSEClient）
# ──────────────────────────────────────────────

class OfficialAdjClient:
    """TWSE/TPEx 官方公告抓取 client。

    - 每個 request 至少間隔 ``delay`` 秒（TWSE/TPEx 社群實證約 5 秒 3 個 request 安全）
    - 5xx/429 指數退避重試
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT, "Accept": "application/json"})
        self._last_at = 0.0

    def _wait(self) -> None:
        remaining = self.delay - (time.monotonic() - self._last_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_at = time.monotonic()

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                    wait_s = 2.0 * (2 ** attempt)
                    logger.warning("[official_adj] %s status=%d retrying in %.0fs",
                                   url, resp.status_code, wait_s)
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait_s = 2.0 * (2 ** attempt)
                    logger.warning("[official_adj] %s error=%s retrying in %.0fs", url, exc, wait_s)
                    time.sleep(wait_s)
                    continue
                raise TWSEError(f"HTTP request failed: {url}: {exc}") from exc
            except ValueError as exc:
                raise TWSEError(f"JSON decode failed: {url}: {exc}") from exc
        raise TWSEError(f"HTTP request exhausted retries: {url}: {last_exc}")

    # ── raw payload fetchers（回原始 JSON，checkpoint 由呼叫端負責）──
    # cache_bust：TWSE CDN 會把偶發的錯誤 stat（「查詢結束日期小於查詢開始日期」等）
    # 以 query string 為 key 快取住，之後同 URL 永遠回同一個錯誤——重試時必須帶
    # 隨機 `_` 參數（jQuery 慣例）繞開毒快取（2026-07-10 實測）。

    @staticmethod
    def _bust(params: Dict[str, str], cache_bust: bool) -> Dict[str, str]:
        if cache_bust:
            params["_"] = str(int(time.time() * 1000))
        return params

    def fetch_twse_ex_rights_raw(self, start: date, end: date, cache_bust: bool = False) -> Any:
        return self._request_json("GET", TWSE_EX_RIGHTS_URL, params=self._bust({
            "startDate": start.strftime("%Y%m%d"),
            "endDate": end.strftime("%Y%m%d"),
            "response": "json",
        }, cache_bust))

    def fetch_twse_capital_reduction_raw(self, start: date, end: date,
                                         cache_bust: bool = False) -> Any:
        return self._request_json("GET", TWSE_CAPITAL_REDUCTION_URL, params=self._bust({
            "startDate": start.strftime("%Y%m%d"),
            "endDate": end.strftime("%Y%m%d"),
            "response": "json",
        }, cache_bust))

    def fetch_tpex_ex_rights_raw(self, start: date, end: date, cache_bust: bool = False) -> Any:
        return self._request_json("POST", TPEX_EX_RIGHTS_URL, data=self._bust({
            "startDate": start.strftime("%Y/%m/%d"),
            "endDate": end.strftime("%Y/%m/%d"),
            "response": "json",
        }, cache_bust))

    def fetch_tpex_capital_reduction_raw(self, start: date, end: date,
                                         cache_bust: bool = False) -> Any:
        return self._request_json("POST", TPEX_CAPITAL_REDUCTION_URL, data=self._bust({
            "startDate": start.strftime("%Y/%m/%d"),
            "endDate": end.strftime("%Y/%m/%d"),
            "response": "json",
        }, cache_bust))

    def fetch_twse_par_value_change_raw(self, start: date, end: date,
                                        cache_bust: bool = False) -> Any:
        return self._request_json("GET", TWSE_PAR_VALUE_CHANGE_URL, params=self._bust({
            "startDate": start.strftime("%Y%m%d"),
            "endDate": end.strftime("%Y%m%d"),
            "response": "json",
        }, cache_bust))

    def fetch_tpex_par_value_change_raw(self, start: date, end: date,
                                        cache_bust: bool = False) -> Any:
        return self._request_json("POST", TPEX_PAR_VALUE_CHANGE_URL, data=self._bust({
            "startDate": start.strftime("%Y/%m/%d"),
            "endDate": end.strftime("%Y/%m/%d"),
            "response": "json",
        }, cache_bust))


#: (fetch_kind, parser, fetcher attr)——script 端 checkpoint 用
FETCH_SPECS: List[Tuple[str, Any, str]] = [
    ("twse_ex_rights", parse_twse_ex_rights, "fetch_twse_ex_rights_raw"),
    ("twse_capital_reduction", parse_twse_capital_reduction, "fetch_twse_capital_reduction_raw"),
    ("twse_par_value_change", parse_twse_par_value_change, "fetch_twse_par_value_change_raw"),
    ("tpex_ex_rights", parse_tpex_ex_rights, "fetch_tpex_ex_rights_raw"),
    ("tpex_capital_reduction", parse_tpex_capital_reduction, "fetch_tpex_capital_reduction_raw"),
    ("tpex_par_value_change", parse_tpex_par_value_change, "fetch_tpex_par_value_change_raw"),
]


# ──────────────────────────────────────────────
# 事件比率 → 累積 factor（步階展開到全交易日）
# ──────────────────────────────────────────────

def compute_stock_factor_series(
    event_dates: Sequence[date],
    ratios: Sequence[float],
    trading_dates: Sequence[date],
) -> np.ndarray:
    """單股：事件比率 → 全交易日累積 factor（最近日 = 1.0，往過去累乘）。

    factor(d) = ∏ ratio_e for all events e with event_date > d
    （事件日當天屬「事件後」段：除權息日開盤即以參考價為基準）。

    trading_dates 必須已排序（遞增）；event_dates 不需排序、可含相同日期
    （同日多事件比率相乘，例如同日除權息 + 減資）。
    步階函數本身即是 ffill 語義（事件間缺日沿用前值），與
    build_features.apply_adj_factors 的 per-stock ffill 一致。
    """
    if len(trading_dates) == 0:
        return np.array([], dtype=float)
    if len(event_dates) != len(ratios):
        raise ValueError("event_dates 與 ratios 長度不一致")
    if len(event_dates) == 0:
        return np.ones(len(trading_dates), dtype=float)

    order = np.argsort(np.asarray(event_dates, dtype="datetime64[D]"))
    ev = np.asarray(event_dates, dtype="datetime64[D]")[order]
    rs = np.asarray(ratios, dtype=float)[order]
    if np.any(~np.isfinite(rs)) or np.any(rs <= 0):
        raise ValueError("ratios 必須為正的有限值")

    # suffix_prod[j] = ∏ rs[j:]；suffix_prod[k] = 1.0（最後一個事件之後）
    k = len(rs)
    suffix_prod = np.ones(k + 1, dtype=float)
    suffix_prod[:k] = np.multiply.accumulate(rs[::-1])[::-1]

    td = np.asarray(trading_dates, dtype="datetime64[D]")
    # idx = 該交易日（含當天）之前發生的事件數 → factor = suffix_prod[idx]
    idx = np.searchsorted(ev, td, side="right")
    return suffix_prod[idx]


def build_factor_frame(events_df: pd.DataFrame, trading_days_df: pd.DataFrame) -> pd.DataFrame:
    """全市場：官方事件 → per-stock 全交易日 factor 序列。

    Args:
        events_df: events_to_dataframe() 輸出（stock_id / event_date / ratio ...）
        trading_days_df: 該窗口內 per-stock 實際交易日（stock_id / trading_date），
            一般由 raw_prices SELECT DISTINCT 而來。

    Returns:
        DataFrame(stock_id, trading_date, adj_factor)——只含「至少一個事件」的股票
        （無事件股票 factor 恆為 1.0，消費端缺 stock 時應視同 1.0，
        與 build 端「整檔無 factor → 1.0」語義一致）。

    ⚠️ factor 只在抓取窗口內有效：窗口外（更早）的事件未知，
    序列以窗口內最後事件之後 = 1.0 為基準。
    """
    if events_df.empty:
        return pd.DataFrame(columns=["stock_id", "trading_date", "adj_factor"])
    td = trading_days_df.copy()
    td["stock_id"] = td["stock_id"].astype(str)
    td["trading_date"] = pd.to_datetime(td["trading_date"]).dt.date
    td = td.sort_values(["stock_id", "trading_date"])

    ev_groups = {sid: g for sid, g in events_df.groupby("stock_id")}
    parts: List[pd.DataFrame] = []
    for sid, g in td.groupby("stock_id", sort=False):
        ev = ev_groups.get(str(sid))
        if ev is None or ev.empty:
            continue
        dates = g["trading_date"].tolist()
        factors = compute_stock_factor_series(
            ev["event_date"].tolist(), ev["ratio"].tolist(), dates
        )
        parts.append(pd.DataFrame({
            "stock_id": str(sid),
            "trading_date": dates,
            "adj_factor": factors,
        }))
    if not parts:
        return pd.DataFrame(columns=["stock_id", "trading_date", "adj_factor"])
    return pd.concat(parts, ignore_index=True)


# ──────────────────────────────────────────────
# 對帳：官方事件 vs 凍結快照 factor
# ──────────────────────────────────────────────

#: mismatch 原因分類
REASON_MATCHED = "matched"
REASON_MATCHED_OPENING_REF = "matched_via_opening_ref"    # 用開盤競價基準才 match（現金增資口徑）
REASON_MATCHED_NEXT_GAP = "matched_via_next_snapshot_gap"  # 快照晚一交易日才調整
#   （官方比率在下一個快照 gap 命中：FinMind 把調整記在恢復買賣日的次一交易日，
#   實證：1538 官方 2024-12-30 / 快照跳 12-31、4806 官方 2025-10-03 / 快照跳 10-07。
#   官方口徑正確、快照僅該一日錯位——計入 matched 並在 summary 單獨計數。）
REASON_MISSING_IN_SNAPSHOT = "missing_in_snapshot"        # 快照無跳動（缺事件）
REASON_MISSING_REDUCTION = "missing_in_snapshot_capital_reduction"
REASON_MISSING_PAR_VALUE = "missing_in_snapshot_par_value_change"
REASON_MISSING_MARKET_CLOSED = "missing_in_snapshot_market_closed"
#   （事件日為平日但全市場停市——颱風假；FinMind 快照已證實整批漏調整
#   （2024-07-24/25 凱米 63 筆、10-02/03 山陀兒、10-31 康芮），屬快照缺陷，
#   排除於 match rate 分母、於 summary 單獨計數。官方 factor 修復這批事件。）
REASON_RATIO_DIFF_CASH_INCREASE = "ratio_diff_cash_increase"
REASON_RATIO_DIFF_REDUCTION = "ratio_diff_capital_reduction"
REASON_RATIO_DIFF_PAR_VALUE = "ratio_diff_par_value_change"
REASON_RATIO_DIFF_OTHER = "ratio_diff_other"
REASON_STOCK_NO_SNAPSHOT = "stock_no_snapshot"            # 快照無此股（下市/新上市/ETF）
REASON_CLIP_TOUCHED = "snapshot_clip_touched"             # 快照 factor 觸 clip，不可信
REASON_OUT_OF_COVERAGE = "out_of_snapshot_coverage"       # 事件日不在快照日期範圍內


@dataclass
class ReconcileResult:
    """對帳結果。

    match_rate 分母 = 可對帳事件（快照有該股、事件日在覆蓋範圍內、非 clip 股）。
    extra_jumps = 快照有跳動但官方無事件（可能：FinMind 現金增資還原、資料錯誤）。
    """

    event_results: pd.DataFrame   # 每一官方事件一列（含 matched / reason / 比率）
    extra_jumps: pd.DataFrame     # 快照多出的跳動
    summary: Dict[str, Any]

    @property
    def match_rate(self) -> float:
        return float(self.summary.get("match_rate", float("nan")))


def _classify_mismatch(group: pd.DataFrame, snap_flat: bool,
                       market_closed: bool = False) -> str:
    has_reduction = (group["source"] == SOURCE_CAPITAL_REDUCTION).any()
    has_par_value = (group["source"] == SOURCE_PAR_VALUE_CHANGE).any()
    has_cash = bool(group["cash_increase_suspected"].any())
    if snap_flat:
        if market_closed:
            return REASON_MISSING_MARKET_CLOSED
        if has_par_value:
            return REASON_MISSING_PAR_VALUE
        if has_reduction:
            return REASON_MISSING_REDUCTION
        return REASON_MISSING_IN_SNAPSHOT
    if has_cash:
        return REASON_RATIO_DIFF_CASH_INCREASE
    if has_par_value:
        return REASON_RATIO_DIFF_PAR_VALUE
    if has_reduction:
        return REASON_RATIO_DIFF_REDUCTION
    return REASON_RATIO_DIFF_OTHER


def reconcile_events_vs_snapshot(
    events_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    *,
    tolerance: float = 0.005,
    jump_threshold: float = 0.005,
    clip_touched: Optional[set] = None,
    extra_jump_min_date: Optional[date] = None,
) -> ReconcileResult:
    """官方事件 vs 快照 factor 逐 (stock, date) 對帳。

    快照側事件比率：r_snap(d) = factor(前一交易日) / factor(d)
    （factor 越早越小 → 除息日 r_snap < 1，與官方 r = 參考價/前收盤 同向可比）。

    事件對應：官方事件日 e 對應「快照中第一個 >= e 的交易日」的 gap
    （停牌／減資停止買賣期間事件自然落到恢復買賣日的 gap）；
    同一 gap 內多個官方事件比率相乘後合併比對。

    match 判準：|r_official / r_snap - 1| < tolerance（0.5%）。

    次一 gap 備援：mapped gap 平坦但官方比率在「下一個快照 gap」命中 →
    matched_via_next_snapshot_gap（FinMind 晚一交易日調整；官方正確，計入 matched）。

    全市場停市缺調整：事件日為平日但不在任何快照交易日（颱風假停市）且快照平坦 →
    missing_in_snapshot_market_closed（已證實快照缺陷，排除於分母）。
    """
    clip_touched = clip_touched or set()

    snap = snapshot_df.copy()
    snap["stock_id"] = snap["stock_id"].astype(str)
    snap["trading_date"] = pd.to_datetime(snap["trading_date"]).dt.date
    snap["adj_factor"] = pd.to_numeric(snap["adj_factor"], errors="coerce")
    snap = snap.dropna(subset=["adj_factor"]).sort_values(["stock_id", "trading_date"])
    snap_groups = {sid: g for sid, g in snap.groupby("stock_id", sort=False)}
    #: 全市場交易日（快照所有股票聯集）——判定「事件日全市場停市」用
    market_days: set = set(snap["trading_date"].unique())

    ev = events_df.copy()
    ev["stock_id"] = ev["stock_id"].astype(str)
    ev["event_date"] = pd.to_datetime(ev["event_date"]).dt.date

    result_rows: List[Dict[str, Any]] = []
    extra_jump_rows: List[Dict[str, Any]] = []
    official_gap_keys: set = set()  # (stock_id, 快照 gap 日) — 官方有事件的 gap

    for sid, g in ev.groupby("stock_id", sort=False):
        sg = snap_groups.get(sid)
        if sid in clip_touched:
            for _, row in g.iterrows():
                result_rows.append({**_event_row(row), "matched": False,
                                    "reason": REASON_CLIP_TOUCHED,
                                    "r_snap": None, "mapped_date": None})
            continue
        if sg is None or sg.empty:
            for _, row in g.iterrows():
                result_rows.append({**_event_row(row), "matched": False,
                                    "reason": REASON_STOCK_NO_SNAPSHOT,
                                    "r_snap": None, "mapped_date": None})
            continue

        dates = np.asarray(sg["trading_date"].tolist(), dtype="datetime64[D]")
        factors = sg["adj_factor"].to_numpy(dtype=float)

        # 每一官方事件對應快照 gap（第一個 >= event_date 的快照日）
        ev_dates = np.asarray(g["event_date"].tolist(), dtype="datetime64[D]")
        pos = np.searchsorted(dates, ev_dates, side="left")

        g = g.assign(_pos=pos)
        for p, grp in g.groupby("_pos"):
            p = int(p)
            if p <= 0 or p >= len(dates):
                # 事件日早於快照第一天（無前值可算 gap）或晚於快照最後一天
                for _, row in grp.iterrows():
                    result_rows.append({**_event_row(row), "matched": False,
                                        "reason": REASON_OUT_OF_COVERAGE,
                                        "r_snap": None, "mapped_date": None})
                continue
            mapped_date = dates[p].astype(object)
            r_snap = factors[p - 1] / factors[p] if factors[p] > 0 else np.nan
            r_official = float(np.prod(grp["ratio"].to_numpy(dtype=float)))
            ratio_open = grp["ratio_opening"].astype(float)
            r_official_open = (
                float(np.prod(ratio_open.to_numpy()))
                if ratio_open.notna().all() else None
            )

            matched, reason = False, None
            if np.isfinite(r_snap):
                if abs(r_official / r_snap - 1.0) < tolerance:
                    matched, reason = True, REASON_MATCHED
                elif r_official_open is not None and abs(r_official_open / r_snap - 1.0) < tolerance:
                    matched, reason = True, REASON_MATCHED_OPENING_REF
            snap_flat = np.isfinite(r_snap) and abs(r_snap - 1.0) <= jump_threshold
            if not matched and snap_flat and p + 1 < len(dates):
                # 次一 gap 備援：FinMind 晚一交易日調整（1538/4806 pattern）
                r_next = factors[p] / factors[p + 1] if factors[p + 1] > 0 else np.nan
                if np.isfinite(r_next) and (
                    abs(r_official / r_next - 1.0) < tolerance
                    or (r_official_open is not None
                        and abs(r_official_open / r_next - 1.0) < tolerance)
                ):
                    matched, reason = True, REASON_MATCHED_NEXT_GAP
                    mapped_date = dates[p + 1].astype(object)
                    r_snap = r_next
            if not matched:
                market_closed = snap_flat and all(
                    d.weekday() < 5 and d not in market_days
                    for d in grp["event_date"]
                )
                reason = _classify_mismatch(grp, snap_flat, market_closed)
            official_gap_keys.add((sid, mapped_date))

            for _, row in grp.iterrows():
                result_rows.append({
                    **_event_row(row),
                    "matched": matched,
                    "reason": reason,
                    "r_official_gap": r_official,
                    "r_snap": float(r_snap) if np.isfinite(r_snap) else None,
                    "rel_diff": (float(abs(r_official / r_snap - 1.0))
                                 if np.isfinite(r_snap) and r_snap > 0 else None),
                    "mapped_date": mapped_date,
                })

    # ── 快照多出的跳動（官方無事件）──
    for sid, sg in snap_groups.items():
        if sid in clip_touched:
            continue
        factors = sg["adj_factor"].to_numpy(dtype=float)
        dts = sg["trading_date"].tolist()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = factors[:-1] / factors[1:]
        for i in np.nonzero(np.isfinite(r) & (np.abs(r - 1.0) > jump_threshold))[0]:
            d = dts[i + 1]
            if (sid, d) in official_gap_keys:
                continue
            if extra_jump_min_date is not None and d < extra_jump_min_date:
                continue  # 快照為算首事件 gap 多載了 buffer 天，buffer 段跳動非官方窗內
            extra_jump_rows.append({
                "stock_id": sid, "trading_date": d,
                "r_snap": float(r[i]),
                "prev_date": dts[i],
            })

    event_results = pd.DataFrame(result_rows)
    extra_jumps = pd.DataFrame(extra_jump_rows,
                               columns=["stock_id", "trading_date", "r_snap", "prev_date"])

    # ── summary ──
    # REASON_MISSING_MARKET_CLOSED：已證實快照缺陷（颱風停市日 FinMind 整批漏調整），
    # 排除於 match rate 分母，於 n_missing_market_closed 單獨計數。
    excluded_reasons = {REASON_STOCK_NO_SNAPSHOT, REASON_CLIP_TOUCHED,
                        REASON_OUT_OF_COVERAGE, REASON_MISSING_MARKET_CLOSED}
    if event_results.empty:
        eligible = matched = pd.DataFrame()
        n_eligible = n_matched = 0
    else:
        eligible = event_results[~event_results["reason"].isin(excluded_reasons)]
        matched = eligible[eligible["matched"]]
        n_eligible, n_matched = len(eligible), len(matched)

    reason_counts = (event_results["reason"].value_counts().to_dict()
                     if not event_results.empty else {})
    summary: Dict[str, Any] = {
        "n_official_events": int(len(event_results)),
        "n_eligible": int(n_eligible),
        "n_matched": int(n_matched),
        "match_rate": (n_matched / n_eligible) if n_eligible else None,
        "n_matched_via_opening_ref": int(
            (eligible["reason"] == REASON_MATCHED_OPENING_REF).sum()) if n_eligible else 0,
        "n_matched_via_next_gap": int(
            (eligible["reason"] == REASON_MATCHED_NEXT_GAP).sum()) if n_eligible else 0,
        "n_missing_market_closed": int(reason_counts.get(REASON_MISSING_MARKET_CLOSED, 0)),
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "n_extra_snapshot_jumps": int(len(extra_jumps)),
        "tolerance": tolerance,
        "jump_threshold": jump_threshold,
    }
    return ReconcileResult(event_results=event_results, extra_jumps=extra_jumps, summary=summary)


def _event_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "stock_id": row["stock_id"],
        "event_date": row["event_date"],
        "market": row["market"],
        "source": row["source"],
        "event_type": row["event_type"],
        "prev_close": row["prev_close"],
        "ref_price": row["ref_price"],
        "r_official": row["ratio"],
        "r_official_opening": row.get("ratio_opening"),
        "cash_increase_suspected": bool(row.get("cash_increase_suspected", False)),
    }


def load_clip_touched(path: str) -> set:
    """讀 populate_adj_factors 產出的 clip 觸及清單（快照 factor 不可信股票）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).keys())
    except FileNotFoundError:
        logger.warning("[official_adj] clip 清單不存在：%s（不排除任何股票）", path)
        return set()
