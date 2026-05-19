"""TWSE/TPEx 官方資料 client（取代 FinMind 用，降低 rate limit 壓力）。

研究結論（2026-05-19）：
- 官方 OpenAPI（openapi.twse.com.tw, www.tpex.org.tw/openapi）所有 endpoint **不接受 date 參數**，
  永遠回最新一個交易日。OpenAPI 僅能做日增量。
- 若要 backfill 歷史，必須走 Legacy endpoint（www.twse.com.tw/rwd/zh/…、
  www.tpex.org.tw/www/zh-tw/…），可指定 date。
- 三大法人個股 daily（T86）TWSE OpenAPI 無此資料，只能走 Legacy。

本 client 提供 4 類資料 × (TWSE 上市, TPEx 上櫃) × (latest, history) 共 16 個 method。
所有 raw 回傳統一 normalize 成貼近 FinMind schema 的 `dict` list，方便後續對照。

Rate limit：官方未公佈精確值，社群實證約「5 秒 3 個 request」是安全值。
本 client 預設每個 request 間至少間隔 `DEFAULT_DELAY` 秒（可由 env var 覆寫）。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_DELAY: float = float(os.environ.get("TWSE_REQUEST_DELAY", "1.5"))
DEFAULT_TIMEOUT: float = float(os.environ.get("TWSE_TIMEOUT", "30"))
DEFAULT_MAX_RETRIES: int = int(os.environ.get("TWSE_MAX_RETRIES", "3"))
DEFAULT_RETRY_BACKOFF: float = float(os.environ.get("TWSE_RETRY_BACKOFF", "2.0"))
USER_AGENT: str = "Mozilla/5.0 (compatible; stock_bot/1.0)"

# 5xx + 429（Cloudflare 限流）視為可重試。404/400/403 不重試（語意錯誤）。
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504, 520, 521, 522, 524}

# TWSE OpenAPI（無 date 參數）
TWSE_OAPI_STOCK_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_OAPI_MI_MARGN = "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN"
TWSE_OAPI_BWIBBU_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"

# TWSE Legacy（接受 date=YYYYMMDD）
TWSE_LEGACY_STOCK_DAY_ALL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"
TWSE_LEGACY_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TWSE_LEGACY_MI_MARGN = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
TWSE_LEGACY_BWIBBU = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"

# TPEx OpenAPI（無 date 參數）
TPEX_OAPI_DAILY_QUOTES = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TPEX_OAPI_3INSTI = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"
TPEX_OAPI_MARGIN = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_balance"
TPEX_OAPI_PER = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"

# TPEx Legacy（接受 date=YYYY/MM/DD）
TPEX_LEGACY_DAILY_QUOTES = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
TPEX_LEGACY_3INSTI = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
TPEX_LEGACY_MARGIN = "https://www.tpex.org.tw/www/zh-tw/margin/balance"
TPEX_LEGACY_PER = "https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate"


class TWSEError(Exception):
    """TWSE/TPEx 請求或解析失敗。"""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def roc_date_to_west(roc_str: str) -> date:
    """民國年字串轉西元 date。

    支援格式：
    - '1150515'（7 碼，無分隔）
    - '115/05/15'（斜線分隔）

    無法解析時 raise TWSEError。
    """
    s = (roc_str or "").strip()
    if not s:
        raise TWSEError(f"empty ROC date string")
    try:
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 3:
                raise ValueError(f"slash format expects 3 parts, got {parts}")
            year = int(parts[0]) + 1911
            month = int(parts[1])
            day = int(parts[2])
        else:
            if len(s) != 7:
                raise ValueError(f"compact format expects 7 chars, got {len(s)}")
            year = int(s[:3]) + 1911
            month = int(s[3:5])
            day = int(s[5:7])
        return date(year, month, day)
    except (ValueError, IndexError) as exc:
        raise TWSEError(f"unparseable ROC date: {roc_str!r}") from exc


def strip_commas(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).replace(",", "").strip()


def safe_float(s: Optional[str]) -> Optional[float]:
    """TWSE 數值解析。空字串 / '-' / '--' 視為 None（缺值或 N/A）。"""
    stripped = strip_commas(s)
    if stripped in ("", "-", "--", "X"):
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def safe_int(s: Optional[str]) -> Optional[int]:
    val = safe_float(s)
    if val is None:
        return None
    return int(val)


def normalize_key(key: str) -> str:
    """處理 TPEx 3insti JSON key 的 leading/trailing 空白與大小寫不一致 bug。"""
    return "".join(ch for ch in key.lower().strip() if ch.isalnum())


# ──────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────

@dataclass
class _RateLimiter:
    delay: float
    _last_at: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_at
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_at = time.monotonic()


class TWSEClient:
    """TWSE + TPEx 統一 client。

    使用方式：
        client = TWSEClient()
        # 日增量（OpenAPI，無 date 參數）
        rows = client.fetch_prices_latest()
        # 歷史回補（Legacy，可指定 date）
        rows = client.fetch_prices_history(date(2026, 5, 15))

    所有 fetch_* method 都回傳 list[dict]，schema 接近 FinMind。
    失敗時 raise TWSEError；個別 row 解析失敗會 logger.warning 並 skip。
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._limiter = _RateLimiter(delay=delay)

    # ── HTTP 底層 ────────────────────────────────
    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.timeout, allow_redirects=True
                )
                if resp.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                    wait_s = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "TWSE %s status=%d (retryable) attempt %d/%d, sleeping %.1fs",
                        url, resp.status_code, attempt + 1, self.max_retries, wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait_s = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "TWSE %s request error attempt %d/%d: %s; sleeping %.1fs",
                        url, attempt + 1, self.max_retries, exc, wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                raise TWSEError(f"HTTP request failed: {url} params={params}: {exc}") from exc
            except ValueError as exc:
                raise TWSEError(f"JSON decode failed: {url} params={params}: {exc}") from exc
        raise TWSEError(f"HTTP request exhausted retries: {url} params={params}: {last_exc}")

    # ── 1. OHLCV ───────────────────────────────────
    def fetch_prices_latest(self) -> List[Dict[str, Any]]:
        """TWSE + TPEx OpenAPI 最新一日全市場 OHLCV。"""
        rows: List[Dict[str, Any]] = []
        rows.extend(self._parse_twse_oapi_stock_day(self._get_json(TWSE_OAPI_STOCK_DAY_ALL)))
        rows.extend(self._parse_tpex_oapi_quotes(self._get_json(TPEX_OAPI_DAILY_QUOTES)))
        return rows

    def fetch_prices_history(self, trading_date: date) -> List[Dict[str, Any]]:
        """TWSE + TPEx Legacy 指定 date 全市場 OHLCV。

        ⚠️ 實測（2026-05-19）：**TWSE STOCK_DAY_ALL legacy 不接受 date 參數**，
        即使送 `?date=20260515` 也永遠回最新一天。本方法呼叫後會驗證 server
        回的 `date` 是否等於請求的 `trading_date`，不一致時：
        - WARN log + 拋出 TWSEError 避免把錯日資料寫入 DB

        如果你需要的是「最新一天」，請改用 fetch_prices_latest()，行為明確。
        """
        rows: List[Dict[str, Any]] = []
        twse_resp = self._get_json(
            TWSE_LEGACY_STOCK_DAY_ALL,
            params={"date": trading_date.strftime("%Y%m%d"), "response": "json"},
        )
        # Sanity check：legacy STOCK_DAY_ALL 回的 date 必須 == 請求 date，否則代表
        # server 忽略了 date 參數（已實測 TWSE 此 endpoint 不尊重 date）。
        server_date = (twse_resp.get("date") if isinstance(twse_resp, dict) else None) or ""
        expected = trading_date.strftime("%Y%m%d")
        if server_date and server_date != expected:
            raise TWSEError(
                f"TWSE STOCK_DAY_ALL ignored date param: requested {expected}, "
                f"server returned {server_date}. Use fetch_prices_latest() instead."
            )
        rows.extend(self._parse_twse_legacy_stock_day(twse_resp, trading_date))
        tpex_resp = self._get_json(
            TPEX_LEGACY_DAILY_QUOTES,
            params={"date": trading_date.strftime("%Y/%m/%d"), "type": "EW", "response": "json"},
        )
        rows.extend(self._parse_tpex_legacy_quotes(tpex_resp, trading_date))
        return rows

    # ── 2. 三大法人（個股 daily）──────────────────
    def fetch_institutional_history(self, trading_date: date) -> List[Dict[str, Any]]:
        """TWSE T86 Legacy + TPEx 3insti Legacy。TWSE OpenAPI 沒有 T86，**必走 legacy**。"""
        rows: List[Dict[str, Any]] = []
        twse_resp = self._get_json(
            TWSE_LEGACY_T86,
            params={
                "date": trading_date.strftime("%Y%m%d"),
                "selectType": "ALLBUT0999",
                "response": "json",
            },
        )
        rows.extend(self._parse_twse_legacy_t86(twse_resp, trading_date))
        tpex_resp = self._get_json(
            TPEX_LEGACY_3INSTI,
            params={
                "date": trading_date.strftime("%Y/%m/%d"),
                "type": "Daily",
                "sect": "EW",
                "response": "json",
            },
        )
        rows.extend(self._parse_tpex_legacy_3insti(tpex_resp, trading_date))
        return rows

    # ── 3. 融資融券 ────────────────────────────────
    def fetch_margin_short_latest(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._parse_twse_oapi_margin(self._get_json(TWSE_OAPI_MI_MARGN)))
        rows.extend(self._parse_tpex_oapi_margin(self._get_json(TPEX_OAPI_MARGIN)))
        return rows

    def fetch_margin_short_history(self, trading_date: date) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        twse_resp = self._get_json(
            TWSE_LEGACY_MI_MARGN,
            params={
                "date": trading_date.strftime("%Y%m%d"),
                "selectType": "ALL",
                "response": "json",
            },
        )
        rows.extend(self._parse_twse_legacy_margin(twse_resp, trading_date))
        tpex_resp = self._get_json(
            TPEX_LEGACY_MARGIN,
            params={"date": trading_date.strftime("%Y/%m/%d"), "response": "json"},
        )
        rows.extend(self._parse_tpex_legacy_margin(tpex_resp, trading_date))
        return rows

    # ── 4. PER / PBR / 殖利率 ─────────────────────
    def fetch_per_latest(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._parse_twse_oapi_per(self._get_json(TWSE_OAPI_BWIBBU_ALL)))
        rows.extend(self._parse_tpex_oapi_per(self._get_json(TPEX_OAPI_PER)))
        return rows

    def fetch_per_history(self, trading_date: date) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        twse_resp = self._get_json(
            TWSE_LEGACY_BWIBBU,
            params={"date": trading_date.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"},
        )
        rows.extend(self._parse_twse_legacy_per(twse_resp, trading_date))
        tpex_resp = self._get_json(
            TPEX_LEGACY_PER,
            params={"date": trading_date.strftime("%Y/%m/%d"), "response": "json"},
        )
        rows.extend(self._parse_tpex_legacy_per(tpex_resp, trading_date))
        return rows

    # ──────────────────────────────────────────────
    # Parsers — pure functions, no network I/O
    # （所有 parser 取為 staticmethod 方便用 fixture JSON 測試）
    # ──────────────────────────────────────────────

    @staticmethod
    def _parse_twse_oapi_stock_day(payload: Any) -> List[Dict[str, Any]]:
        """TWSE OpenAPI STOCK_DAY_ALL: 扁平 array，英文 key。"""
        if not isinstance(payload, list):
            raise TWSEError(f"expected list for TWSE OpenAPI STOCK_DAY_ALL, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload:
            try:
                out.append({
                    "stock_id": r["Code"],
                    "name": r.get("Name"),
                    "trading_date": roc_date_to_west(r["Date"]),
                    "open": safe_float(r.get("OpeningPrice")),
                    "high": safe_float(r.get("HighestPrice")),
                    "low": safe_float(r.get("LowestPrice")),
                    "close": safe_float(r.get("ClosingPrice")),
                    "volume": safe_int(r.get("TradeVolume")),
                    "amount": safe_float(r.get("TradeValue")),
                    "transactions": safe_int(r.get("Transaction")),
                    "spread": safe_float(r.get("Change")),
                    "market": "TWSE",
                })
            except (KeyError, TWSEError) as exc:
                logger.warning("TWSE OAPI STOCK_DAY row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_oapi_quotes(payload: Any) -> List[Dict[str, Any]]:
        """TPEx OpenAPI daily_close_quotes: 扁平 array，英文 key（注意 LatesAskPrice 拼字錯字）。"""
        if not isinstance(payload, list):
            raise TWSEError(f"expected list for TPEx OAPI quotes, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload:
            try:
                out.append({
                    "stock_id": r["SecuritiesCompanyCode"],
                    "name": r.get("CompanyName"),
                    "trading_date": roc_date_to_west(r["Date"]),
                    "open": safe_float(r.get("Open")),
                    "high": safe_float(r.get("High")),
                    "low": safe_float(r.get("Low")),
                    "close": safe_float(r.get("Close")),
                    "volume": safe_int(r.get("TradingShares")),
                    "amount": safe_float(r.get("TransactionAmount")),
                    "transactions": safe_int(r.get("TransactionNumber")),
                    "spread": safe_float((r.get("Change") or "").strip()),
                    "market": "TPEx",
                })
            except (KeyError, TWSEError) as exc:
                logger.warning("TPEx OAPI quotes row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_twse_legacy_stock_day(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        """TWSE Legacy STOCK_DAY_ALL: {stat, date, fields, data}，data row 為 list。

        實測欄位順序（2026-05 確認）：
        0:證券代號 1:證券名稱 2:成交股數 3:成交金額 4:開盤價
        5:最高價 6:最低價 7:收盤價 8:漲跌價差 9:成交筆數
        """
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TWSE legacy STOCK_DAY_ALL, got {type(payload).__name__}")
        if payload.get("stat") and payload["stat"] != "OK":
            logger.warning("TWSE legacy STOCK_DAY_ALL stat=%s", payload["stat"])
            return []
        out: List[Dict[str, Any]] = []
        for r in payload.get("data") or []:
            try:
                if not r or len(r) < 9:
                    continue
                out.append({
                    "stock_id": r[0],
                    "name": r[1] if len(r) > 1 else None,
                    "trading_date": trading_date,
                    "volume": safe_int(r[2]),
                    "amount": safe_float(r[3]),
                    "open": safe_float(r[4]),
                    "high": safe_float(r[5]),
                    "low": safe_float(r[6]),
                    "close": safe_float(r[7]),
                    "spread": safe_float(r[8]),
                    "transactions": safe_int(r[9]) if len(r) > 9 else None,
                    "market": "TWSE",
                })
            except (IndexError, KeyError) as exc:
                logger.warning("TWSE legacy STOCK_DAY row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_legacy_quotes(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        """TPEx Legacy dailyQuotes: {date, tables: [{fields, data}]}，中文 fields。

        TPEx data row 欄位（依官網實測）：
        [代號, 名稱, 收盤, 漲跌, 開盤, 最高, 最低, 均價, 成交股數, 成交金額, 成交筆數, ...]
        """
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TPEx legacy quotes, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for table in payload.get("tables") or []:
            for r in table.get("data") or []:
                try:
                    if not r or len(r) < 9:
                        continue
                    out.append({
                        "stock_id": r[0],
                        "name": r[1] if len(r) > 1 else None,
                        "trading_date": trading_date,
                        "close": safe_float(r[2]),
                        "spread": safe_float(r[3]) if len(r) > 3 else None,
                        "open": safe_float(r[4]) if len(r) > 4 else None,
                        "high": safe_float(r[5]) if len(r) > 5 else None,
                        "low": safe_float(r[6]) if len(r) > 6 else None,
                        "volume": safe_int(r[8]) if len(r) > 8 else None,
                        "amount": safe_float(r[9]) if len(r) > 9 else None,
                        "transactions": safe_int(r[10]) if len(r) > 10 else None,
                        "market": "TPEx",
                    })
                except (IndexError, KeyError) as exc:
                    logger.warning("TPEx legacy quotes row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_twse_legacy_t86(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        """TWSE Legacy T86 三大法人：19 個欄位的扁平 row。

        欄位（依官網順序）：
        0: 證券代號  1: 證券名稱
        2-4: 外陸資 買進/賣出/買賣超（不含外資自營商）
        5-7: 外資自營商 買進/賣出/買賣超
        8-10: 投信 買進/賣出/買賣超
        11: 自營商買賣超
        12-14: 自營商(自行買賣) 買進/賣出/買賣超
        15-17: 自營商(避險) 買進/賣出/買賣超
        18: 三大法人買賣超
        """
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TWSE legacy T86, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload.get("data") or []:
            try:
                if not r or len(r) < 19:
                    continue
                foreign_net = (safe_int(r[4]) or 0) + (safe_int(r[7]) or 0)
                trust_net = safe_int(r[10])
                dealer_self_net = safe_int(r[14])
                dealer_hedging_net = safe_int(r[17])
                out.append({
                    "stock_id": r[0],
                    "name": r[1],
                    "trading_date": trading_date,
                    "foreign_buy": (safe_int(r[2]) or 0) + (safe_int(r[5]) or 0),
                    "foreign_sell": (safe_int(r[3]) or 0) + (safe_int(r[6]) or 0),
                    "foreign_net": foreign_net,
                    "trust_buy": safe_int(r[8]),
                    "trust_sell": safe_int(r[9]),
                    "trust_net": trust_net,
                    "dealer_self_net": dealer_self_net,
                    "dealer_hedging_net": dealer_hedging_net,
                    "dealer_net": safe_int(r[11]),
                    "total_net": safe_int(r[18]),
                    "market": "TWSE",
                })
            except (IndexError, KeyError) as exc:
                logger.warning("TWSE legacy T86 row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_legacy_3insti(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        """TPEx Legacy insti/dailyTrade：中文欄位重複，依固定順序解析。

        實測欄位順序（25 欄）：
        0: 代號  1: 名稱
        2-4: 外資及陸資合計（不含外資自營商）買/賣/淨
        5-7: 外資自營商 買/賣/淨
        8-10: 外資及陸資合計（含外資自營商）買/賣/淨
        11-13: 投信 買/賣/淨
        14-16: 自營商(自行買賣) 買/賣/淨
        17-19: 自營商(避險) 買/賣/淨
        20-22: 自營商合計 買/賣/淨
        23: 三大法人合計淨買賣 ... 餘略
        """
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TPEx legacy 3insti, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for table in payload.get("tables") or []:
            for r in table.get("data") or []:
                try:
                    if not r or len(r) < 23:
                        continue
                    out.append({
                        "stock_id": r[0],
                        "name": r[1],
                        "trading_date": trading_date,
                        "foreign_buy": safe_int(r[8]),
                        "foreign_sell": safe_int(r[9]),
                        "foreign_net": safe_int(r[10]),
                        "trust_buy": safe_int(r[11]),
                        "trust_sell": safe_int(r[12]),
                        "trust_net": safe_int(r[13]),
                        "dealer_self_net": safe_int(r[16]),
                        "dealer_hedging_net": safe_int(r[19]),
                        "dealer_net": safe_int(r[22]),
                        "total_net": safe_int(r[23]) if len(r) > 23 else None,
                        "market": "TPEx",
                    })
                except (IndexError, KeyError) as exc:
                    logger.warning("TPEx legacy 3insti row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_twse_oapi_margin(payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            raise TWSEError(f"expected list for TWSE OAPI margin, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload:
            try:
                out.append({
                    "stock_id": r.get("股票代號") or r.get("Code"),
                    "name": r.get("股票名稱") or r.get("Name"),
                    "margin_purchase_buy": safe_int(r.get("融資買進")),
                    "margin_purchase_sell": safe_int(r.get("融資賣出")),
                    "margin_purchase_cash_repayment": safe_int(r.get("融資現金償還")),
                    "margin_purchase_yesterday_balance": safe_int(r.get("融資前日餘額")),
                    "margin_purchase_today_balance": safe_int(r.get("融資今日餘額")),
                    "margin_purchase_limit": safe_int(r.get("融資限額")),
                    "short_sale_buy": safe_int(r.get("融券買進")),
                    "short_sale_sell": safe_int(r.get("融券賣出")),
                    "short_sale_cash_repayment": safe_int(r.get("融券現券償還")),
                    "short_sale_yesterday_balance": safe_int(r.get("融券前日餘額")),
                    "short_sale_today_balance": safe_int(r.get("融券今日餘額")),
                    "short_sale_limit": safe_int(r.get("融券限額")),
                    "offset_loan_and_short": safe_int(r.get("資券互抵")),
                    "note": (r.get("註記") or "").strip(),
                    "market": "TWSE",
                })
            except (KeyError, TWSEError) as exc:
                logger.warning("TWSE OAPI margin row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_oapi_margin(payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            raise TWSEError(f"expected list for TPEx OAPI margin, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload:
            try:
                out.append({
                    "stock_id": r.get("SecuritiesCompanyCode"),
                    "name": r.get("CompanyName"),
                    "trading_date": roc_date_to_west(r["Date"]) if r.get("Date") else None,
                    "margin_purchase_yesterday_balance": safe_int(r.get("MarginPurchaseBalancePreviousDay")),
                    "margin_purchase_buy": safe_int(r.get("MarginPurchase")),
                    "margin_purchase_sell": safe_int(r.get("MarginSales")),
                    "margin_purchase_cash_repayment": safe_int(r.get("CashRedemption")),
                    "margin_purchase_today_balance": safe_int(r.get("MarginPurchaseBalance")),
                    "margin_purchase_limit": safe_int(r.get("MarginPurchaseQuota")),
                    "short_sale_yesterday_balance": safe_int(r.get("ShortSaleBalancePreviousDay")),
                    "short_sale_sell": safe_int(r.get("ShortSale")),
                    # 注意：TPEx OAPI 把 ShortCovering 拼成 ShortConvering（官方錯字）
                    "short_sale_buy": safe_int(r.get("ShortConvering") or r.get("ShortCovering")),
                    "short_sale_cash_repayment": safe_int(r.get("StockRedemption")),
                    "short_sale_today_balance": safe_int(r.get("ShortSaleBalance")),
                    "short_sale_limit": safe_int(r.get("ShortSaleQuota")),
                    "offset_loan_and_short": safe_int(r.get("Offsetting")),
                    "note": (r.get("Note") or "").strip(),
                    "market": "TPEx",
                })
            except (KeyError, TWSEError) as exc:
                logger.warning("TPEx OAPI margin row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_twse_legacy_margin(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TWSE legacy margin, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        # legacy 回傳含兩個 tables，個股 daily 通常在第二個
        for table in payload.get("tables") or []:
            for r in table.get("data") or []:
                if not r or len(r) < 16:
                    continue
                # 個股 daily 第一欄為股票代號（4 碼），市場彙總列首欄為文字
                stock_id = r[0]
                if not (isinstance(stock_id, str) and stock_id and stock_id[0].isdigit()):
                    continue
                try:
                    out.append({
                        "stock_id": stock_id,
                        "name": r[1],
                        "trading_date": trading_date,
                        "margin_purchase_buy": safe_int(r[2]),
                        "margin_purchase_sell": safe_int(r[3]),
                        "margin_purchase_cash_repayment": safe_int(r[4]),
                        "margin_purchase_yesterday_balance": safe_int(r[5]),
                        "margin_purchase_today_balance": safe_int(r[6]),
                        "margin_purchase_limit": safe_int(r[7]),
                        "short_sale_buy": safe_int(r[8]),
                        "short_sale_sell": safe_int(r[9]),
                        "short_sale_cash_repayment": safe_int(r[10]),
                        "short_sale_yesterday_balance": safe_int(r[11]),
                        "short_sale_today_balance": safe_int(r[12]),
                        "short_sale_limit": safe_int(r[13]),
                        "offset_loan_and_short": safe_int(r[14]),
                        "note": (r[15] or "").strip() if isinstance(r[15], str) else "",
                        "market": "TWSE",
                    })
                except (IndexError, KeyError) as exc:
                    logger.warning("TWSE legacy margin row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_legacy_margin(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TPEx legacy margin, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for table in payload.get("tables") or []:
            for r in table.get("data") or []:
                if not r or len(r) < 18:
                    continue
                stock_id = r[0]
                if not (isinstance(stock_id, str) and stock_id and stock_id[0].isdigit()):
                    continue
                try:
                    out.append({
                        "stock_id": stock_id,
                        "name": r[1],
                        "trading_date": trading_date,
                        "margin_purchase_yesterday_balance": safe_int(r[2]),
                        "margin_purchase_buy": safe_int(r[3]),
                        "margin_purchase_sell": safe_int(r[4]),
                        "margin_purchase_cash_repayment": safe_int(r[5]),
                        "margin_purchase_today_balance": safe_int(r[6]),
                        "margin_purchase_limit": safe_int(r[9]),
                        "short_sale_yesterday_balance": safe_int(r[10]),
                        "short_sale_sell": safe_int(r[11]),
                        "short_sale_buy": safe_int(r[12]),
                        "short_sale_cash_repayment": safe_int(r[13]),
                        "short_sale_today_balance": safe_int(r[14]),
                        "short_sale_limit": safe_int(r[17]),
                        "offset_loan_and_short": safe_int(r[18]) if len(r) > 18 else None,
                        "note": (r[19] or "").strip() if len(r) > 19 and isinstance(r[19], str) else "",
                        "market": "TPEx",
                    })
                except (IndexError, KeyError) as exc:
                    logger.warning("TPEx legacy margin row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_twse_oapi_per(payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            raise TWSEError(f"expected list for TWSE OAPI PER, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload:
            try:
                out.append({
                    "stock_id": r["Code"],
                    "name": r.get("Name"),
                    "trading_date": roc_date_to_west(r["Date"]) if r.get("Date") else None,
                    "per": safe_float(r.get("PEratio")),
                    "pbr": safe_float(r.get("PBratio")),
                    "dividend_yield": safe_float(r.get("DividendYield")),
                    "market": "TWSE",
                })
            except (KeyError, TWSEError) as exc:
                logger.warning("TWSE OAPI PER row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_oapi_per(payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            raise TWSEError(f"expected list for TPEx OAPI PER, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        for r in payload:
            try:
                out.append({
                    "stock_id": r["SecuritiesCompanyCode"],
                    "name": r.get("CompanyName"),
                    "trading_date": roc_date_to_west(r["Date"]) if r.get("Date") else None,
                    "per": safe_float(r.get("PriceEarningRatio")),
                    "pbr": safe_float(r.get("PriceBookRatio")),
                    "dividend_yield": safe_float(r.get("YieldRatio")),
                    "dividend_per_share": safe_float(r.get("DividendPerShare")),
                    "market": "TPEx",
                })
            except (KeyError, TWSEError) as exc:
                logger.warning("TPEx OAPI PER row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_twse_legacy_per(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TWSE legacy PER, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        # fields: 證券代號、證券名稱、收盤價、殖利率(%)、股利年度、本益比、股價淨值比、財報年/季
        for r in payload.get("data") or []:
            try:
                if not r or len(r) < 7:
                    continue
                out.append({
                    "stock_id": r[0],
                    "name": r[1],
                    "trading_date": trading_date,
                    "close": safe_float(r[2]),
                    "dividend_yield": safe_float(r[3]),
                    "dividend_year": r[4] if len(r) > 4 else None,
                    "per": safe_float(r[5]),
                    "pbr": safe_float(r[6]),
                    "fiscal_year_q": r[7] if len(r) > 7 else None,
                    "market": "TWSE",
                })
            except (IndexError, KeyError) as exc:
                logger.warning("TWSE legacy PER row skipped: %s row=%s", exc, r)
        return out

    @staticmethod
    def _parse_tpex_legacy_per(payload: Any, trading_date: date) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            raise TWSEError(f"expected dict for TPEx legacy PER, got {type(payload).__name__}")
        out: List[Dict[str, Any]] = []
        # fields: 股票代號、公司名稱、本益比、每股股利、股利年度、殖利率(%)、股價淨值比、財報年/季
        for table in payload.get("tables") or []:
            for r in table.get("data") or []:
                try:
                    if not r or len(r) < 7:
                        continue
                    out.append({
                        "stock_id": r[0],
                        "name": r[1],
                        "trading_date": trading_date,
                        "per": safe_float(r[2]),
                        "dividend_per_share": safe_float(r[3]) if len(r) > 3 else None,
                        "dividend_year": r[4] if len(r) > 4 else None,
                        "dividend_yield": safe_float(r[5]) if len(r) > 5 else None,
                        "pbr": safe_float(r[6]) if len(r) > 6 else None,
                        "fiscal_year_q": r[7] if len(r) > 7 else None,
                        "market": "TPEx",
                    })
                except (IndexError, KeyError) as exc:
                    logger.warning("TPEx legacy PER row skipped: %s row=%s", exc, r)
        return out
