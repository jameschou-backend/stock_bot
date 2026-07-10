"""處置股 / 注意股 live-only 名單（執行衛生過濾，非 alpha）。

處置股在處置期間分盤交易（約 5~20 分鐘撮合一次）＋預收款券，月底換股可能出不掉；
daily_pick 對「進場候選」剔除處置股，注意股僅記錄（attention_flagged）。

資料來源（官方 OpenAPI，2026-07-10 以 swagger.json + 實際 payload 驗證）：
- TWSE 處置：https://openapi.twse.com.tw/v1/announcement/punish（集中市場公布處置股票）
  欄位：Code / Name / Date(民國 1150702) / DispositionPeriod("115/07/03～115/07/16")
- TWSE 注意：https://openapi.twse.com.tw/v1/announcement/notice（集中市場當日公布注意股票）
  ⚠️ 無資料時回傳單列 placeholder（Number="0"、Code=""），需跳過空 Code。
- TPEx 處置：https://www.tpex.org.tw/openapi/v1/tpex_disposal_information（上櫃處置有價證券資訊）
  欄位：SecuritiesCompanyCode / CompanyName / Date / DispositionPeriod("1150710~1150723")
- TPEx 注意：https://www.tpex.org.tw/openapi/v1/tpex_trading_warning_information（上櫃公布注意股票資訊）

OpenAPI 均為「最新快照」、不接受 date 參數（同 app/twse_client.py 2026-05-19 研究結論），
live-only 過濾夠用；回測一致版需歷史名單回補，另案處理。

語義：
- 處置名單含近一個月內公告（處置期可能已結束），僅納入「處置迄日 >= as_of」者；
  處置期間無法解析時**保守納入**（執行衛生：寧可多剔一檔，不可買到分盤股）。
- 公告含權證（6 碼）/可轉債（5 碼），回傳集合僅保留四碼股票（專案 stock_id 規範，
  非四碼永遠不會是 daily_pick 候選）；原始名單完整留在快取 records 供稽核。
- 快取：artifacts/disposition/YYYY-MM-DD.json，當日已抓過直接讀快取；
  部分來源失敗時**不寫快取**（同日重跑可重試），僅回傳已取得的部分名單。
- 失敗語義 fail-open：任何錯誤 → 回空集合 + logger.warning 留痕
  （寧可當天不過濾，也不能讓 daily_pick 掛掉）。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from app.twse_client import TWSEError, roc_date_to_west

logger = logging.getLogger(__name__)

# ── 官方 OpenAPI endpoints（無 date 參數，永遠回最新快照）──
TWSE_OAPI_PUNISH = "https://openapi.twse.com.tw/v1/announcement/punish"
TWSE_OAPI_NOTICE = "https://openapi.twse.com.tw/v1/announcement/notice"
TPEX_OAPI_DISPOSAL = "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information"
TPEX_OAPI_ATTENTION = "https://www.tpex.org.tw/openapi/v1/tpex_trading_warning_information"

DEFAULT_TIMEOUT: float = 15.0
USER_AGENT = "Mozilla/5.0 (compatible; stock_bot/1.0)"

DISPOSITION_CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "disposition"

# 專案規範：stock_id 只允許四碼台股（CLAUDE.md）；權證/可轉債直接濾出集合外。
_STOCK_ID_RE = re.compile(r"^\d{4}$")

# 處置期間分隔符：TWSE 用全形波浪（～ U+FF5E），TPEx 用半形（~）；防禦性納入 U+301C。
_PERIOD_SPLIT_RE = re.compile(r"[～〜~]")


def _parse_roc_date_safe(s: Optional[str]) -> Optional[date]:
    """民國日期字串安全解析：失敗回 None（公告欄位偶有缺漏，不可讓整批失敗）。"""
    try:
        return roc_date_to_west(str(s or ""))
    except TWSEError:
        return None


def _parse_period(s: Optional[str]) -> Tuple[Optional[date], Optional[date]]:
    """解析處置期間字串為 (起日, 迄日)。

    支援 TWSE "115/07/03～115/07/16" 與 TPEx "1150710~1150723" 兩種格式。
    無法解析時回 (None, None)，由 _is_active 走保守納入路徑。
    """
    txt = (s or "").strip()
    if not txt:
        return None, None
    parts = [p.strip() for p in _PERIOD_SPLIT_RE.split(txt) if p.strip()]
    if len(parts) != 2:
        return None, None
    start = _parse_roc_date_safe(parts[0])
    end = _parse_roc_date_safe(parts[1])
    if start is None or end is None:
        return None, None
    return start, end


def _fetch_json(url: str, timeout: float) -> Any:
    """單一 endpoint GET + JSON 解析。失敗 raise（由 _fetch_all_records 統一 fail-open）。"""
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────
# Parsers — pure functions（同 twse_client 慣例，方便 fixture 測試）
# ──────────────────────────────────────────────

def _extract_twse_punish(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise TWSEError(f"expected list for TWSE punish, got {type(payload).__name__}")
    out: List[Dict[str, Any]] = []
    for r in payload:
        code = str(r.get("Code") or "").strip()
        if not code:
            continue
        start, end = _parse_period(r.get("DispositionPeriod"))
        announce = _parse_roc_date_safe(r.get("Date"))
        out.append({
            "source": "twse",
            "list_type": "disposition",
            "stock_id": code,
            "name": r.get("Name"),
            "announce_date": announce.isoformat() if announce else None,
            "period_start": start.isoformat() if start else None,
            "period_end": end.isoformat() if end else None,
            "reason": r.get("ReasonsOfDisposition"),
        })
    return out


def _extract_twse_notice(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise TWSEError(f"expected list for TWSE notice, got {type(payload).__name__}")
    out: List[Dict[str, Any]] = []
    for r in payload:
        code = str(r.get("Code") or "").strip()
        if not code:  # 無資料時的 placeholder 列（Number="0", Code=""）
            continue
        announce = _parse_roc_date_safe(r.get("Date"))
        out.append({
            "source": "twse",
            "list_type": "attention",
            "stock_id": code,
            "name": r.get("Name"),
            "announce_date": announce.isoformat() if announce else None,
        })
    return out


def _extract_tpex_disposal(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise TWSEError(f"expected list for TPEx disposal, got {type(payload).__name__}")
    out: List[Dict[str, Any]] = []
    for r in payload:
        code = str(r.get("SecuritiesCompanyCode") or "").strip()
        if not code:
            continue
        start, end = _parse_period(r.get("DispositionPeriod"))
        announce = _parse_roc_date_safe(r.get("Date"))
        out.append({
            "source": "tpex",
            "list_type": "disposition",
            "stock_id": code,
            "name": r.get("CompanyName"),
            "announce_date": announce.isoformat() if announce else None,
            "period_start": start.isoformat() if start else None,
            "period_end": end.isoformat() if end else None,
            "reason": r.get("DispositionReasons"),
        })
    return out


def _extract_tpex_attention(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise TWSEError(f"expected list for TPEx attention, got {type(payload).__name__}")
    out: List[Dict[str, Any]] = []
    for r in payload:
        code = str(r.get("SecuritiesCompanyCode") or "").strip()
        if not code:
            continue
        announce = _parse_roc_date_safe(r.get("Date"))
        out.append({
            "source": "tpex",
            "list_type": "attention",
            "stock_id": code,
            "name": r.get("CompanyName"),
            "announce_date": announce.isoformat() if announce else None,
        })
    return out


_ENDPOINTS: Tuple[Tuple[str, str, Any], ...] = (
    ("twse_punish", TWSE_OAPI_PUNISH, _extract_twse_punish),
    ("twse_notice", TWSE_OAPI_NOTICE, _extract_twse_notice),
    ("tpex_disposal", TPEX_OAPI_DISPOSAL, _extract_tpex_disposal),
    ("tpex_attention", TPEX_OAPI_ATTENTION, _extract_tpex_attention),
)


def _fetch_all_records(timeout: float) -> Tuple[List[Dict[str, Any]], bool]:
    """逐一抓 4 個 endpoint。個別失敗 → logger.warning + 跳過（fail-open），回傳 all_ok 供快取決策。"""
    records: List[Dict[str, Any]] = []
    all_ok = True
    for name, url, extractor in _ENDPOINTS:
        try:
            rows = extractor(_fetch_json(url, timeout))
            records.extend(rows)
            logger.info("disposition_filter: %s 取得 %d 筆", name, len(rows))
        except Exception as exc:
            all_ok = False
            logger.warning(
                "disposition_filter: %s 抓取失敗（fail-open，該來源當日不過濾）：%s",
                name, exc,
            )
    return records, all_ok


def _is_active(record: Dict[str, Any], as_of: date) -> bool:
    """處置公告是否對 as_of 有效：處置迄日 >= as_of（含尚未開始者——明日進場仍會撞到）。

    迄日缺漏/無法解析 → 保守納入（寧可多剔一檔，不可買到分盤股）。
    """
    end_s = record.get("period_end")
    if not end_s:
        return True
    try:
        return date.fromisoformat(str(end_s)) >= as_of
    except ValueError:
        return True


def _build_sets(records: List[Dict[str, Any]], as_of: date) -> Dict[str, Set[str]]:
    disposition: Set[str] = set()
    attention: Set[str] = set()
    for r in records:
        sid = str(r.get("stock_id") or "")
        if not _STOCK_ID_RE.fullmatch(sid):
            continue  # 權證/可轉債等非四碼，永遠不會是候選
        if r.get("list_type") == "disposition":
            if _is_active(r, as_of):
                disposition.add(sid)
        elif r.get("list_type") == "attention":
            attention.add(sid)
    return {"disposition": disposition, "attention": attention}


def _read_cache(cache_path: Path) -> Optional[Dict[str, Set[str]]]:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return {
            "disposition": {str(s) for s in payload["disposition"]},
            "attention": {str(s) for s in payload["attention"]},
        }
    except (ValueError, KeyError, TypeError, OSError) as exc:
        logger.warning("disposition_filter: 快取 %s 損壞（%s），重新抓取", cache_path, exc)
        return None


def _write_cache(
    cache_path: Path,
    as_of: date,
    records: List[Dict[str, Any]],
    result: Dict[str, Set[str]],
) -> None:
    """寫入當日快取。失敗僅留痕不 raise（快取失敗不可影響回傳結果）。"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "as_of": as_of.isoformat(),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "disposition": sorted(result["disposition"]),
            "attention": sorted(result["attention"]),
            # 原始公告（含權證/可轉債、已過期處置）供稽核與日後歷史回補比對
            "records": records,
        }
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("disposition_filter: 快取寫入失敗 %s：%s", cache_path, exc)


def fetch_disposition_lists(
    as_of: Optional[date] = None,
    cache_dir: Optional[Path] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Set[str]]:
    """抓當日 TWSE + TPEx 處置股/注意股名單（含當日快取）。

    回傳 {"disposition": set[stock_id], "attention": set[stock_id]}（僅四碼股票）。
    保證不 raise：任何未預期錯誤 → 空集合 + logger.warning（fail-open，
    daily_pick 依賴此契約，不可讓每日選股掛掉）。
    """
    try:
        as_of = as_of or date.today()
        cache_root = Path(cache_dir) if cache_dir is not None else DISPOSITION_CACHE_DIR
        cache_path = cache_root / f"{as_of.isoformat()}.json"
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

        records, all_ok = _fetch_all_records(timeout)
        result = _build_sets(records, as_of)
        if all_ok:
            _write_cache(cache_path, as_of, records, result)
        else:
            logger.warning(
                "disposition_filter: 部分來源失敗，當日快取不寫入（重跑可重試）；"
                "本次 disposition=%d attention=%d",
                len(result["disposition"]), len(result["attention"]),
            )
        return result
    except Exception as exc:  # 契約：對呼叫端永不 raise
        logger.warning(
            "disposition_filter: 未預期錯誤（fail-open 回空集合）：%s",
            exc, exc_info=True,
        )
        return {"disposition": set(), "attention": set()}
