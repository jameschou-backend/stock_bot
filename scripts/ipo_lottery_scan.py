#!/usr/bin/env python
"""公開申購抽籤掃描（IPO / 現金增資 / 可轉債承銷）。

散戶 +EV 機會掃描：公開申購承銷價通常折價於市價，中籤靠抽籤（與本專案
選股策略零相關）。本工具抓取「進行中 / 即將開始」的公開申購案，計算折價
與期望值，輸出終端表格 + JSON。

資料源（單一 endpoint 覆蓋上市 + 上櫃 + 初上市 / 初上櫃 / 增資 / 轉換公司債）：
    https://www.twse.com.tw/rwd/zh/announcement/publicForm?response=json
    （證交所「公開申購公告-抽籤日程表」，公開申購平台由證交所統一辦理，
     TPEx 上櫃案件亦列於同一公告；openapi.twse.com.tw 對應端點回 302 不可用。）
    每次執行僅發送一個 HTTP request（禮貌抓取，附 UA / timeout）。

計算定義：
    折價%       = 市價 / 承銷價 - 1（市價 = repo raw_prices 最新收盤；DB 無該股 → n/a）
    折價金額     = (市價 - 承銷價) × 申購股數（每筆申購單位，通常 1,000 股）
    期望值       = 折價金額 × 中籤率 - 手續費（預設 20 元，申購處理費，不論中籤與否均收）
    中籤率缺時（未抽籤的案件公告皆為 0）輸出保本門檻：
        保本中籤率   = 手續費 / 折價金額（中籤率高於此值時期望值 > 0）
        保本折價門檻 = 手續費 / (承銷價 × 申購股數)（折價比例高於此值時，
                       單次中籤的折價金額足以覆蓋該筆手續費）

注意：
    - 中籤者實務上另有約 50 元郵匯 / 中籤處理費，未計入預設手續費（可用 --fee 調整）。
    - 中央登錄公債固定排除（無折價套利空間，非任務標的）。
    - 市價查詢僅限四碼台股代號（repo 規範 ^\\d{4}$）；轉換公司債等五碼以上
      代號一律標 n/a（raw_prices 亦無 CB 報價）。
    - JSON 內比率欄位（discount / win_rate / breakeven_*）一律為小數
      （0.25 = 25%），與 --min-discount 同單位。

用法：
    python scripts/ipo_lottery_scan.py                     # 進行中 + 即將開始
    python scripts/ipo_lottery_scan.py --min-discount 0.1  # 只留折價 >= 10%
    python scripts/ipo_lottery_scan.py --all               # 含今年已截止案件（可看實際中籤率）
    python scripts/ipo_lottery_scan.py --no-db             # 不查 DB 市價（離線）
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

logger = logging.getLogger(__name__)

TWSE_PUBLIC_FORM_URL = "https://www.twse.com.tw/rwd/zh/announcement/publicForm"
DEFAULT_FEE_TWD = 20.0
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "ipo_lottery"
REQUEST_TIMEOUT_S = 30.0
USER_AGENT = "stock-bot-ipo-lottery-scan/1.0 (personal research; single polite request)"

# 發行市場含以下關鍵字者排除（非 IPO/現增/可轉債標的）
EXCLUDED_MARKET_KEYWORDS = ("公債",)

# 公告表格必要欄位（schema 變更時 fail loud，禁止 silent fallback）
_REQUIRED_FIELDS = (
    "抽籤日期",
    "證券名稱",
    "證券代號",
    "發行市場",
    "申購開始日",
    "申購結束日",
    "承銷價(元)",
    "實際承銷價(元)",
    "撥券日期(上市、上櫃日期)",
    "主辦券商",
    "申購股數",
    "中籤率(%)",
    "取消公開抽籤",
)

_STOCK_ID_RE = re.compile(r"^\d{4}$")  # repo 規範：市價查詢僅限四碼台股
_MISSING_TOKENS = frozenset({"", "-", "--", "---", "未訂出", "N/A", "n/a"})


# ──────────────────────────────────────────────
# 純函式：解析
# ──────────────────────────────────────────────
def roc_to_date(value: str) -> Optional[date]:
    """民國日期字串轉西元 date，例如 '115/07/23' → date(2026, 7, 23)。

    無法解析（空值 / 佔位符）回傳 None。
    """
    text = (value or "").strip()
    if text in _MISSING_TOKENS:
        return None
    parts = text.split("/")
    if len(parts) != 3:
        return None
    try:
        year, month, day = (int(p) for p in parts)
        return date(year + 1911, month, day)
    except ValueError:
        return None


def parse_decimal(value: str) -> Optional[float]:
    """數字字串轉 float；千分位逗號容忍，'未訂出'/'---'/空值回傳 None。"""
    text = (value or "").strip().replace(",", "")
    if text in _MISSING_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class Announcement:
    """單筆公開申購公告（已正規化）。"""

    stock_id: str
    name: str
    market_type: str
    draw_date: Optional[date]
    sub_start: Optional[date]
    sub_end: Optional[date]
    allotment_date: Optional[date]
    underwriter: str
    underwriting_price: Optional[float]  # 公告承銷價
    actual_price: Optional[float]  # 實際承銷價（未訂出 → None）
    shares_per_unit: Optional[float]  # 每筆申購股數
    win_rate: Optional[float]  # 中籤率（小數；未抽籤公告為 0 → None）
    cancelled: bool

    @property
    def effective_price(self) -> Optional[float]:
        """有效承銷價：實際承銷價優先，未訂出時退回公告承銷價。"""
        return self.actual_price if self.actual_price is not None else self.underwriting_price


def parse_announcements(payload: dict[str, Any]) -> list[Announcement]:
    """解析 TWSE publicForm JSON payload 為 Announcement list。

    只做正規化與排除「公債類」；取消 / 已截止案件保留原樣（由呼叫端過濾），
    方便 --all 模式檢視歷史中籤率。schema 缺欄位時直接 raise（fail loud）。
    """
    fields = [str(f).strip() for f in payload.get("fields", [])]
    idx = {name: i for i, name in enumerate(fields)}
    missing = [f for f in _REQUIRED_FIELDS if f not in idx]
    if missing:
        raise ValueError(f"TWSE publicForm schema 變更，缺少欄位: {missing}（實際欄位: {fields}）")

    def col(row: list[str], name: str) -> str:
        return str(row[idx[name]]).strip()

    announcements: list[Announcement] = []
    for row in payload.get("data", []):
        market_type = col(row, "發行市場")
        if any(kw in market_type for kw in EXCLUDED_MARKET_KEYWORDS):
            continue
        win_rate_pct = parse_decimal(col(row, "中籤率(%)"))
        # 公告在抽籤前一律填 0 → 視為「尚無中籤率」
        win_rate = win_rate_pct / 100.0 if win_rate_pct else None
        announcements.append(
            Announcement(
                stock_id=col(row, "證券代號"),
                name=col(row, "證券名稱"),
                market_type=market_type,
                draw_date=roc_to_date(col(row, "抽籤日期")),
                sub_start=roc_to_date(col(row, "申購開始日")),
                sub_end=roc_to_date(col(row, "申購結束日")),
                allotment_date=roc_to_date(col(row, "撥券日期(上市、上櫃日期)")),
                underwriter=col(row, "主辦券商"),
                underwriting_price=parse_decimal(col(row, "承銷價(元)")),
                actual_price=parse_decimal(col(row, "實際承銷價(元)")),
                shares_per_unit=parse_decimal(col(row, "申購股數")),
                win_rate=win_rate,
                cancelled=bool(col(row, "取消公開抽籤")),
            )
        )
    return announcements


# ──────────────────────────────────────────────
# 純函式：狀態與期望值
# ──────────────────────────────────────────────
def classify_status(ann: Announcement, today: date) -> str:
    """依申購期間分類：ongoing（申購中）/ upcoming（即將開始）/ closed（已截止）。"""
    if ann.sub_start is None or ann.sub_end is None:
        return "closed"
    if today < ann.sub_start:
        return "upcoming"
    if today > ann.sub_end:
        return "closed"
    return "ongoing"


@dataclass
class Metrics:
    """單筆申購案的折價 / 期望值計算結果（比率欄位皆為小數）。"""

    discount: Optional[float]  # 市價/承銷價 - 1
    discount_amount: Optional[float]  # (市價-承銷價) × 申購股數（元）
    expected_value: Optional[float]  # 折價金額 × 中籤率 - 手續費（元）
    breakeven_win_rate: Optional[float]  # 手續費 / 折價金額
    breakeven_discount: Optional[float]  # 手續費 / (承銷價 × 申購股數)


def compute_metrics(
    underwriting_price: Optional[float],
    market_price: Optional[float],
    shares_per_unit: Optional[float],
    win_rate: Optional[float],
    fee: float = DEFAULT_FEE_TWD,
) -> Metrics:
    """計算折價與期望值。缺輸入的欄位回傳 None（不猜、不 fallback）。

    期望值 = 折價金額 × 中籤率 - 手續費（手續費不論中籤與否均支付）。
    中籤率缺時期望值為 None，改看保本門檻：
      - breakeven_win_rate：中籤率高於此值時期望值轉正（僅折價 > 0 時有意義）
      - breakeven_discount：折價比例高於此值時，單次中籤的折價金額 ≥ 手續費
    """
    if fee < 0:
        raise ValueError(f"fee 不可為負: {fee}")

    price_ok = underwriting_price is not None and underwriting_price > 0
    discount: Optional[float] = None
    discount_amount: Optional[float] = None
    if price_ok and market_price is not None:
        discount = market_price / underwriting_price - 1.0
        if shares_per_unit is not None and shares_per_unit > 0:
            discount_amount = (market_price - underwriting_price) * shares_per_unit

    expected_value: Optional[float] = None
    if discount_amount is not None and win_rate is not None:
        expected_value = discount_amount * win_rate - fee

    breakeven_win_rate: Optional[float] = None
    if discount_amount is not None and discount_amount > 0:
        breakeven_win_rate = fee / discount_amount

    breakeven_discount: Optional[float] = None
    if price_ok and shares_per_unit is not None and shares_per_unit > 0:
        breakeven_discount = fee / (underwriting_price * shares_per_unit)

    return Metrics(
        discount=discount,
        discount_amount=discount_amount,
        expected_value=expected_value,
        breakeven_win_rate=breakeven_win_rate,
        breakeven_discount=breakeven_discount,
    )


def build_scan_items(
    announcements: list[Announcement],
    market_prices: dict[str, tuple[float, date]],
    today: date,
    fee: float = DEFAULT_FEE_TWD,
    include_closed: bool = False,
    min_discount: Optional[float] = None,
) -> list[dict[str, Any]]:
    """組合公告 + 市價 → 掃描結果項目（dict，供表格與 JSON 輸出）。

    - 取消案件一律排除。
    - 預設只留 ongoing / upcoming；include_closed=True 時含已截止（看實際中籤率）。
    - min_discount 過濾時，無市價（折價未知）的案件一併排除（無法證明達標）。
    """
    items: list[dict[str, Any]] = []
    for ann in announcements:
        if ann.cancelled:
            continue
        status = classify_status(ann, today)
        if status == "closed" and not include_closed:
            continue

        price_info = market_prices.get(ann.stock_id)
        market_price, market_price_date = price_info if price_info else (None, None)
        metrics = compute_metrics(
            underwriting_price=ann.effective_price,
            market_price=market_price,
            shares_per_unit=ann.shares_per_unit,
            win_rate=ann.win_rate,
            fee=fee,
        )
        if min_discount is not None and (metrics.discount is None or metrics.discount < min_discount):
            continue

        items.append(
            {
                "stock_id": ann.stock_id,
                "name": ann.name,
                "market_type": ann.market_type,
                "status": status,
                "draw_date": ann.draw_date.isoformat() if ann.draw_date else None,
                "sub_start": ann.sub_start.isoformat() if ann.sub_start else None,
                "sub_end": ann.sub_end.isoformat() if ann.sub_end else None,
                "allotment_date": ann.allotment_date.isoformat() if ann.allotment_date else None,
                "underwriter": ann.underwriter,
                "underwriting_price": ann.underwriting_price,
                "actual_price": ann.actual_price,
                "effective_price": ann.effective_price,
                "shares_per_unit": ann.shares_per_unit,
                "market_price": market_price,
                "market_price_date": market_price_date.isoformat() if market_price_date else None,
                "discount": _round(metrics.discount, 6),
                "discount_amount": _round(metrics.discount_amount, 2),
                "win_rate": _round(ann.win_rate, 6),
                "expected_value_twd": _round(metrics.expected_value, 2),
                "breakeven_win_rate": _round(metrics.breakeven_win_rate, 6),
                "breakeven_discount": _round(metrics.breakeven_discount, 6),
            }
        )
    # 折價高的排前面（None 排最後），其次依申購結束日
    items.sort(
        key=lambda it: (
            -(it["discount"] if it["discount"] is not None else float("-inf")),
            it["sub_end"] or "9999-12-31",
        )
    )
    return items


def _round(value: Optional[float], ndigits: int) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


# ──────────────────────────────────────────────
# I/O：TWSE 抓取與 DB 市價
# ──────────────────────────────────────────────
def fetch_public_form(url: str = TWSE_PUBLIC_FORM_URL, timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """抓取 TWSE 公開申購公告 JSON（單一 request，禮貌抓取）。"""
    resp = requests.get(
        url,
        params={"response": "json"},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("stat") != "OK":
        raise RuntimeError(f"TWSE publicForm 回應異常: stat={payload.get('stat')!r}")
    return payload


def load_market_prices(stock_ids: list[str]) -> dict[str, tuple[float, date]]:
    """從 raw_prices 讀每檔最新收盤價。回傳 {stock_id: (close, trading_date)}。

    僅查四碼台股代號（repo 規範 ^\\d{4}$）；轉換公司債等其他代號直接略過
    （呼叫端顯示 n/a）。DB 連線失敗會直接 raise（fail loud，離線請用 --no-db）。
    """
    valid_ids = [sid for sid in stock_ids if _STOCK_ID_RE.match(sid)]
    if not valid_ids:
        return {}

    from sqlalchemy import func, select

    from app.db import get_session
    from app.models import RawPrice

    with get_session() as session:
        latest = (
            select(RawPrice.stock_id, func.max(RawPrice.trading_date).label("max_date"))
            .where(RawPrice.stock_id.in_(valid_ids))
            .group_by(RawPrice.stock_id)
            .subquery()
        )
        rows = session.execute(
            select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close).join(
                latest,
                (RawPrice.stock_id == latest.c.stock_id)
                & (RawPrice.trading_date == latest.c.max_date),
            )
        ).all()
    return {sid: (float(close), tdate) for sid, tdate, close in rows if close is not None}


# ──────────────────────────────────────────────
# 輸出：終端表格 + JSON
# ──────────────────────────────────────────────
_STATUS_LABEL = {"ongoing": "申購中", "upcoming": "即將開始", "closed": "已截止"}


def _display_width(text: str) -> int:
    """終端顯示寬度（CJK 全形字算 2）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:+.2f}%" if value is not None else "n/a"


def _fmt_num(value: Optional[float], fmt: str = ",.2f") -> str:
    return format(value, fmt) if value is not None else "n/a"


def format_table(items: list[dict[str, Any]]) -> str:
    """把掃描結果排成終端表格（CJK 對齊）。"""
    if not items:
        return "（無符合條件的公開申購案件）"
    headers = ["代號", "名稱", "市場", "申購期間", "狀態", "承銷價", "市價", "折價", "中籤率", "期望值(元)", "保本中籤率"]
    rows: list[list[str]] = []
    for it in items:
        period = f"{it['sub_start'] or '?'} ~ {it['sub_end'] or '?'}"
        rows.append(
            [
                it["stock_id"],
                it["name"],
                it["market_type"],
                period,
                _STATUS_LABEL.get(it["status"], it["status"]),
                _fmt_num(it["effective_price"]),
                _fmt_num(it["market_price"]),
                _fmt_pct(it["discount"]),
                _fmt_pct(it["win_rate"]) if it["win_rate"] is not None else "未抽",
                _fmt_num(it["expected_value_twd"]),
                _fmt_pct(it["breakeven_win_rate"]).lstrip("+") if it["breakeven_win_rate"] is not None else "n/a",
            ]
        )
    widths = [
        max(_display_width(headers[i]), *(_display_width(r[i]) for r in rows)) for i in range(len(headers))
    ]
    sep = "  "
    lines = [
        sep.join(_pad(headers[i], widths[i]) for i in range(len(headers))),
        sep.join("-" * w for w in widths),
    ]
    lines.extend(sep.join(_pad(r[i], widths[i]) for i in range(len(headers))) for r in rows)
    return "\n".join(lines)


def write_json(
    items: list[dict[str, Any]],
    scan_date: date,
    fee: float,
    min_discount: Optional[float],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """輸出 artifacts/ipo_lottery/scan_YYYY-MM-DD.json。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"scan_{scan_date.isoformat()}.json"
    payload = {
        "scan_date": scan_date.isoformat(),
        "source": TWSE_PUBLIC_FORM_URL,
        "fee_twd": fee,
        "min_discount": min_discount,
        "count": len(items),
        "note": "比率欄位（discount/win_rate/breakeven_*）皆為小數（0.1=10%）；"
        "expected_value = discount_amount × win_rate − fee；中籤者另有約 50 元郵匯費未計入",
        "items": items,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="公開申購抽籤掃描（IPO/現增/可轉債）")
    parser.add_argument("--min-discount", type=float, default=None, help="最低折價過濾（小數，0.1 = 10%%）")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE_TWD, help=f"申購手續費（元，預設 {DEFAULT_FEE_TWD:.0f}）")
    parser.add_argument("--all", action="store_true", help="含今年已截止案件（可檢視實際中籤率）")
    parser.add_argument("--no-db", action="store_true", help="不查 DB 市價（離線模式，折價全標 n/a）")
    parser.add_argument("--today", type=date.fromisoformat, default=None, help="覆蓋今天日期（YYYY-MM-DD，測試/重放用）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="JSON 輸出目錄")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = args.today or date.today()

    payload = fetch_public_form()
    announcements = parse_announcements(payload)
    logger.info("TWSE 公開申購公告：%d 筆（已排除公債類）", len(announcements))

    active = [a for a in announcements if not a.cancelled and (args.all or classify_status(a, today) != "closed")]
    market_prices: dict[str, tuple[float, date]] = {}
    if args.no_db:
        logger.info("--no-db：跳過市價查詢，折價/期望值標 n/a")
    else:
        market_prices = load_market_prices(sorted({a.stock_id for a in active}))
        logger.info("DB 市價命中 %d / %d 檔", len(market_prices), len({a.stock_id for a in active}))

    items = build_scan_items(
        announcements,
        market_prices,
        today=today,
        fee=args.fee,
        include_closed=args.all,
        min_discount=args.min_discount,
    )

    print(f"\n公開申購抽籤掃描  scan_date={today.isoformat()}  fee={args.fee:.0f} 元"
          + (f"  min_discount={args.min_discount:.1%}" if args.min_discount is not None else ""))
    print(format_table(items))
    print(
        "\n說明：折價 = 市價/承銷價 - 1；期望值 = 折價金額×中籤率 − 手續費；"
        "未抽籤案件看「保本中籤率」（中籤率高於它即 +EV）。市價 = raw_prices 最新收盤（未還原）。"
    )

    out_path = write_json(items, today, fee=args.fee, min_discount=args.min_discount, output_dir=args.output_dir)
    print(f"JSON 已輸出：{out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
