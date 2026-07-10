#!/usr/bin/env python
"""訊號價版 paper NAV — forward track record（2026-07-10 總體檢 P0 項）。

模擬「完全跟隨 A 線生產訊號」的紙上淨值：

- 持倉口徑：月頻換股 — 每月**第一個 pick 日**的 picks（daily_pick topN）＝當月持倉，
  等權，於該日收盤價「進場」（訊號價；與回測 ``entry_delay_days=0`` 同口徑）。
  同月其後每日的 picks 只作記錄、不改變持倉。
- 每日 mark-to-market：``raw_prices`` 收盤價（等權買入後**持股數固定**，
  月內權重自然漂移，非每日重設等權）。
- 輸出：``artifacts/paper_nav/nav.jsonl``，每交易日一行：
  ``{"date", "nav", "holdings_n", "config_version", "notes"}``

⚠️ 誠實聲明（明文偏差方向；每行 notes 欄同步標注）：

1. **收盤價未還原**（raw close）：post-sponsor adj factor 凍結（快照只到 2026-06-23），
   持股逢除權息時本 NAV 會把除息缺口記成「假跌」→ 本紀錄**系統性低估**
   實際策略報酬（偏保守方向，非樂觀方向）。
2. 無交易成本、無滑價（訊號價＝收盤價成交）；實際成交價落差另由月頻人工回填
   記錄（缺就缺），不在本檔範圍。
3. 停牌／缺價日沿用最後收盤價 mark-to-market；下市股會停在最後一筆價
   （此偏差方向為高估，但月頻換股下暴露 ≤1 個月）。
4. **fallback 時間戳偏差**：daily_pick 的 fallback 機制（fallback_days 預設 10）
   會把 pick_date 蓋成比實際執行日早最多 10 個交易日的特徵日——該情況下本 NAV
   宣稱的進場時點早於訊號實際存在時點（方向為樂觀）。凡月首 pick 日帶
   fallback_days > 0，該再平衡行 notes 會標注 ``⚠fallback`` 並印警告
   （讀 picks.reason_json._selection_meta.fallback_days；舊 picks 無此欄位則
   無法判定、不標注）。

版本標記（總體檢缺陷 6 規則 2）：

- ``config_version`` 讀 env ``PAPER_NAV_CONFIG_VERSION``（預設 ``v2-20260703``）。
- 生產配置每次被觸碰（ex-date 止血、處置股 filter、adj factor 修復……）須**人工 bump**
  此版本字串，使 forward 紀錄可按配置切段；否則 12-24 個月後的 live 資料是
  多個配置的拼接，什麼都檢定不了。

冪等性（同日重跑覆蓋當日行）：

- 每次執行以 picks + raw_prices 重放完整序列（確定性），但**凍結歷史行**：
  日期早於檔案最後一行者原樣保留（含其當時寫入的 config_version / notes）；
  檔案最後一行當日及其後的日期以本次計算覆蓋／追加。
  → 同日重跑＝覆蓋當日行；漏跑數日＝下次執行自動補齊。
- **config_version 一律 first-write**：檔案最後一行被本次重放覆蓋時，nav/notes
  用新值（同日重跑語意），但 config_version 保留該行首次寫入的值——該行的持倉
  是舊配置產生的，bump ``PAPER_NAV_CONFIG_VERSION`` 不得回溯改標配置段界。
- 若重放值與凍結行漂移（相對差 > 1e-6），代表上游 DB 被回溯修改：印警告、
  保留凍結值（forward 紀錄以first-write為準）。
- **--start 鎖定**：nav.jsonl 已存在時，本次重放的錨定行（第一行）日期必須與
  檔案第一行一致，否則直接報錯拒跑——不同 --start 會以新起點重新錨定 nav=1.0，
  靜默寫入等於把 forward 紀錄拼接成兩個 NAV 基底。要改起算日請刪檔重建。

預設起算日 2026-07-03：健檢修復日。此前 picks 有 P0-1 index 錯位（2/13 起
picks 用別檔股票的特徵打分），不可入 forward 紀錄。

用法：
    python scripts/paper_nav.py                      # 增量更新到最新價格日
    python scripts/paper_nav.py --start 2026-07-03   # 指定重放起算日
    PAPER_NAV_CONFIG_VERSION=v3-exdate-patch python scripts/paper_nav.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_left
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

DEFAULT_CONFIG_VERSION = "v2-20260703"
DEFAULT_START = date(2026, 7, 3)  # 健檢修復日；此前 picks 有 index 錯位，不可入 forward 紀錄
NAV_PATH = ROOT / "artifacts" / "paper_nav" / "nav.jsonl"

# 每行 notes 欄固定前綴：明文標注偏差方向（收盤價未還原 → 系統性低估策略）
BASE_NOTE = "raw_close未還原;凍結adj期間除息記假跌→系統性低估;無成本無滑價"

# 凍結行 vs 重放值的漂移容忍（相對差）；超過代表上游 DB 被回溯修改
DRIFT_RTOL = 1e-6


def resolve_config_version() -> str:
    """讀 env PAPER_NAV_CONFIG_VERSION；生產配置每次被觸碰時人工 bump（缺陷 6 規則 2）。"""
    return os.environ.get("PAPER_NAV_CONFIG_VERSION", DEFAULT_CONFIG_VERSION)


# ──────────────────────────────────────────────────────────
# 核心計算（純函式，可測試）
# ──────────────────────────────────────────────────────────

def monthly_rebalance_dates(pick_dates: List[date]) -> List[date]:
    """每月第一個 pick 日 = 該月再平衡日（持倉 = 該日 picks）。"""
    firsts: Dict[Tuple[int, int], date] = {}
    for d in pick_dates:
        key = (d.year, d.month)
        if key not in firsts or d < firsts[key]:
            firsts[key] = d
    return sorted(firsts.values())


def compute_nav_series(
    picks: pd.DataFrame,
    prices: pd.DataFrame,
    start: date,
    config_version: str,
    fallback_by_date: Dict[date, int] | None = None,
) -> List[dict]:
    """重放訊號價 paper NAV 序列（確定性純函式）。

    Args:
        picks: columns [pick_date(date), stock_id(str)]，daily_pick 每日輸出。
        prices: columns [trading_date(date), stock_id(str), close(float)]，raw 收盤價（未還原）。
        start: 重放起算日（含）；取第一個 >= start 的月度再平衡日為 NAV=1.0 錨點。
        config_version: 寫入每行的配置版本標記。
        fallback_by_date: pick_date → daily_pick 當次的 fallback_days（自
            reason_json._selection_meta 讀出）。> 0 表示該 pick_date 是回退的
            特徵日、早於實際執行日——再平衡行 notes 標注 ``⚠fallback``
            （誠實聲明 4：時間戳偏樂觀）。None/缺鍵 = 無法判定，不標注。

    Returns:
        每交易日一行 [{date, nav, holdings_n, config_version, notes}, ...]，按日期排序。
        持倉模型：再平衡日以當日收盤等權買入 → 持股數固定 → 每日 sum(shares*close)
        mark-to-market（月內權重自然漂移）。缺價日沿用最後收盤（ffill）。
    """
    if picks.empty or prices.empty:
        return []

    picks = picks.copy()
    picks["pick_date"] = pd.to_datetime(picks["pick_date"]).dt.date
    picks["stock_id"] = picks["stock_id"].astype(str)
    picks = picks[picks["pick_date"] >= start]
    if picks.empty:
        return []

    rb_dates = monthly_rebalance_dates(sorted(picks["pick_date"].unique()))

    prices = prices.copy()
    prices["trading_date"] = pd.to_datetime(prices["trading_date"]).dt.date
    prices["stock_id"] = prices["stock_id"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    # pivot: index=trading_date, columns=stock_id；ffill = 停牌/缺價沿用最後收盤價
    pivot = prices.pivot_table(index="trading_date", columns="stock_id", values="close")
    pivot.sort_index(inplace=True)
    pivot_ffill = pivot.ffill()

    first_rb = rb_dates[0]
    trading_dates = [d for d in pivot.index if d >= first_rb]
    if not trading_dates:
        return []

    # 再平衡 pick 日 → 生效交易日（第一個 >= pick 日的價格日；正常情況兩者相同）
    rb_effective: Dict[date, date] = {}
    for rb in rb_dates:
        idx = bisect_left(trading_dates, rb)
        if idx < len(trading_dates):
            rb_effective[trading_dates[idx]] = rb

    picks_by_date: Dict[date, List[str]] = (
        picks.groupby("pick_date")["stock_id"].apply(lambda s: sorted(s.unique())).to_dict()
    )

    rows: List[dict] = []
    shares: Dict[str, float] = {}  # stock_id -> 持股數（再平衡日固定，月內不變）
    nav = 1.0

    for d in trading_dates:
        notes = [BASE_NOTE]

        # 1) 先 mark 舊持倉到今日收盤（再平衡日：舊倉持有至當日收盤才換股）
        if shares:
            nav = sum(q * float(pivot_ffill.at[d, sid]) for sid, q in shares.items())
            stale = [sid for sid in shares if pd.isna(pivot.at[d, sid])]
            if stale:
                notes.append(f"{len(stale)}檔缺當日價沿用前收:{','.join(stale)}")

        # 2) 再平衡日：以今日收盤等權換入本月持倉（訊號價進場）
        if d in rb_effective:
            pick_day = rb_effective[d]
            candidates = picks_by_date[pick_day]
            valid = [
                sid
                for sid in candidates
                if sid in pivot_ffill.columns and pd.notna(pivot_ffill.at[d, sid])
            ]
            skipped = [sid for sid in candidates if sid not in valid]
            if valid:
                per_stock = nav / len(valid)
                shares = {sid: per_stock / float(pivot_ffill.at[d, sid]) for sid in valid}
                notes.append(f"rebalance:{len(valid)}檔等權進場(picks@{pick_day.isoformat()})")
                if skipped:
                    notes.append(f"進場略過無價:{','.join(skipped)}")
            else:
                notes.append(f"rebalance失敗:picks@{pick_day.isoformat()}全數無價,沿用舊倉")
            # 誠實聲明 4：pick_date 為 fallback 回退的特徵日 → 進場時間戳早於
            # 訊號實際存在時點（偏樂觀），明文標注
            _fd = int((fallback_by_date or {}).get(pick_day) or 0)
            if _fd > 0:
                notes.append(
                    f"⚠fallback:訊號日{pick_day.isoformat()}為回退{_fd}個交易日的特徵日,"
                    "實際執行晚於此時間戳(偏樂觀)"
                )

        rows.append(
            {
                "date": d.isoformat(),
                "nav": round(float(nav), 8),
                "holdings_n": len(shares),
                "config_version": config_version,
                "notes": ";".join(notes),
            }
        )

    return rows


def merge_rows(existing: List[dict], new_rows: List[dict]) -> Tuple[List[dict], List[str]]:
    """冪等合併：凍結歷史行，覆蓋最後一行及其後（同日重跑 → 覆蓋當日行）。

    - 日期 < 既有檔最後一行者：原樣保留（含 first-write 的 config_version / notes）。
    - 日期 >= 既有檔最後一行者：以本次重放覆蓋／追加。
      **例外**：被覆蓋的既有行（= cutoff 當日行）config_version 保留 first-write
      值——該行的持倉/NAV 是當時配置產生的，bump PAPER_NAV_CONFIG_VERSION 後
      首次執行不得把前一交易日誤標成新版本（配置切段以 first-write 為準）。
    - 凍結行 vs 重放值相對差 > DRIFT_RTOL → 產生警告（上游 DB 被回溯修改的訊號），
      但仍保留凍結值。

    Returns:
        (合併後按日期排序的 rows, 漂移警告清單)
    """
    warnings: List[str] = []
    if not existing:
        return sorted(new_rows, key=lambda r: r["date"]), warnings

    cutoff = max(r["date"] for r in existing)
    new_by_date = {r["date"]: r for r in new_rows}

    merged: Dict[str, dict] = dict(new_by_date)
    for row in existing:
        if row["date"] < cutoff or row["date"] not in merged:
            # 凍結歷史行（或重放範圍未覆蓋的行）
            merged[row["date"]] = row
            replay = new_by_date.get(row["date"])
            if replay is not None:
                old_nav, new_nav = float(row["nav"]), float(replay["nav"])
                denom = max(abs(old_nav), 1e-12)
                if abs(new_nav - old_nav) / denom > DRIFT_RTOL:
                    warnings.append(
                        f"[drift] {row['date']}: 凍結 nav={old_nav} vs 重放 nav={new_nav}"
                        "（上游 DB 疑被回溯修改；保留凍結值）"
                    )
        else:
            # cutoff 當日行被本次重放覆蓋：nav/notes 用新值（同日重跑語意），
            # 但 config_version 維持 first-write（版本標籤不可回溯改寫）
            _old_cv = row.get("config_version")
            if _old_cv is not None and merged[row["date"]].get("config_version") != _old_cv:
                merged[row["date"]] = {**merged[row["date"]], "config_version": _old_cv}

    return [merged[d] for d in sorted(merged)], warnings


# ──────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────

def load_nav_file(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_nav_file(path: Path, rows: List[dict]) -> None:
    """原子寫入（tmp + os.replace），避免中斷留下半行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_picks_from_db(start: date) -> pd.DataFrame:
    from sqlalchemy import select

    from app.db import get_session
    from app.models import Pick

    with get_session() as session:
        rows = session.execute(
            select(Pick.pick_date, Pick.stock_id).where(Pick.pick_date >= start)
        ).fetchall()
    return pd.DataFrame(rows, columns=["pick_date", "stock_id"])


def extract_fallback_days(reason_json: object) -> int | None:
    """從 Pick.reason_json 讀 _selection_meta.fallback_days（缺欄位 → None=無法判定）。"""
    payload = reason_json
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get("_selection_meta")
    if not isinstance(meta, dict):
        return None
    fd = meta.get("fallback_days")
    try:
        return int(fd) if fd is not None else None
    except (TypeError, ValueError):
        return None


def load_fallback_meta_from_db(start: date) -> Dict[date, int]:
    """每個 pick_date 的 fallback_days（誠實聲明 4 用；同日各列 meta 相同，取最大保守）。"""
    from sqlalchemy import select

    from app.db import get_session
    from app.models import Pick

    with get_session() as session:
        rows = session.execute(
            select(Pick.pick_date, Pick.reason_json).where(Pick.pick_date >= start)
        ).fetchall()

    out: Dict[date, int] = {}
    for pick_date, reason_json in rows:
        fd = extract_fallback_days(reason_json)
        if fd is not None:
            out[pick_date] = max(out.get(pick_date, 0), fd)
    return out


def load_prices_from_db(stock_ids: List[str], start: date) -> pd.DataFrame:
    from sqlalchemy import select

    from app.db import get_session
    from app.models import RawPrice

    with get_session() as session:
        rows = session.execute(
            select(RawPrice.trading_date, RawPrice.stock_id, RawPrice.close).where(
                RawPrice.stock_id.in_(stock_ids),
                RawPrice.trading_date >= start,
            )
        ).fetchall()
    return pd.DataFrame(rows, columns=["trading_date", "stock_id", "close"])


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="訊號價版 paper NAV（forward track record）")
    parser.add_argument(
        "--start",
        default=DEFAULT_START.isoformat(),
        help=f"重放起算日 YYYY-MM-DD（預設 {DEFAULT_START.isoformat()}＝健檢修復日，此前 picks 不可信）",
    )
    parser.add_argument(
        "--nav-path",
        default=str(NAV_PATH),
        help="nav.jsonl 輸出路徑（預設 artifacts/paper_nav/nav.jsonl）",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    nav_path = Path(args.nav_path)
    config_version = resolve_config_version()

    picks = load_picks_from_db(start)
    if picks.empty:
        print(f"[paper_nav] picks 表自 {start} 起無資料，本次不寫入。")
        return 0

    stock_ids = sorted(picks["stock_id"].astype(str).unique())
    prices = load_prices_from_db(stock_ids, start)
    if prices.empty:
        print(f"[paper_nav] raw_prices 自 {start} 起無持倉股價格，本次不寫入。")
        return 0

    fallback_by_date = load_fallback_meta_from_db(start)
    new_rows = compute_nav_series(picks, prices, start, config_version, fallback_by_date)
    if not new_rows:
        print("[paper_nav] 重放結果為空（無再平衡日或無交易日），本次不寫入。")
        return 0

    existing = load_nav_file(nav_path)

    # ── NAV 基底鎖定：重放錨定行必須與既有檔第一行同日 ──
    # 不同 --start 會以新起點重新錨定 nav=1.0；若靜默合併，檔案在接縫處變成
    # 兩個 NAV 基底的拼接（假跌/假漲），下游 forward 報酬/Sharpe 全數失真。
    if existing and new_rows[0]["date"] != existing[0]["date"]:
        print(
            f"[paper_nav][ERROR] 重放錨定日 {new_rows[0]['date']}（--start {start.isoformat()}）"
            f"與既有檔第一行 {existing[0]['date']} 不一致——拒絕寫入。\n"
            f"  不同起算日會重新錨定 nav=1.0，合併將把 forward 紀錄拼接成兩個 NAV 基底。\n"
            f"  請改用與既有檔一致的 --start，或刪除 {nav_path} 後重建整個序列。"
        )
        return 2

    merged, warnings = merge_rows(existing, new_rows)
    write_nav_file(nav_path, merged)

    for w in warnings:
        print(f"[paper_nav][WARN] {w}")
    _fb_hits = sorted(
        {r["date"] for r in new_rows if "⚠fallback" in r.get("notes", "")}
    )
    for d in _fb_hits:
        print(f"[paper_nav][WARN] {d}: 月首 pick 為 fallback 特徵日，"
              "進場時間戳早於訊號實際存在時點（見誠實聲明 4）")

    last = merged[-1]
    n_new = len(merged) - max(len(existing) - 1, 0)  # 覆蓋最後一行 + 追加行
    print(
        f"[paper_nav] 寫入 {nav_path}（共 {len(merged)} 行，本次更新 {max(n_new, 1)} 行）；"
        f"最新 {last['date']} nav={last['nav']} holdings={last['holdings_n']} "
        f"config={last['config_version']}"
    )
    print(f"[paper_nav] 提醒：{BASE_NOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
