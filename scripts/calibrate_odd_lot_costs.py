"""零股（odd-lot）交易成本實證校準。

資料源（皆為官方免費 API，盤中零股每日彙總行情）：
  - TWSE 盤中零股交易行情單（TWTC7U）：
    https://www.twse.com.tw/rwd/zh/afterTrading/TWTC7U?date=YYYYMMDD&response=json
    欄位：證券代號/名稱/成交股數/成交筆數/成交金額/當日第一次成交價/當日最後一次成交價/
          當日最高價/當日最低價/最後揭示買價/量/最後揭示賣價/量
  - TPEx 盤中零股每日收盤行情（oddQuote）：
    https://www.tpex.org.tw/www/zh-tw/afterTrading/oddQuote?d=ROC_DATE&response=json
    欄位：代號/名稱/最後成交價/漲跌/首筆成交價/最高/最低/成交股數/成交金額/成交筆數/
          最後買價/量/最後賣價/量

校準邏輯（詳見 skills/odd_lot_costs.py docstring 與 docs/prereg_odd_lot_arm_20260711.md）：
  1. 抓樣本窗（預設近 12 個月）兩市場盤中零股日行情，只留四碼普通股（stocks.security_type='stock'）。
  2. 與 DB raw_prices 整股收盤價 join；amt_20 = 整股 close×volume 的 20 日滾動均值（與回測
     tiered slippage 的 amt_20 同語義：日均成交金額）。
  3. 每檔-每日計算三個量：
     - half_spread_rel = (最後賣價-最後買價)/2/mid：收盤零股報價半價差（單邊立即成交成本的直接量測）
     - vwap_dev = 零股VWAP/整股close - 1：日內時間差混入的偏離（分佈統計用）
     - basis = 零股最後成交價/整股close - 1：兩簿收盤價基差（分佈統計用）
  4. 按 amt_20 固定門檻分層（0.1/0.3/1/5 億），輸出各層 premium 統計與模型採用值
     （per-side premium = 該層 half_spread_rel 的 P75，偏保守；悲觀臂另 ×1.5）。

輸出：
  - artifacts/odd_lot/odd_lot_daily.parquet（正規化的原始樣本，可續跑/重算）
  - artifacts/odd_lot/calibration.json（分層 premium 表 + 全部統計 + 樣本 metadata）

用法：
  python scripts/calibrate_odd_lot_costs.py --start 2025-07-01 --end 2026-06-30
  python scripts/calibrate_odd_lot_costs.py --calibrate-only   # 已有 parquet 時只重算統計
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import get_session  # noqa: E402
from sqlalchemy import text  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "odd_lot"
PARQUET_PATH = OUT_DIR / "odd_lot_daily.parquet"
CALIBRATION_PATH = OUT_DIR / "calibration.json"

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/TWTC7U"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/oddQuote"
RATE_LIMIT_SECONDS = 1.5
FOUR_DIGIT = re.compile(r"^\d{4}$")

# 流動性分層門檻（元；與 backtest tiered slippage 的 amt_20 同語義）。
# 個人口徑臂會交易 <1 億的微型股，故在低流動性端加密分層。
TIER_BOUNDS = [1e7, 3e7, 1e8, 5e8]  # 0.1 / 0.3 / 1 / 5 億
TIER_LABELS = ["lt_0.1yi", "0.1_0.3yi", "0.3_1yi", "1_5yi", "ge_5yi"]


def _to_float(s: str) -> float | None:
    s = str(s).replace(",", "").strip()
    if s in ("", "-", "--", "0.00", "None"):
        # 0.00 在價格欄位視為無效（無成交/無揭示）；數量欄位另行處理
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str) -> int:
    s = str(s).replace(",", "").strip()
    if s in ("", "-", "--"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def fetch_twse(d: date, session: requests.Session) -> list[dict]:
    """TWSE 盤中零股交易行情單（TWTC7U）單日。"""
    params = {"date": d.strftime("%Y%m%d"), "response": "json"}
    r = session.get(TWSE_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("stat") != "OK" or not j.get("data"):
        return []
    rows = []
    for row in j["data"]:
        sid = str(row[0]).strip()
        if not FOUR_DIGIT.match(sid):
            continue
        rows.append({
            "trading_date": d,
            "stock_id": sid,
            "market": "TWSE",
            "odd_shares": _to_int(row[2]),
            "odd_n_tx": _to_int(row[3]),
            "odd_amount": _to_int(row[4]),
            "odd_last": _to_float(row[6]),
            "odd_bid": _to_float(row[9]),
            "odd_ask": _to_float(row[11]),
        })
    return rows


def fetch_tpex(d: date, session: requests.Session) -> list[dict]:
    """TPEx 盤中零股每日收盤行情（oddQuote）單日。"""
    roc = f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
    params = {"d": roc, "response": "json"}
    r = session.get(TPEX_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    tables = j.get("tables") or []
    if not tables or not tables[0].get("data"):
        return []
    rows = []
    for row in tables[0]["data"]:
        sid = str(row[0]).strip()
        if not FOUR_DIGIT.match(sid):
            continue
        rows.append({
            "trading_date": d,
            "stock_id": sid,
            "market": "TPEX",
            "odd_shares": _to_int(row[7]),
            "odd_n_tx": _to_int(row[9]),
            "odd_amount": _to_int(row[8]),
            "odd_last": _to_float(row[2]),
            "odd_bid": _to_float(row[10]),
            "odd_ask": _to_float(row[12]),
        })
    return rows


def load_trading_dates(start: date, end: date) -> list[date]:
    with get_session() as s:
        rows = s.execute(
            text(
                "SELECT DISTINCT trading_date FROM raw_prices "
                "WHERE trading_date BETWEEN :a AND :b ORDER BY trading_date"
            ),
            {"a": start, "b": end},
        ).fetchall()
    return [r[0] for r in rows]


def load_common_stock_ids() -> set[str]:
    """上市/上櫃普通股（排除興櫃/ETF/權證），與生產 universe 同口徑。"""
    with get_session() as s:
        rows = s.execute(
            text(
                "SELECT stock_id FROM stocks "
                "WHERE security_type='stock' AND market IN ('TWSE','TPEX')"
            )
        ).fetchall()
    return {r[0] for r in rows if FOUR_DIGIT.match(str(r[0]))}


def fetch_sample(start: date, end: date) -> pd.DataFrame:
    """抓樣本窗兩市場零股日行情（可續跑：已在 parquet 的日期直接跳過）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = load_trading_dates(start, end)
    existing = pd.DataFrame()
    done_dates: set = set()
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        done_dates = set(pd.to_datetime(existing["trading_date"]).dt.date.unique())
        print(f"[resume] 既有 parquet {len(existing)} 列 / {len(done_dates)} 個日期")

    todo = [d for d in dates if d not in done_dates]
    print(f"[fetch] 樣本窗 {start} ~ {end}：{len(dates)} 個交易日，待抓 {len(todo)} 日")
    http = requests.Session()
    http.headers["User-Agent"] = "Mozilla/5.0 (odd-lot cost calibration; stock_bot research)"

    buf: list[dict] = []
    for i, d in enumerate(todo, 1):
        for fetcher, name in ((fetch_twse, "TWSE"), (fetch_tpex, "TPEX")):
            for attempt in range(3):
                try:
                    buf.extend(fetcher(d, http))
                    break
                except Exception as e:  # noqa: BLE001 - 網路層 retry，最後一次仍失敗則拋出
                    if attempt == 2:
                        raise RuntimeError(f"{name} {d} 抓取失敗（3 次重試後）：{e}") from e
                    time.sleep(5 * (attempt + 1))
            time.sleep(RATE_LIMIT_SECONDS)
        if i % 20 == 0 or i == len(todo):
            # checkpoint：每 20 日寫一次 parquet
            new_df = pd.DataFrame(buf)
            merged = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            merged.to_parquet(PARQUET_PATH, index=False)
            existing = merged
            buf = []
            print(f"[fetch] {i}/{len(todo)} 日完成，累計 {len(merged)} 列（checkpoint 已寫入）")

    if buf:
        new_df = pd.DataFrame(buf)
        existing = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        existing.to_parquet(PARQUET_PATH, index=False)
    return existing


def load_board_prices(start: date, end: date) -> pd.DataFrame:
    """整股日行情（close、close×volume→amt），往前多抓 40 個日曆日供 amt_20 warmup。"""
    with get_session() as s:
        rows = s.execute(
            text(
                "SELECT stock_id, trading_date, close, volume FROM raw_prices "
                "WHERE trading_date BETWEEN DATE_SUB(:a, INTERVAL 40 DAY) AND :b"
            ),
            {"a": start, "b": end},
        ).fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "trading_date", "close", "volume"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["amt"] = df["close"] * df["volume"]
    df = df.sort_values(["stock_id", "trading_date"])
    df["amt_20"] = (
        df.groupby("stock_id")["amt"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    )
    return df[["stock_id", "trading_date", "close", "amt_20"]]


def _tier_of(amt_20: float) -> str:
    for bound, label in zip(TIER_BOUNDS, TIER_LABELS[:-1]):
        if amt_20 < bound:
            return label
    return TIER_LABELS[-1]


def calibrate(start: date, end: date) -> dict:
    odd = pd.read_parquet(PARQUET_PATH)
    odd["trading_date"] = pd.to_datetime(odd["trading_date"]).dt.date
    odd = odd[(odd["trading_date"] >= start) & (odd["trading_date"] <= end)]

    common = load_common_stock_ids()
    odd = odd[odd["stock_id"].isin(common)].copy()

    board = load_board_prices(start, end)
    board["trading_date"] = pd.to_datetime(board["trading_date"]).dt.date
    df = odd.merge(board, on=["stock_id", "trading_date"], how="inner")
    df = df[(df["close"] > 0) & df["amt_20"].notna() & (df["amt_20"] > 0)].copy()

    # ── 三個量測 ──
    bid, ask = df["odd_bid"], df["odd_ask"]
    valid_quote = bid.notna() & ask.notna() & (ask >= bid) & (bid > 0)
    mid = (bid + ask) / 2.0
    df["half_spread_rel"] = ((ask - bid) / 2.0 / mid).where(valid_quote)
    # 報價半價差解讀為單邊立即成交成本；離群（>10%）多為無效簿（漲跌停鎖死/極端書），winsorize 到 10%
    df["half_spread_rel"] = df["half_spread_rel"].clip(upper=0.10)

    traded = df["odd_shares"] > 0
    with pd.option_context("mode.use_inf_as_na", True):
        df["odd_vwap"] = (df["odd_amount"] / df["odd_shares"]).where(traded)
    df["vwap_dev"] = (df["odd_vwap"] / df["close"] - 1).where(traded)
    df["basis"] = (df["odd_last"] / df["close"] - 1).where(df["odd_last"].notna())
    # vwap_dev / basis 離群（|x|>15%）為資料錯誤或跨簿極端狀況，排除於統計
    df.loc[df["vwap_dev"].abs() > 0.15, "vwap_dev"] = pd.NA
    df.loc[df["basis"].abs() > 0.15, "basis"] = pd.NA

    df["tier"] = df["amt_20"].map(_tier_of)

    tiers_out = {}
    for label in TIER_LABELS:
        g = df[df["tier"] == label]
        hs = g["half_spread_rel"].dropna().astype(float)
        vd = g["vwap_dev"].dropna().astype(float)
        bs = g["basis"].dropna().astype(float)
        n = len(g)
        tiers_out[label] = {
            "n_stock_days": int(n),
            "n_stocks": int(g["stock_id"].nunique()),
            "no_odd_trade_rate": round(float((g["odd_shares"] == 0).mean()), 4) if n else None,
            "no_quote_rate": round(float(g["half_spread_rel"].isna().mean()), 4) if n else None,
            "half_spread_rel": {
                "n": int(len(hs)),
                "median": round(float(hs.median()), 5) if len(hs) else None,
                "p75": round(float(hs.quantile(0.75)), 5) if len(hs) else None,
                "p90": round(float(hs.quantile(0.90)), 5) if len(hs) else None,
                "mean": round(float(hs.mean()), 5) if len(hs) else None,
            },
            "abs_vwap_dev": {
                "n": int(len(vd)),
                "median": round(float(vd.abs().median()), 5) if len(vd) else None,
                "p75": round(float(vd.abs().quantile(0.75)), 5) if len(vd) else None,
                "mean_signed": round(float(vd.mean()), 5) if len(vd) else None,
            },
            "abs_basis": {
                "n": int(len(bs)),
                "median": round(float(bs.abs().median()), 5) if len(bs) else None,
                "p75": round(float(bs.abs().quantile(0.75)), 5) if len(bs) else None,
                "mean_signed": round(float(bs.mean()), 5) if len(bs) else None,
            },
            # 模型採用值：per-side premium = 該層收盤零股報價半價差的 P75（偏保守；
            # 悲觀臂另乘 1.5）。無報價樣本的層 fallback 至上一層（更差流動性層）值。
            "premium_per_side": round(float(hs.quantile(0.75)), 5) if len(hs) else None,
        }

    # fallback：由最差層往最好層方向依序補 None（正常不會發生，防禦性處理）
    prev = None
    for label in TIER_LABELS:
        if tiers_out[label]["premium_per_side"] is None and prev is not None:
            tiers_out[label]["premium_per_side"] = prev
        prev = tiers_out[label]["premium_per_side"]

    out = {
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "sample_window": {"start": start.isoformat(), "end": end.isoformat()},
        "n_rows_raw": int(len(odd)),
        "n_rows_joined": int(len(df)),
        "estimator": (
            "premium_per_side = P75(half_spread_rel) per tier; "
            "half_spread_rel = (last_ask - last_bid) / 2 / mid（收盤零股報價半價差，winsorize 10%）；"
            "vwap_dev 與 basis 為佐證分佈，不進模型"
        ),
        "tier_bounds_yi": [b / 1e8 for b in TIER_BOUNDS],
        "tier_labels": TIER_LABELS,
        "tiers": tiers_out,
        "sources": {
            "twse": TWSE_URL + "（盤中零股交易行情單 TWTC7U）",
            "tpex": TPEX_URL + "（盤中零股每日收盤行情 oddQuote）",
        },
        "notes": [
            "盤中零股交易 2020-10-26 上線；之前只有盤後零股（一天一撮），該時代 premium 以 ×2 近似（era multiplier）",
            "amt_20 = 整股 close×volume 20 日滾動均值（與 backtest tiered slippage 同語義）",
            "只含上市/上櫃四碼普通股（stocks.security_type='stock'）",
        ],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=str, default="2025-07-01")
    ap.add_argument("--end", type=str, default="2026-06-30")
    ap.add_argument("--calibrate-only", action="store_true", help="跳過抓取，只重算統計")
    ap.add_argument("--fetch-only", action="store_true", help="只抓取，不算統計")
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if not args.calibrate_only:
        fetch_sample(start, end)
    if args.fetch_only:
        return

    out = calibrate(start, end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[calibrate] 已寫入 {CALIBRATION_PATH}")
    print(json.dumps({k: v["premium_per_side"] for k, v in out["tiers"].items()}, indent=2))


if __name__ == "__main__":
    main()
