#!/usr/bin/env python
"""用 FinMind 還原股價反推真實 adj_factor，填入 price_adjust_factors。

adj_factor = finmind_adj_close / raw_close（build_features: adj_close = close × adj_factor）。
sanitize：clip 到 [0.1, 10]（去 penny tick 雜訊/極端減資假值）。
**觸 clip 的股票其還原序列 by construction 不可信**，清單輸出到
artifacts/adj_prices/adj_factor_clip_touched.json，供基準驗證時評估是否排除。

缺日展開（2026-07-03 健檢 P1-1）：factor 只在除權息/減資日跳變，
per-stock 把 factor ffill/bfill 展開到該股「全部」raw 交易日——
內部缺日沿用前值、leading gap 用最早已知 factor。
（先前只寫 raw∩adj 交集日，10.7 萬列內部缺日在 build 端被 fillna(1.0)，
在 factor<1 區段中間製造 ±(1/f-1) 單日假跳動，同時污染特徵與 label。
build_features/build_labels 端現也有 ffill 防禦，此處展開是雙保險 +
讓直接讀 factor 表的其他消費者拿到完整序列。）

整檔無 adj 的股票（160 檔下市股）不寫 → build 端自動 1.0（未還原）。

⚠️ sponsor 已過期，adj 只到 2026-06-23 的快照；之後新交易日無 factor，
由 build 端 ffill 沿用最後 factor（序列連續，優於回退 1.0 的假跳）。
還原方式：UPDATE price_adjust_factors SET adj_factor=1.0（或從 features.bak_preadj 還原特徵）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text
from app.db import get_session

ADJ_PARQUET = ROOT / "artifacts" / "adj_prices" / "adj_prices_10y.parquet"
CLIP_TOUCHED_JSON = ROOT / "artifacts" / "adj_prices" / "adj_factor_clip_touched.json"
LO, HI = 0.1, 10.0


def main() -> int:
    adj = pd.read_parquet(ADJ_PARQUET)[["stock_id", "trading_date", "close"]]
    adj["stock_id"] = adj["stock_id"].astype(str)
    adj["trading_date"] = pd.to_datetime(adj["trading_date"]).dt.normalize()
    adj = adj.rename(columns={"close": "adj_close"})

    with get_session() as s:
        raw = pd.read_sql(text(r"SELECT stock_id, trading_date, close FROM raw_prices "
                               r"WHERE stock_id REGEXP '^[0-9]{4}$'"), s.connection())
    raw["stock_id"] = raw["stock_id"].astype(str)
    raw["trading_date"] = pd.to_datetime(raw["trading_date"]).dt.normalize()
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")

    m = raw.merge(adj, on=["stock_id", "trading_date"], how="inner")
    m = m[(m["close"] > 0) & (m["adj_close"] > 0)].copy()
    ratio = m["adj_close"] / m["close"]
    m["adj_factor"] = ratio.clip(LO, HI)

    # ── P2-9：記錄觸 clip 的股票（還原序列不可信，供訓練/驗證排除評估）──
    clipped_mask = (ratio < LO) | (ratio > HI)
    payload: dict = {}
    if clipped_mask.any():
        stats = (
            m[clipped_mask]
            .assign(_ratio=ratio[clipped_mask])
            .groupby("stock_id")["_ratio"]
            .agg(["min", "max", "size"])
        )
        payload = {
            sid: {
                "clipped_days": int(row["size"]),
                "min_ratio": float(row["min"]),
                "max_ratio": float(row["max"]),
            }
            for sid, row in stats.iterrows()
        }
    CLIP_TOUCHED_JSON.parent.mkdir(parents=True, exist_ok=True)
    CLIP_TOUCHED_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"clip 觸及股票 {len(payload)} 檔 / {int(clipped_mask.sum()):,} 列 → {CLIP_TOUCHED_JSON}")

    # ── P1-1：per-stock 展開到全部 raw 交易日（內部缺日 ffill、leading gap bfill）──
    factors = m[["stock_id", "trading_date", "adj_factor"]]
    covered = raw[raw["stock_id"].isin(factors["stock_id"].unique())][["stock_id", "trading_date"]]
    full = covered.merge(factors, on=["stock_id", "trading_date"], how="left")
    full = full.sort_values(["stock_id", "trading_date"])
    n_gap = int(full["adj_factor"].isna().sum())
    full["adj_factor"] = full.groupby("stock_id", sort=False)["adj_factor"].ffill()
    full["adj_factor"] = full.groupby("stock_id", sort=False)["adj_factor"].bfill()
    out = full.dropna(subset=["adj_factor"]).copy()
    out["trading_date"] = out["trading_date"].dt.date
    print(f"待寫入 factor={len(out):,}（含缺日展開 {n_gap:,} 列） "
          f"median={out.adj_factor.median():.4f}")

    with get_session() as s:
        s.execute(text("TRUNCATE TABLE price_adjust_factors"))
        s.commit()
        eng = s.get_bind()
        CH = 50000
        for i in range(0, len(out), CH):
            out.iloc[i:i+CH].to_sql("price_adjust_factors", eng, if_exists="append",
                                    index=False, method="multi")
            if (i // CH) % 10 == 0:
                print(f"  寫入 {min(i+CH, len(out)):,}/{len(out):,}", flush=True)
        r = s.execute(text("SELECT COUNT(*), AVG(adj_factor), MIN(adj_factor), MAX(adj_factor) "
                           "FROM price_adjust_factors")).fetchone()
    print(f"完成：price_adjust_factors {r[0]:,} 列  avg={r[1]:.4f} min={r[2]:.4f} max={r[3]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
