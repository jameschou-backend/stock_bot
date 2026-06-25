#!/usr/bin/env python
"""用 FinMind 還原股價反推真實 adj_factor，填入 price_adjust_factors。

adj_factor = finmind_adj_close / raw_close（build_features: adj_close = close × adj_factor）。
sanitize：clip 到 [0.1, 10]（去 penny tick 雜訊/極端減資假值，0.34% 異常）。
無 adj 的 (stock,date) 不寫 → build_features merge 時自動填 1.0（未還原）。

⚠️ sponsor 已過期，adj 只到 2026-06-23 的快照；之後新交易日無 adj（factor=1.0）。
還原方式：UPDATE price_adjust_factors SET adj_factor=1.0（或從 features.bak_preadj 還原特徵）。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text
from app.db import get_session

ADJ_PARQUET = ROOT / "artifacts" / "adj_prices" / "adj_prices_10y.parquet"
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
    m["adj_factor"] = (m["adj_close"] / m["close"]).clip(LO, HI)
    out = m[["stock_id", "trading_date", "adj_factor"]].dropna()
    out["trading_date"] = out["trading_date"].dt.date
    print(f"待寫入 factor={len(out):,}  median={out.adj_factor.median():.4f} "
          f"clip 觸及={(((m.adj_close/m.close)<LO)|((m.adj_close/m.close)>HI)).sum():,}")

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
