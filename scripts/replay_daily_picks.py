#!/usr/bin/env python
"""每日訊號歷史重放（2026-07-05）：用「修復後的系統」重放指定區間每個交易日的選股。

回答「過去一個月如果每天都用（修好的）系統選股，結果如何」。與正式回測的差異：
- 粒度：每「交易日」一個 cohort（正式回測是月頻再平衡）
- 模型：單一模型，訓練截止 = 區間起點前第 20 個「交易日」（label horizon 不跨入重放區間，
  無前向洩漏），之後不重訓——模擬「當時部署的模型」
- 報酬：每 cohort 等權 topN，還原價，D → D+20 交易日（不足則 mark-to-market 到資料末端）

⚠️ 誠實聲明（這不是 live track record）：
1. 特徵/價格是「現在」的 DB（含事後回補與修正），非當時 point-in-time 可得版本
2. 一個月 ≈ 月頻策略 1 期樣本；相鄰 cohort 的 20 日窗口高度重疊，非獨立觀測
3. 它能證明的是「系統管線行為」與「近期市況下的粗略表現」，不能證明策略優劣

用法：
    python scripts/replay_daily_picks.py --start 2026-06-01                # 到資料末端
    python scripts/replay_daily_picks.py --start 2026-06-01 --min-amt20 0  # 個人口徑（無門檻）
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import get_session
from skills import data_store
from skills.build_features import PRUNED_FEATURE_COLS
from skills.model_params import RANKER_PROD_PARAMS

HORIZON_TDAYS = 20
TRAIN_LOOKBACK_YEARS = 3


def _adj_close(session, start: date, end: date) -> pd.DataFrame:
    df = pd.read_sql(text(
        "SELECT p.stock_id, p.trading_date, p.close, COALESCE(f.adj_factor, 1.0) AS adj_factor "
        "FROM raw_prices p LEFT JOIN price_adjust_factors f "
        "ON p.stock_id = f.stock_id AND p.trading_date = f.trading_date "
        "WHERE p.trading_date BETWEEN :s AND :e"
    ), session.connection(), params={"s": start, "e": end})
    df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["adj_factor"], errors="coerce")
    return df[["stock_id", "trading_date", "adj_close"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="每日訊號歷史重放（修復後系統）")
    ap.add_argument("--start", required=True, help="重放起始日（YYYY-MM-DD）")
    ap.add_argument("--end", default=None, help="重放結束日（預設：特徵最新日）")
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--min-amt20", type=float, default=5e7,
                    help="20 日均成交值門檻 TWD（預設 5e7 生產口徑；0=個人口徑無門檻）")
    args = ap.parse_args()
    start = date.fromisoformat(args.start)

    with get_session() as s:
        feat_end = s.execute(text("SELECT max(trading_date) FROM features")).scalar()
        end = date.fromisoformat(args.end) if args.end else feat_end
        train_start = start - timedelta(days=365 * TRAIN_LOOKBACK_YEARS)

        print(f"[replay] 區間 {start} ~ {end}，訓練窗 {train_start} 起，topN={args.topn}，"
              f"門檻={args.min_amt20:,.0f}")

        feats = data_store.get_features(s, train_start, end)
        labels = data_store.get_labels(s, train_start, end)

        # 訓練截止：start 前第 HORIZON 個交易日（label 不跨入重放區間）
        all_tds = sorted(feats["trading_date"].unique())
        pos = int(np.searchsorted(np.array(all_tds), start))
        cutoff = all_tds[pos - HORIZON_TDAYS] if pos > HORIZON_TDAYS else train_start
        print(f"[replay] 訓練 label cutoff（交易日制）：{cutoff}")

        merged = feats.merge(labels, on=["stock_id", "trading_date"], how="inner")
        train = merged[merged["trading_date"] < cutoff]
        if train.empty:
            print("訓練資料不足"); return 1

        feat_cols = [c for c in PRUNED_FEATURE_COLS if c in train.columns]
        X = train[feat_cols].astype(float)
        medians = X.median()
        X = X.fillna(medians).fillna(0)
        y = train["future_ret_h"].astype(float)
        w = np.log1p(pd.to_numeric(train.get("amt_20"), errors="coerce").fillna(0).clip(lower=0).values)
        w = w / w.mean() if w.mean() > 0 else None

        import lightgbm as lgb
        model = lgb.LGBMRegressor(**RANKER_PROD_PARAMS)
        model.fit(X.values, y.values, sample_weight=w)
        print(f"[replay] 模型訓練完成（{len(X):,} 筆，{len(feat_cols)} 特徵）")

        px = _adj_close(s, start, end)
        px_days = sorted(px["trading_date"].unique())
        px_piv = px.pivot_table(index="trading_date", columns="stock_id", values="adj_close")

        replay_days = [d for d in all_tds if start <= d <= end]
        rows = []
        for d in replay_days:
            day = feats[feats["trading_date"] == d]
            day = day[day["stock_id"].astype(str).str.fullmatch(r"\d{4}")]
            if args.min_amt20 > 0 and "amt_20" in day.columns:
                day = day[pd.to_numeric(day["amt_20"], errors="coerce") >= args.min_amt20]
            if day.empty:
                continue
            Xd = day[feat_cols].astype(float).fillna(medians).fillna(0)
            scores = model.predict(Xd.values)
            top = day.assign(score=scores).nlargest(args.topn, "score")

            i = int(np.searchsorted(np.array(px_days), d))
            if i >= len(px_days) or px_days[i] != d:
                continue
            j = min(i + HORIZON_TDAYS, len(px_days) - 1)
            exit_d = px_days[j]
            complete = (j == i + HORIZON_TDAYS)

            def _cohort_ret(sids) -> float | None:
                rs = []
                for sid in sids:
                    sid = str(sid)
                    if sid in px_piv.columns:
                        p0, p1 = px_piv.at[d, sid], px_piv.at[exit_d, sid]
                        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                            rs.append(max(float(p1 / p0 - 1), -0.50))
                return float(np.mean(rs)) if rs else None

            pick_ret = _cohort_ret(top["stock_id"])
            bench_ret = _cohort_ret(day["stock_id"])
            rows.append({"date": d, "n_cand": len(day), "pick_ret": pick_ret,
                         "bench_ret": bench_ret,
                         "excess": None if (pick_ret is None or bench_ret is None) else pick_ret - bench_ret,
                         "held_to": exit_d, "complete_20td": complete})

    out = pd.DataFrame(rows)
    if out.empty:
        print("區間內無可重放交易日"); return 1
    pd.set_option("display.width", 160)
    print(out.to_string(index=False,
                        formatters={c: "{:+.2%}".format for c in ("pick_ret", "bench_ret", "excess")}))
    ex = out["excess"].dropna()
    print(f"\n[summary] {len(out)} 個 cohort | picks 平均 {out['pick_ret'].mean():+.2%} | "
          f"同門檻 universe 平均 {out['bench_ret'].mean():+.2%} | 平均超額 {ex.mean():+.2%} | "
          f"正超額 cohort {int((ex > 0).sum())}/{len(ex)}")
    print("⚠️ 相鄰 cohort 窗口重疊，非獨立樣本；此為管線驗證與粗略觀察，非策略證明")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
