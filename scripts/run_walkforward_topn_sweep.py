from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
from sqlalchemy import func

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature, Label
from skills.backtest import run_backtest


def _year_windows(start_date: date, end_date: date, train_years: int = 5, test_years: int = 1) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date + timedelta(days=365 * train_years)
    while cursor + timedelta(days=365 * test_years) <= end_date:
        test_start = cursor
        test_end = cursor + timedelta(days=365 * test_years) - timedelta(days=1)
        windows.append((test_start, test_end))
        cursor += timedelta(days=365 * test_years)
    return windows


def _parse_topn_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        val = int(text)
        if val <= 0:
            raise ValueError(f"topn 必須 > 0, got {val}")
        values.append(val)
    uniq = sorted(set(values))
    if not uniq:
        raise ValueError("topn 清單不可為空")
    return uniq


def _resolve_round_trip_cost(raw_cost: float) -> float:
    return raw_cost * 4.1 if raw_cost < 0.005 else raw_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward TopN 掃描")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--topn-list", type=str, default="3,5,8,12,20")
    args = parser.parse_args()

    cfg = load_config()
    topn_list = _parse_topn_list(args.topn_list)
    round_trip_cost = _resolve_round_trip_cost(float(cfg.transaction_cost_pct))
    output_dir = PROJECT_ROOT / "artifacts" / "ai_answers"
    output_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        min_date = session.query(func.min(Feature.trading_date)).scalar()
        max_feat = session.query(func.max(Feature.trading_date)).scalar()
        max_label = session.query(func.max(Label.trading_date)).scalar()
        if min_date is None or max_feat is None or max_label is None:
            raise RuntimeError("features / labels 資料不足")
        end_date = min(max_feat, max_label)
        windows = _year_windows(min_date, end_date, train_years=args.train_years, test_years=args.test_years)
        if not windows:
            raise RuntimeError("可用資料不足以建立 walk-forward 視窗")

        rows: list[dict] = []
        for topn in topn_list:
            print(f"[topn-sweep] start topn={topn}")
            for idx, (test_start, test_end) in enumerate(windows, start=1):
                result = run_backtest(
                    cfg,
                    session,
                    backtest_months=args.test_years * 12,
                    topn=topn,
                    stoploss_pct=cfg.stoploss_pct,
                    transaction_cost_pct=round_trip_cost,
                    retrain_freq_months=3,
                    eval_start=test_start,
                    eval_end=test_end,
                    train_lookback_days=args.train_years * 365,
                    position_sizing="vol_inverse",
                    rebalance_freq="W",
                    min_avg_turnover=0.5,
                )
                s = result.get("summary", {})
                rows.append(
                    {
                        "topn": topn,
                        "window_index": idx,
                        "test_start": test_start.isoformat(),
                        "test_end": test_end.isoformat(),
                        "annualized_return": float(s.get("annualized_return", 0.0)),
                        "max_drawdown": float(s.get("max_drawdown", 0.0)),
                        "sharpe_ratio": float(s.get("sharpe_ratio", 0.0)),
                        "excess_return": float(s.get("excess_return", 0.0)),
                        "win_rate": float(s.get("win_rate", 0.0)),
                    }
                )
                print(f"[topn-sweep] topn={topn} window={idx}/{len(windows)} done")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("TopN 掃描無結果")

    windows_csv = output_dir / "walkforward_topn_windows.csv"
    df.to_csv(windows_csv, index=False)

    agg = (
        df.groupby("topn", as_index=False)
        .agg(
            windows=("window_index", "count"),
            annualized_return_mean=("annualized_return", "mean"),
            annualized_return_median=("annualized_return", "median"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_worst=("max_drawdown", "min"),
            sharpe_mean=("sharpe_ratio", "mean"),
            win_rate_mean=("win_rate", "mean"),
            excess_return_mean=("excess_return", "mean"),
        )
    )
    # 簡單風險調整分數：偏好高 Sharpe / 低回撤
    agg["score"] = agg["sharpe_mean"] + agg["annualized_return_mean"] - 0.7 * agg["max_drawdown_worst"].abs()
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

    payload = {
        "train_years": args.train_years,
        "test_years": args.test_years,
        "topn_list": topn_list,
        "transaction_cost_round_trip": round_trip_cost,
        "recommended_topn": int(agg.iloc[0]["topn"]),
        "summary_rows": agg.to_dict(orient="records"),
        "windows_csv": str(windows_csv),
    }

    summary_json = output_dir / "walkforward_topn_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[topn-sweep] wrote: {windows_csv}")
    print(f"[topn-sweep] wrote: {summary_json}")


if __name__ == "__main__":
    main()
