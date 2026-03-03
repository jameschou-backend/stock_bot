from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class WindowResult:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    excess_return: float
    win_rate: float


def _year_windows(start_date: date, end_date: date, train_years: int = 5, test_years: int = 1) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date + timedelta(days=365 * train_years)
    while cursor + timedelta(days=365 * test_years) <= end_date:
        test_start = cursor
        test_end = cursor + timedelta(days=365 * test_years) - timedelta(days=1)
        windows.append((test_start, test_end))
        cursor += timedelta(days=365 * test_years)
    return windows


def _resolve_round_trip_cost(raw_cost: float) -> float:
    # 向後相容：若 config 為單邊手續費（0.1425%），轉成來回成本（含證交稅）
    return raw_cost * 4.1 if raw_cost < 0.005 else raw_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Train5Y-Test1Y Walk-forward 驗證")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--topn", type=int, default=None, help="每期選股數（預設使用 config TOPN）")
    parser.add_argument("--resume", action="store_true", help="Resume from existing summary/trades files")
    args = parser.parse_args()

    config = load_config()
    output_dir = PROJECT_ROOT / "artifacts" / "ai_answers"
    output_dir.mkdir(parents=True, exist_ok=True)
    round_trip_cost = _resolve_round_trip_cost(float(config.transaction_cost_pct))
    topn = int(args.topn) if args.topn is not None else int(config.topn)

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

        out_path = output_dir / "walkforward_summary.csv"
        trades_csv_path = output_dir / "walkforward_trades.csv"

        rows: list[WindowResult] = []
        all_trades = []
        completed_test_starts = set()

        if args.resume and out_path.exists():
            try:
                df_exists = pd.read_csv(out_path)
                for _, r in df_exists.iterrows():
                    rows.append(
                        WindowResult(
                            train_start=pd.to_datetime(r["train_start"]).date(),
                            train_end=pd.to_datetime(r["train_end"]).date(),
                            test_start=pd.to_datetime(r["test_start"]).date(),
                            test_end=pd.to_datetime(r["test_end"]).date(),
                            annualized_return=float(r["annualized_return"]),
                            max_drawdown=float(r["max_drawdown"]),
                            sharpe_ratio=float(r["sharpe_ratio"]),
                            excess_return=float(r["excess_return"]),
                            win_rate=float(r["win_rate"]),
                        )
                    )
                    completed_test_starts.add(pd.to_datetime(r["test_start"]).date())
                print(f"[resume] Loaded {len(rows)} windows from {out_path}")
            except Exception as e:
                print(f"[resume] Could not load summary: {e}")

        if args.resume and trades_csv_path.exists():
            try:
                all_trades = pd.read_csv(trades_csv_path).to_dict(orient="records")
                print(f"[resume] Loaded {len(all_trades)} trades from {trades_csv_path}")
            except Exception as e:
                print(f"[resume] Could not load trades: {e}")

        for test_start, test_end in windows:
            if test_start in completed_test_starts:
                print(f"[resume] Skipping window test_start={test_start} (already completed)")
                continue

            train_end = test_start - timedelta(days=1)
            train_start = train_end - timedelta(days=365 * args.train_years)
            result = run_backtest(
                config,
                session,
                backtest_months=args.test_years * 12,
                topn=topn,
                stoploss_pct=config.stoploss_pct,
                transaction_cost_pct=round_trip_cost,
                retrain_freq_months=3,
                eval_start=test_start,
                eval_end=test_end,
                train_lookback_days=args.train_years * 365,
                position_sizing="vol_inverse",
                rebalance_freq="W",
                min_avg_turnover=0.5,
            )
            summary = result.get("summary", {})
            rows.append(
                WindowResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    annualized_return=float(summary.get("annualized_return", 0.0)),
                    max_drawdown=float(summary.get("max_drawdown", 0.0)),
                    sharpe_ratio=float(summary.get("sharpe_ratio", 0.0)),
                    excess_return=float(summary.get("excess_return", 0.0)),
                    win_rate=float(summary.get("win_rate", 0.0)),
                )
            )
            all_trades.extend(result.get("trades_log", []))

            # Incremental save
            out_df = pd.DataFrame([r.__dict__ for r in rows])
            out_df.to_csv(out_path, index=False)
            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                trades_df.to_csv(trades_csv_path, index=False)
            print(f"[walkforward] incrementally saved at {test_start}")

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    out_path = output_dir / "walkforward_summary.csv"
    out_df.to_csv(out_path, index=False)

    summary = {
        "windows": len(out_df),
        "topn": topn,
        "transaction_cost_round_trip": round_trip_cost,
        "annualized_return_mean": float(out_df["annualized_return"].mean()),
        "max_drawdown_mean": float(out_df["max_drawdown"].mean()),
        "sharpe_mean": float(out_df["sharpe_ratio"].mean()),
        "excess_return_mean": float(out_df["excess_return"].mean()),
        "win_rate_mean": float(out_df["win_rate"].mean()),
    }
    json_path = output_dir / "walkforward_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[walkforward] wrote: {out_path}")
    print(f"[walkforward] wrote: {json_path}")
    
    if all_trades:
        print(f"[walkforward] wrote trades log: {trades_csv_path}")


if __name__ == "__main__":
    main()
