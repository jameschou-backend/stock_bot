#!/usr/bin/env python
"""Stage 9.2: Optuna 超參數搜尋（60mo 快速搜尋，10y 驗證 top-K）。

設計：
  - 60mo backtest 當作搜尋目標（單 trial ~3-5 min）
  - TPE sampler（Bayesian optimization）
  - SQLite storage 可中斷續跑
  - 每 trial 自動進 MLflow tracking
  - 5 維精簡搜尋空間（避免高維災難）

用法：
    python scripts/optuna_search.py --n-trials 30        # 跑 30 trials
    python scripts/optuna_search.py --n-trials 30 --resume   # 續跑既有 study
    python scripts/optuna_search.py --study-name my_run --storage sqlite:///foo.db

搜尋完後 top-5 candidates 列印；自行手動跑 10y 驗證。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna

from app.config import load_config
from app.db import get_session
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS
from skills.trial_registry import record_backtest_trial
from skills.mlflow_tracking import (
    log_backtest_result,
    log_metrics,
    log_params,
    start_mlflow_run,
)

# 預設搜尋空間 — 5 維度（精簡）
# NOTE: min_avg_turnover 單位是「億元」（backtest.py:310 內部 × 1e8）
#       生產預設 1.0（1 億元），範圍 0.5~3.0 對應 5千萬~3億元，合理流動性門檻
DEFAULT_SEARCH_DIMS = {
    "topn": [10, 15, 20, 25],
    "min_avg_turnover": (0.5, 3.0),      # 億元，step 0.5
    "vol_target_pct": (0.0, 0.40),       # 0=disabled, > 0 啟用
    "ensemble_n_checkpoints": [1, 3, 5],
    "liquidity_weighting": [True, False],
}


def objective(trial: optuna.Trial, months: int, mlflow_experiment: str | None) -> float:
    """單一 trial：跑一次 60mo backtest，回傳 Sharpe ratio。"""
    # ── 從 search space 取參 ──
    topn = trial.suggest_categorical("topn", DEFAULT_SEARCH_DIMS["topn"])
    min_avg_turnover = trial.suggest_float(
        "min_avg_turnover",
        DEFAULT_SEARCH_DIMS["min_avg_turnover"][0],
        DEFAULT_SEARCH_DIMS["min_avg_turnover"][1],
        step=0.5,
    )
    vol_target_pct = trial.suggest_float(
        "vol_target_pct",
        DEFAULT_SEARCH_DIMS["vol_target_pct"][0],
        DEFAULT_SEARCH_DIMS["vol_target_pct"][1],
        step=0.05,
    )
    ensemble_n = trial.suggest_categorical("ensemble_n_checkpoints",
                                            DEFAULT_SEARCH_DIMS["ensemble_n_checkpoints"])
    liquidity_weighting = trial.suggest_categorical("liquidity_weighting",
                                                     DEFAULT_SEARCH_DIMS["liquidity_weighting"])

    cfg = load_config()
    run_name = f"trial_{trial.number:03d}"

    # ── 跑 backtest（在 mlflow run 內）──
    with start_mlflow_run(
        experiment_name=mlflow_experiment or "optuna_search",
        run_name=run_name,
        tags={"trial_number": str(trial.number), "study": trial.study.study_name},
        enabled=mlflow_experiment is not None,
    ) as ml_run:
        log_params({
            "topn": topn,
            "min_avg_turnover": min_avg_turnover,
            "vol_target_pct": vol_target_pct,
            "ensemble_n_checkpoints": ensemble_n,
            "liquidity_weighting": liquidity_weighting,
            "months": months,
        }, mlflow_run=ml_run)

        with get_session() as db:
            result = run_backtest(
                cfg, db,
                backtest_months=months,
                topn=topn,
                stoploss_pct=0.0,
                enable_seasonal_filter=True,
                market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
                market_filter_min_positions=2,
                liquidity_weighting=liquidity_weighting,
                feature_columns=PRUNED_FEATURE_COLS,
                min_avg_turnover=min_avg_turnover,
                vol_target_pct=vol_target_pct,
                ensemble_n_checkpoints=ensemble_n,
                label_horizon_buffer=20,
            )
        log_backtest_result(result, mlflow_run=ml_run)

    # 統計紀律：每個 Optuna trial 都是一次 selection candidate，必須進 trial registry
    # （只記最後「採用的」會低估 DSR 的 multiple-testing 折扣；失敗不阻斷搜尋）
    record_backtest_trial(
        result, months=months, source="optuna_search",
        params={"trial_number": trial.number, "study": trial.study.study_name, **trial.params},
    )

    s = result.get("summary", {})
    sharpe = float(s.get("sharpe_ratio") or 0.0)
    mdd = float(s.get("max_drawdown") or 0.0)
    # 主目標：Sharpe
    # 次目標（trial.set_user_attr 紀錄但不直接優化）
    trial.set_user_attr("max_drawdown", mdd)
    trial.set_user_attr("calmar_ratio", float(s.get("calmar_ratio") or 0.0))

    return sharpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20, help="trial 數（預設 20）")
    p.add_argument("--months", type=int, default=60, help="每 trial 回測月數（預設 60）")
    p.add_argument("--study-name", type=str, default="stockbot_v1",
                   help="optuna study 名（用於儲存 + resume）")
    p.add_argument("--storage", type=str, default="sqlite:///optuna.db",
                   help="optuna storage URI（預設 SQLite）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow-experiment", type=str, default="optuna_search",
                   help="每 trial 進 mlflow（設為 'none' 停用）")
    p.add_argument("--no-mlflow", action="store_true", help="完全不記錄 mlflow")
    p.add_argument("--resume", action="store_true", help="續跑既有 study")
    args = p.parse_args()

    mlflow_exp = None if args.no_mlflow else args.mlflow_experiment

    print(f"\n{'='*70}")
    print(f"Stage 9.2：Optuna 超參數搜尋")
    print(f"{'='*70}")
    print(f"  study:          {args.study_name}")
    print(f"  storage:        {args.storage}")
    print(f"  n_trials:       {args.n_trials}")
    print(f"  per-trial:      {args.months}mo backtest")
    print(f"  mlflow exp:     {mlflow_exp or '(off)'}")
    print(f"  search dims:    {list(DEFAULT_SEARCH_DIMS.keys())}")

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5)
    if args.resume:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage, sampler=sampler)
        print(f"  resumed at trial {len(study.trials)}")
    else:
        study = optuna.create_study(
            study_name=args.study_name, storage=args.storage,
            direction="maximize", sampler=sampler, load_if_exists=True,
        )

    study.optimize(
        lambda t: objective(t, months=args.months, mlflow_experiment=mlflow_exp),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 70)
    print(f"完成。total trials: {len(study.trials)}")
    print(f"best sharpe: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")
    print("\nTop-5 trials by Sharpe:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: -t.value,
    )
    print(f"{'#':>4} {'sharpe':>8} {'mdd':>8} {'calmar':>8}  params")
    for t in sorted_trials[:5]:
        mdd = t.user_attrs.get("max_drawdown", 0)
        calmar = t.user_attrs.get("calmar_ratio", 0)
        print(f"{t.number:>4d} {t.value:>+8.4f} {mdd:>+8.4f} {calmar:>+8.4f}  {t.params}")
    print("=" * 70)
    print("\n下一步：對 top-3 候選跑 --months 120 完整驗證")


if __name__ == "__main__":
    main()
