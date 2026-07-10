#!/usr/bin/env python
"""Walk-forward backtest CLI.

基本用法:
    python scripts/run_backtest.py                      # 預設 24 個月
    python scripts/run_backtest.py --months 36          # 36 個月
    python scripts/run_backtest.py --topn 10            # 選 10 檔

出場策略:
    python scripts/run_backtest.py --stoploss -0.07             # 固定停損 -7%
    python scripts/run_backtest.py --no-stoploss                # 不設停損
    python scripts/run_backtest.py --trailing-stop -0.12        # 移動停利 -12%
    python scripts/run_backtest.py --atr-stoploss 2.5           # ATR×2.5 動態停損

倉位分配:
    python scripts/run_backtest.py --sizing equal               # 等權（預設）
    python scripts/run_backtest.py --sizing score_tiered        # 依分數分層
    python scripts/run_backtest.py --sizing vol_inverse         # 波動率反比

回測可信度參數:
    python scripts/run_backtest.py --entry-delay 1              # 隔日進場（預設）
    python scripts/run_backtest.py --risk-free 0.015            # 無風險利率 1.5%
    python scripts/run_backtest.py --benchmark-tc 0.0058425     # 敏感度分析：benchmark 每期扣成本
                                                                # （預設 0 = zero_cost，buy-and-hold 近似）

10y 逐步優化實驗（每次只改一個變數）:
    python scripts/run_backtest.py --months 120 --baseline      # 乾淨基準（無時間加權/無複雜過濾）
    python scripts/run_backtest.py --months 120 --baseline --change-a   # Change A: +IC 特徵
    python scripts/run_backtest.py --months 120 --baseline --topn-floor 5  # Change B: topN floor=5
    python scripts/run_backtest.py --months 120 --baseline --slippage  # Change C: 滑價模型
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from skills import data_store
from skills.backtest import compute_hedged_metrics, run_backtest
from skills.build_features import BASELINE_FEATURE_COLS, CHANGE_A_FEATURE_COLS, PRUNED_FEATURE_COLS
from skills.mlflow_tracking import (
    DEFAULT_EXPERIMENT,
    log_backtest_result,
    log_params,
    start_mlflow_run,
)
from skills.statistics import (
    deflated_sharpe_ratio,
    paired_block_bootstrap_sharpe_ci,
    returns_moments,
)

# ── 統計紀律工具化（2026-07-10，總體檢缺陷 6 規則 1）──────────────────────────
# trial registry 實作移至 skills.trial_registry（單一真相源）：除本 CLI 外，
# optuna_search / run_grid_backtest / run_walkforward* 等多 trial 工具在每次
# run_backtest 評估後也 append——「每跑一次回測都算一次 trial」的宣稱語義才成立。
# 此處 re-export 供既有測試 / 呼叫端向後相容。
from skills.trial_registry import (  # noqa: E402
    HISTORICAL_TRIALS_BASE,
    TRIAL_REGISTRY_PATH,
    append_trial_registry,
    registry_trial_count,
)

# bootstrap CI 固定參數（缺陷 6 規則 1：月報酬 paired block-bootstrap，block~6、1000 次）
BOOTSTRAP_BLOCK_SIZE = 6
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42

# 跨 trial Sharpe 標準差（DSR 的 sr_estimates_std）：Bailey & LdP 規定以「所有 trial
# 的 SR 估計值離散度」估計。本專案歷史 trial 年化 Sharpe（docs/experiments_history.md
# + memory/decisions.md）：0.49（Exp F）~ 1.33（Stage 10.1，後證洩漏），大量 NEGATIVE
# 實驗聚集 0.6~1.0 → 年化 std ≈ 0.25。DSR 公式用「每期（月）SR」單位 → /sqrt(12)。
# 註：若改用「完全未知」保守上限 1.0（年化），E[max SR|null] ≈ 年化 2.46，對本專案
# 任何結果恆 p≈0——過度保守到喪失鑑別力，故採實證估計（調整時同步改此註解）。
TRIAL_SR_STD_ANNUAL = 0.25
TRIAL_SR_STD_MONTHLY = TRIAL_SR_STD_ANNUAL / (12 ** 0.5)


def production_baseline_overrides() -> dict:
    """生產基準配置 preset（--production-baseline 一鍵套用）。

    對應 CLAUDE.md「生產 CLI 指令」+ 生產 universe 流動性門檻，消除手打旗標漂移：
        --topn 30 --seasonal-filter --no-stoploss
        --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2
        --liq-weighted --pruned-features --min-avg-turnover 0.5

    內容由 tests/test_production_invariants.py 鎖定；變更生產配置須同步改測試 + CLAUDE.md。
    CLI 顯式旗標優先於本 preset（見 main() 內套用邏輯）。
    """
    return {
        "topn": 30,
        "enable_seasonal_filter": True,
        "no_stoploss": True,
        "market_filter_tiers": "-0.05:0.5,-0.10:0.25,-0.15:0.10",
        "market_filter_min_positions": 2,
        "liquidity_weighting": True,
        "pruned_features": True,
        "min_avg_turnover": 0.5,  # 億元；對齊生產 daily_pick universe 門檻（config.min_avg_turnover）
    }


def resolve_round_trip_cost(cli_cost, config) -> tuple:
    """解析「來回」交易成本（P2-7a：消除 <0.005 猜單位 silent fallback）。

    優先序：
      1. CLI --transaction-cost / --cost（來回，語義照舊）
      2. config.transaction_cost_round_trip（來回，顯式設定）
      3. config.transaction_cost_pct × 4.1（單邊→來回顯式換算：
         買 0.1425% + 賣 0.1425% + 證交稅 0.3% ≈ 單邊 × 4.1）

    Returns:
        (round_trip_cost, source_tag)
    """
    if cli_cost is not None:
        return float(cli_cost), "cli"
    _rt = getattr(config, "transaction_cost_round_trip", None)
    if _rt is not None:
        return float(_rt), "config.transaction_cost_round_trip"
    _one_way = float(config.transaction_cost_pct)
    _round_trip = _one_way * 4.1
    return _round_trip, "config.transaction_cost_pct*4.1"


# ──────────────────────────────────────────────
# 統計紀律：trial registry + bootstrap CI + DSR
# ──────────────────────────────────────────────

def compute_statistics_block(result: dict, n_trials: int, risk_free_rate: float) -> dict | None:
    """從回測 result 計算統計紀律區塊（bootstrap Sharpe CI + DSR p-value）。

    - Sharpe 95% CI：月報酬 paired circular block-bootstrap（block=6、1000 次、
      seed 固定 → deterministic），與 benchmark 同索引重抽，另附 excess Sharpe CI。
    - DSR：Bailey & López de Prado 2014。公式使用「每期（月）SR」+ 月觀察數
      （單位一致性：sqrt(n_obs-1) 是月數）；skew/kurt 取自月報酬實際動差；
      n_trials = trial registry 行數 + HISTORICAL_TRIALS_BASE。

    periods 不足（<2）回傳 None。任何一步失敗不阻斷回測（呼叫端 try/except）。
    """
    periods = result.get("periods") or []
    rets, bench = [], []
    for p in periods:
        r, b = p.get("return"), p.get("benchmark_return")
        if r is not None:
            rets.append(float(r))
            bench.append(float(b) if b is not None else 0.0)
    if len(rets) < 2:
        return None

    import numpy as np

    rets_arr = np.asarray(rets)
    bench_arr = np.asarray(bench)

    boot = paired_block_bootstrap_sharpe_ci(
        rets_arr,
        bench_arr,
        block_size=BOOTSTRAP_BLOCK_SIZE,
        n_boot=BOOTSTRAP_N_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        periods_per_year=12,
        risk_free_rate=risk_free_rate,
    )

    moments = returns_moments(rets_arr)
    # DSR 用每期（月）SR：與 summary 年化 Sharpe 同口徑但不乘 sqrt(12)
    rf_monthly = (1 + risk_free_rate) ** (1 / 12) - 1
    monthly_std = rets_arr.std()  # ddof=0，與 backtest summary 一致
    monthly_sr = float((rets_arr.mean() - rf_monthly) / monthly_std) if monthly_std > 0 else 0.0
    dsr = deflated_sharpe_ratio(
        sr_observed=monthly_sr,
        n_trials=n_trials,
        n_observations=len(rets_arr),
        skewness=moments["skewness"],
        kurtosis=moments["kurtosis"],
        sr_estimates_std=TRIAL_SR_STD_MONTHLY,
    )

    return {
        "sharpe_ci_95": {
            "sharpe_observed": round(boot.sharpe_observed, 4),
            "ci_low": round(boot.ci_low, 4),
            "ci_high": round(boot.ci_high, 4),
            "excess_sharpe_observed": round(boot.excess_sharpe_observed, 4),
            "excess_ci_low": round(boot.excess_ci_low, 4),
            "excess_ci_high": round(boot.excess_ci_high, 4),
            "method": "paired_circular_block_bootstrap",
            "block_size": boot.block_size,
            "n_boot": boot.n_boot,
            "seed": boot.seed,
            "n_observations": boot.n_observations,
        },
        "deflated_sharpe": {
            "sr_observed_monthly": round(dsr.sr_observed, 4),
            "sr_expected_max_under_null": round(dsr.sr_expected_under_null, 4),
            "p_value": round(dsr.p_value, 4),
            "is_significant_5pct": dsr.is_significant_5pct,
            "n_trials": dsr.n_trials,
            "n_observations": dsr.n_observations,
            "sr_estimates_std_monthly": round(TRIAL_SR_STD_MONTHLY, 4),
            "n_trials_source": (
                f"trial_registry({n_trials - HISTORICAL_TRIALS_BASE}) "
                f"+ historical_base({HISTORICAL_TRIALS_BASE})"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward 回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 基礎參數 ──
    parser.add_argument("--months", type=int, default=24, help="回測月數 (預設 24)")
    parser.add_argument("--retrain-freq", type=int, default=3, help="模型重訓頻率（月，預設 3）")
    parser.add_argument("--topn", type=int, default=None, help="每期選股數量 (預設使用 config)")
    parser.add_argument("--cost", "--transaction-cost", type=float, default=None,
                        dest="cost", help="來回交易成本 (如 0.00585)")
    parser.add_argument("--production-baseline", action="store_true",
                        dest="production_baseline",
                        help="一鍵套用生產基準配置（topn=30、seasonal-filter、no-stoploss、"
                             "market-filter-tiers、min-pos=2、liq-weighted、pruned-features、"
                             "min-avg-turnover=0.5）。CLI 顯式旗標優先於 preset。")

    # ── 出場策略 ──
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stoploss", type=float, default=None,
                            help="固定停損比例 (如 -0.07 = -7%%)")
    stop_group.add_argument("--no-stoploss", action="store_true",
                            help="不設停損")
    stop_group.add_argument("--atr-stoploss", type=float, default=None,
                            dest="atr_stoploss_multiplier",
                            help="ATR 倍數動態停損 (如 2.5 = 2.5×ATR)，覆蓋固定停損")
    parser.add_argument("--trailing-stop", type=float, default=None,
                        dest="trailing_stop_pct",
                        help="移動停利：從最高點回落觸發比例 (如 -0.12 = -12%%)")
    parser.add_argument("--atr-period", type=int, default=14,
                        help="ATR 計算週期（日，預設 14）")

    # ── 倉位分配 ──
    parser.add_argument("--sizing", type=str, default=None,
                        dest="position_sizing",
                        choices=["equal", "score_tiered", "vol_inverse"],
                        help="倉位分配方式 (預設 equal)")
    parser.add_argument("--ps-method", type=str, default=None,
                        dest="position_sizing_method",
                        choices=["vol_inverse", "mean_variance", "risk_parity"],
                        help="進階倉位最佳化方法 (預設 risk_parity)")

    # ── 回測可信度 ──
    parser.add_argument("--entry-delay", type=int, default=None,
                        dest="entry_delay_days",
                        help="進場延遲交易日 (預設 1，0=同日收盤進場舊行為)")
    parser.add_argument("--risk-free", type=float, default=None,
                        dest="risk_free_rate",
                        help="無風險利率年化 (預設 0.015 = 1.5%%)")
    parser.add_argument("--no-benchmark-cost", action="store_true",
                        help="DEPRECATED no-op：benchmark 零成本已是預設（2026-07-10 缺陷 3 修復）")
    parser.add_argument("--benchmark-tc", type=float, default=0.0,
                        dest="benchmark_tc",
                        help="敏感度分析：benchmark 每期扣減成本（來回口徑，如 0.0058425）。"
                             "預設 0 = zero_cost（buy-and-hold 近似）。注意：>0 等同假設 "
                             "benchmark 每期 100%% 周轉，僅供敏感度對照，不可作 headline。"
                             "僅影響報表的 benchmark_return/超額；market_filter_tiers 降倉"
                             "訊號讀零成本 market_return_raw，策略臂持倉決策不受本旗標影響")

    # ── 10y 逐步優化實驗參數 ──
    parser.add_argument("--baseline", action="store_true",
                        help="乾淨基準：停用時間加權訓練、停用複雜市場過濾，使用 BASELINE_FEATURE_COLS")
    parser.add_argument("--change-a", action="store_true",
                        help="Change A：在 baseline 特徵集加入 IC 最優特徵（trust_net_5_inv, theme_turnover_ratio, fund_revenue_mom）")
    parser.add_argument("--pruned-features", action="store_true",
                        dest="pruned_features",
                        help="SHAP 剪枝特徵集：移除 8 個低重要性特徵（56→48 個）")
    parser.add_argument("--topn-floor", type=int, default=0,
                        dest="topn_floor",
                        help="topN 最低下限 (0=不強制；5=Change B)；與 --baseline 配合使用")
    parser.add_argument("--slippage", action="store_true",
                        help="Change C：啟用滑價模型（ATR 的 10%%，上限 0.3%%）")
    parser.add_argument("--no-slippage", action="store_true",
                        help="停用滑價模型（baseline 預設已停用）")
    parser.add_argument("--tiered-slippage", action="store_true",
                        dest="tiered_slippage",
                        help="分級滑價模型：依 amt_20 決定流動性層級（<1億 1.0%%, 1~5億 0.6%%, >5億 0.2%% 來回）")
    parser.add_argument("--liq-weighted", action="store_true",
                        dest="liquidity_weighting",
                        help="流動性加權訓練：sample_weight ∝ log(1+amt_20)，讓模型學偏大型股模式")
    parser.add_argument("--rebalance-freq", type=str, default=None,
                        dest="rebalance_freq",
                        choices=["W", "M"],
                        help="再平衡頻率：W=週頻, M=月頻（預設 M）")
    parser.add_argument("--seasonal-filter", action="store_true",
                        dest="enable_seasonal_filter",
                        help="啟用季節性降倉：3/10月 topN×0.5（對應 daily_pick 行為）")
    parser.add_argument("--no-seasonal-filter", action="store_true",
                        dest="no_seasonal_filter",
                        help="明確停用季節性降倉（診斷用）")

    # ── 突破進場 ──
    parser.add_argument("--breakthrough-entry", action="store_true",
                        dest="enable_breakthrough_entry",
                        help="突破確認進場：月底選股後等突破訊號再進場（最多等 10 個交易日）")
    parser.add_argument("--breakthrough-wait", type=int, default=10,
                        dest="breakthrough_max_wait",
                        help="突破進場最大等待交易日（預設 10）")

    # ── 圓桌策略進場過濾組（2026-03-20）──
    parser.add_argument("--filter-group", type=str, default=None,
                        dest="filter_group",
                        choices=["A", "B", "C"],
                        help=(
                            "圓桌策略進場過濾組（搭配 --breakthrough-entry 使用）\n"
                            "  A（低風險）：均線多頭排列 + bias20<=20%%\n"
                            "  B（中風險）：A + 量價背離排除 + 外資過熱排除(top80%%)\n"
                            "  C（高風險）：B + ret_20_rank 35~75%% + ret_60_rank>30%% + volume_surge>=1.5"
                        ))

    # ── 動能懲罰 ──
    parser.add_argument("--momentum-penalty", action="store_true",
                        dest="momentum_penalty",
                        help="對 bias_20/ret_5/ret_20 乘以 0.5 再送入模型（懲罰高動能股）")

    # ── ATR 動態停損 & 大盤過濾 ──
    parser.add_argument("--atr-dynamic-stoploss", action="store_true",
                        dest="atr_dynamic_stoploss",
                        help="ATR 動態停損：低波動股 -15%%、高波動股 -25%%（以 atr_inv 中位數分界）")
    parser.add_argument("--market-filter", action="store_true",
                        dest="market_filter",
                        help="大盤過濾：前期大盤月跌>5%% 持股減半，>10%% 全現金")
    parser.add_argument("--market-filter-tiers", type=str, default=None,
                        dest="market_filter_tiers",
                        help="漸進式大盤過濾，格式：'threshold1:mult1,threshold2:mult2,...' "
                             "例如 '-0.05:0.5,-0.10:0.25,-0.15:0.10'（由淺到深排序）")
    parser.add_argument("--market-filter-min-pos", type=int, default=1,
                        dest="market_filter_min_positions",
                        help="大盤過濾後最低持股數（預設 1，設 2 或 3 防止單押集中風險）")

    # ── 進場訊號過濾 ──
    parser.add_argument("--entry-filter", type=str, default=None,
                        dest="entry_signal_filter",
                        help="進場訊號過濾，格式：'key1=val1,key2=val2,...' "
                             "支援：foreign_buy_streak_max, rsi_min, rsi_max, bias_20_max, volume_surge_ratio_min")

    # ── 流動性過濾 ──
    parser.add_argument("--min-avg-turnover", type=float, default=0.0,
                        dest="min_avg_turnover",
                        help="流動性門檻：20日平均日成交金額（億元），0=不過濾（預設）。"
                             "例：1=1億, 3=3億, 5=5億")

    # ── 診斷 ──
    parser.add_argument("--train-lookback", type=int, default=None,
                        dest="train_lookback_days",
                        help="訓練視窗長度（日，如 1825=5年滾動窗）；預設 None=使用全部歷史")
    parser.add_argument("--no-clip", action="store_true",
                        help="停用單筆損失 clip -50%%（診斷用，傳入 clip_loss_pct=-1.01）")
    parser.add_argument("--cap-daily-return", type=float, default=0.0,
                        dest="cap_daily_return_pct",
                        help="診斷：把持有期每日報酬對稱 winsorize 到 ±此值（如 0.10=±10%%，"
                             "=台股漲跌限語義），中和未還原減資/停牌假跳動。預設 0=停用")
    parser.add_argument("--portfolio-circuit-breaker", type=float, default=None,
                        dest="portfolio_circuit_breaker_pct",
                        help="投資組合熔斷：月中等權累積報酬跌破此值時全出場（如 0.15 = -15%%）。"
                             "傳正數即可，內部自動取負值。")
    parser.add_argument("--excess-label", action="store_true",
                        dest="excess_label",
                        help="P1-2：使用等權超額報酬作為訓練 label（future_ret_h - 市場均值）")
    parser.add_argument("--lambdarank", action="store_true",
                        dest="use_lambdarank",
                        help="P1：使用 LightGBM LambdaRank 替代 regression，直接優化 NDCG@20 截面排名")
    parser.add_argument("--cs-norm", action="store_true",
                        dest="cross_section_normalize",
                        help="P2-1：截面 Z-score 正規化，消除特徵絕對值尺度跨年漂移")
    parser.add_argument("--ensemble", type=int, default=1,
                        dest="ensemble_n_checkpoints", metavar="N",
                        help="P2-2：保留最近 N 次重訓 checkpoint 並平均排名分數（預設 1=停用，建議 3）")
    parser.add_argument("--vol-target", type=float, default=0.0,
                        dest="vol_target_pct", metavar="VOL",
                        help="Stage 7.2 Vol Targeting：picks 60d realized vol > VOL 時拉高 cash_ratio。"
                             "預設 0=停用；建議 0.30（10y WF Sharpe Δ +0.078, MDD Δ +4.29pp）")
    parser.add_argument("--vol-target-lookback", type=int, default=60,
                        dest="vol_target_lookback_days", metavar="N",
                        help="vol-target 回溯天數（預設 60）")
    parser.add_argument("--use-stacking", action="store_true",
                        dest="use_stacking",
                        help="Stage 6.1：啟用 LightGBM+XGBoost+CatBoost rank-averaged stacking。"
                             "Quick eval IC lift +7.1%%；與 --use-lambdarank 互斥。")
    parser.add_argument("--stacking-val-frac", type=float, default=0.20,
                        dest="stacking_val_frac",
                        help="stacking 切尾段 val 比例（預設 0.20）")
    parser.add_argument("--recent-dd-skip", type=float, default=0.0,
                        dest="recent_dd_skip_pct", metavar="PCT",
                        help="Stage 10.4 D1：ret_20 < PCT 時排除 candidate（負值啟用，"
                             "例如 -0.15 代表上月跌 15%% 不選）。0=disabled。")
    parser.add_argument("--max-per-sector", type=int, default=0,
                        dest="max_per_sector", metavar="N",
                        help="Stage 10.5 D2：同產業最大持股 N 檔（0=disabled，建議 3~5）。"
                             "DD attribution 顯示 2025-03 觀光餐旅 2 檔同時崩 -10.92%%")
    parser.add_argument("--hedge-ratio", type=float, default=0.0,
                        dest="hedge_ratio", metavar="H",
                        help="Stage 10.6：beta-hedge 後處理（0=disabled）。回測完算 "
                             "hedged_return = port_ret - H × benchmark_ret，輸出 hedged metrics "
                             "供未來期貨對沖實作評估。建議試 0.5 / 1.0 / OLS beta。")

    # ── Stage 9.1 MLflow ──
    parser.add_argument("--mlflow", action="store_true",
                        dest="mlflow_enabled",
                        help="Stage 9.1：啟用 MLflow 追蹤（寫入 ./mlruns/）。"
                             "用 `mlflow ui --port 5000` 查看 dashboard。")
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_EXPERIMENT,
                        dest="mlflow_experiment",
                        help=f"MLflow experiment 名（預設 {DEFAULT_EXPERIMENT}）")
    parser.add_argument("--mlflow-run-name", type=str, default=None,
                        dest="mlflow_run_name",
                        help="MLflow run 名（預設 timestamp_gitSHA）")

    # ── 速度 ──
    parser.add_argument("--fast", action="store_true",
                        help="快速模式：LightGBM 樹數 500→150，加速 ~3x（精度略降）")

    # ── 輸出 ──
    parser.add_argument("--output", type=str, default=None,
                        help="結果輸出 JSON 路徑")

    # ── 實驗紀錄 ──
    parser.add_argument("--log-experiment", action="store_true",
                        dest="log_experiment",
                        help="自動寫入 artifacts/experiments/<timestamp>_<name>.json 實驗紀錄")
    parser.add_argument("--experiment-name", type=str, default=None,
                        dest="experiment_name",
                        help="實驗名稱（用於紀錄檔名，如 'liq_weighted_v2'）")

    args = parser.parse_args()
    config = load_config()

    # ── --production-baseline preset：一鍵套用生產配置 ──
    # CLI 顯式旗標優先：僅在該參數仍為 argparse 預設值時才由 preset 覆蓋。
    # 限制：store_true 旗標（seasonal/liq-weighted/pruned）preset 只會「開啟」不會關閉；
    # 與 parser 預設值相同的顯式輸入（如 --market-filter-min-pos 1）無法與預設區分，會被 preset 覆蓋。
    if getattr(args, "production_baseline", False):
        _pb = production_baseline_overrides()
        if args.topn is None:
            args.topn = _pb["topn"]
        args.enable_seasonal_filter = args.enable_seasonal_filter or _pb["enable_seasonal_filter"]
        # 停損：使用者未顯式指定任何停損旗標時才套 no-stoploss
        if args.stoploss is None and not args.no_stoploss and args.atr_stoploss_multiplier is None:
            args.no_stoploss = _pb["no_stoploss"]
        if args.market_filter_tiers is None:
            args.market_filter_tiers = _pb["market_filter_tiers"]
        if args.market_filter_min_positions == parser.get_default("market_filter_min_positions"):
            args.market_filter_min_positions = _pb["market_filter_min_positions"]
        args.liquidity_weighting = args.liquidity_weighting or _pb["liquidity_weighting"]
        args.pruned_features = args.pruned_features or _pb["pruned_features"]
        if args.min_avg_turnover == parser.get_default("min_avg_turnover"):
            args.min_avg_turnover = _pb["min_avg_turnover"]
        print(f"[production-baseline] 套用生產基準 preset: {_pb}")

    # ── 解析參數（優先命令列，其次 config.yaml）──
    topn = args.topn or config.topn
    # P2-7a：來回成本顯式解析（CLI > config.transaction_cost_round_trip > 單邊 ×4.1）
    cost, _cost_source = resolve_round_trip_cost(args.cost, config)
    if _cost_source == "config.transaction_cost_pct*4.1":
        print(f"[cost] 單邊 {config.transaction_cost_pct:.6f} → 來回 {cost:.6f}（×4.1 稅費倍率）")
    else:
        print(f"[cost] 來回 {cost:.6f}（來源: {_cost_source}）")

    if args.no_stoploss:
        stoploss = 0.0
    elif args.stoploss is not None:
        stoploss = args.stoploss
    else:
        stoploss = config.stoploss_pct

    entry_delay = args.entry_delay_days if args.entry_delay_days is not None else 0  # 原始：當日收盤進場
    risk_free = args.risk_free_rate if args.risk_free_rate is not None else config.backtest_risk_free_rate
    # ── benchmark 成本口徑（2026-07-10 缺陷 3 修復）──
    # 預設 zero_cost（buy-and-hold 近似）；--benchmark-tc X 顯式指定每期扣減值供敏感度分析。
    # 舊 config.backtest_benchmark_with_cost / --no-benchmark-cost 路徑退役（不再假設
    # benchmark 每月 100% 周轉——該假設 120 期複利拖累 ×0.495，虛增超額約 +190pp）。
    # 口徑/訊號解耦：benchmark_tc 只動報表 benchmark_return；market_filter_tiers 一律讀
    # 零成本 market_return_raw（skills/backtest.py），敏感度分析不會改變策略臂持倉決策。
    if args.no_benchmark_cost:
        print("[benchmark] --no-benchmark-cost 已 DEPRECATED（零成本已是預設，旗標為 no-op）")
    benchmark_tc_pct = float(args.benchmark_tc)
    if benchmark_tc_pct > 0:
        print(f"[benchmark] 成本口徑: per_period_tc（每期扣 {benchmark_tc_pct:.6f}，敏感度分析用）")
    else:
        print("[benchmark] 成本口徑: zero_cost（buy-and-hold 近似；預設）")
    ps_method = args.position_sizing_method or getattr(config, "position_sizing_method", "risk_parity")
    atr_mult = args.atr_stoploss_multiplier  # 原始：無 ATR 停損（None unless explicitly set）

    # ── 預設為還原原始基準設定；--baseline 沿用舊實驗模式（feature_cols 切換用）──
    if args.baseline:
        # --baseline 保留舊行為供 Change A/B/C 實驗比較
        sizing = args.position_sizing or "equal"
        trailing = args.trailing_stop_pct  # None = 無移動停利
        time_weighting = False
        enable_complex_filter = False
        enable_slippage = args.slippage and not args.no_slippage
        if args.change_a:
            feature_columns = CHANGE_A_FEATURE_COLS
        else:
            feature_columns = BASELINE_FEATURE_COLS
    else:
        # 預設：還原原始基準（等權、無 trailing、無 slippage、無複雜過濾）
        sizing = args.position_sizing or "equal"
        trailing = args.trailing_stop_pct  # None unless explicitly set
        time_weighting = False
        enable_complex_filter = False
        enable_slippage = args.slippage and not args.no_slippage  # 預設關，需 --slippage 才開
        if args.pruned_features:
            feature_columns = PRUNED_FEATURE_COLS  # SHAP 剪枝：56→48 特徵
        else:
            feature_columns = None  # 使用 DB 全部特徵

    # --topn-floor 在任何模式下均有效
    topn_floor = args.topn_floor

    rebalance_freq = args.rebalance_freq or "M"

    # --seasonal-filter / --no-seasonal-filter 互斥：no 優先
    if args.no_seasonal_filter:
        enable_seasonal_filter = False
    else:
        enable_seasonal_filter = args.enable_seasonal_filter

    # --no-clip：停用 clip（傳入 -1.01，遠低於 -100% 故永遠不觸發）
    clip_loss_pct = -1.01 if args.no_clip else -0.50

    # --entry-filter：進場訊號過濾
    _entry_signal_filter = None
    if args.entry_signal_filter:
        _entry_signal_filter = {}
        for part in args.entry_signal_filter.split(","):
            k, v = part.strip().split("=")
            _entry_signal_filter[k.strip()] = float(v.strip())

    # --filter-group：圓桌策略進場過濾組（預設 A 的條件包含在 B、C 內）
    _FILTER_GROUPS = {
        "A": {"ma_alignment_min": 1.0, "bias_20_max": 0.20},
        "B": {"ma_alignment_min": 1.0, "bias_20_max": 0.20,
              "price_volume_divergence_min": 0.0, "foreign_buy_intensity_max_pct": 0.80},
        "C": {"ma_alignment_min": 1.0, "bias_20_max": 0.20,
              "price_volume_divergence_min": 0.0, "foreign_buy_intensity_max_pct": 0.80,
              "ret_20_rank_min": 0.35, "ret_20_rank_max": 0.75,
              "ret_60_rank_min": 0.30, "volume_surge_ratio_min": 1.5},
    }
    if getattr(args, "filter_group", None):
        if _entry_signal_filter is None:
            _entry_signal_filter = {}
        _entry_signal_filter.update(_FILTER_GROUPS[args.filter_group])

    # --market-filter-tiers：漸進式大盤過濾
    _market_filter_tiers = None
    if args.market_filter_tiers:
        _market_filter_tiers = []
        for part in args.market_filter_tiers.split(","):
            thr, mult = part.strip().split(":")
            _market_filter_tiers.append((float(thr), float(mult)))

    # --momentum-penalty：對高動能特徵乘以 0.5
    momentum_penalty_cols = None
    if args.momentum_penalty:
        momentum_penalty_cols = {"bias_20": 0.5, "ret_5": 0.5, "ret_20": 0.5}

    # ── Stage 9.1：MLflow run context（disabled 時為 no-op）──
    _mlflow_enabled = getattr(args, "mlflow_enabled", False)
    _mlflow_exp = getattr(args, "mlflow_experiment", DEFAULT_EXPERIMENT)
    _mlflow_run_name = getattr(args, "mlflow_run_name", None)

    with get_session() as session, start_mlflow_run(
        experiment_name=_mlflow_exp,
        run_name=_mlflow_run_name,
        enabled=_mlflow_enabled,
    ) as _mlflow_run:
        # log CLI args 作為 params（過濾 mlflow 自身 + sensitive）
        if _mlflow_run is not None:
            _cli_params = {
                k: v for k, v in vars(args).items()
                if not k.startswith("mlflow_") and k not in ("output",)
            }
            log_params(_cli_params, mlflow_run=_mlflow_run)

        # ── 資料快照身分（實驗可重現性記錄，2026-06-22 危害防護）──
        # 回測資料一律經 skills.data_store parquet cache 載入（backtest.prepare 內）；
        # 開跑前先記一次，跑完再記一次——若兩者不同代表 cache 在本次 run 中被重建，
        # 正是「同指令跑出 Sharpe 1.035 / 0.805」事故的機制，必須留痕。
        try:
            _snap_pre = data_store.snapshot_info()
        except Exception as _snap_exc:  # snapshot 失敗不阻斷回測
            _snap_pre = {"error": str(_snap_exc)}
        print(f"[snapshot] pre-run data snapshot: {json.dumps(_snap_pre, ensure_ascii=False, default=str)}")

        result = run_backtest(
            config=config,
            db_session=session,
            backtest_months=args.months,
            retrain_freq_months=args.retrain_freq,
            topn=topn,
            stoploss_pct=stoploss,
            transaction_cost_pct=cost,
            entry_delay_days=entry_delay,
            risk_free_rate=risk_free,
            benchmark_tc_pct=benchmark_tc_pct,
            position_sizing=sizing,
            position_sizing_method=ps_method,
            trailing_stop_pct=trailing,
            atr_stoploss_multiplier=atr_mult,
            atr_period=args.atr_period,
            fast_mode=args.fast,
            enable_slippage=enable_slippage,
            feature_columns=feature_columns,
            time_weighting=time_weighting,
            enable_complex_filter=enable_complex_filter,
            enable_seasonal_filter=enable_seasonal_filter,
            topn_floor=topn_floor,
            rebalance_freq=rebalance_freq,
            train_lookback_days=args.train_lookback_days,
            clip_loss_pct=clip_loss_pct,
            cap_daily_return_pct=args.cap_daily_return_pct,
            enable_breakthrough_entry=args.enable_breakthrough_entry,
            breakthrough_max_wait=args.breakthrough_max_wait,
            momentum_penalty_cols=momentum_penalty_cols,
            atr_dynamic_stoploss=args.atr_dynamic_stoploss,
            market_filter=args.market_filter,
            market_filter_tiers=_market_filter_tiers,
            market_filter_min_positions=args.market_filter_min_positions,
            entry_signal_filter=_entry_signal_filter,
            min_avg_turnover=args.min_avg_turnover,
            enable_tiered_slippage=args.tiered_slippage,
            liquidity_weighting=args.liquidity_weighting,
            portfolio_circuit_breaker_pct=(
                -abs(args.portfolio_circuit_breaker_pct)
                if args.portfolio_circuit_breaker_pct is not None
                else None
            ),
            label_type="excess" if args.excess_label else "abs",
            use_lambdarank=getattr(args, "use_lambdarank", False),
            cross_section_normalize=getattr(args, "cross_section_normalize", False),
            ensemble_n_checkpoints=getattr(args, "ensemble_n_checkpoints", 1),
            vol_target_pct=getattr(args, "vol_target_pct", 0.0),
            vol_target_lookback_days=getattr(args, "vol_target_lookback_days", 60),
            use_stacking=getattr(args, "use_stacking", False),
            stacking_val_frac=getattr(args, "stacking_val_frac", 0.20),
            recent_dd_skip_pct=getattr(args, "recent_dd_skip_pct", 0.0),
            max_per_sector=getattr(args, "max_per_sector", 0),
        )

        # ── 資料快照（run 後）寫入結果 JSON ──
        try:
            _snap_post = data_store.snapshot_info()
        except Exception as _snap_exc:
            _snap_post = {"error": str(_snap_exc)}
        result["data_snapshot"] = {
            # BacktestPipeline.prepare() 一律經 skills.data_store parquet cache 載入
            "data_source": "data_store_parquet",
            "pre_run": _snap_pre,
            "post_run": _snap_post,
            "cache_rebuilt_during_run": _snap_pre != _snap_post,
        }
        print(f"[snapshot] post-run data snapshot: {json.dumps(_snap_post, ensure_ascii=False, default=str)}")
        if _snap_pre != _snap_post:
            print("[snapshot] ⚠️ 資料快照在本次 run 期間改變（cache 重建）——"
                  "本結果與同指令其他 run 不可直接比較（可重現性危害）")

        _summary = result.get("summary") or {}

        # ── 統計紀律工具化（2026-07-10 缺陷 6 規則 1）──
        # 1) trial registry append（每次 run 都算一次 trial，不論結果）
        # 2) 月報酬 paired block-bootstrap Sharpe 95% CI + DSR p-value
        if _summary:
            try:
                _reg_record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "command": " ".join(str(a) for a in sys.argv),
                    "source": "run_backtest_cli",
                    "sharpe": _summary.get("sharpe_ratio"),
                    "months": args.months,
                    "data_snapshot_summary": _snap_post,
                }
                _reg_count = append_trial_registry(_reg_record)
                _n_trials = _reg_count + HISTORICAL_TRIALS_BASE
                print(f"[trial-registry] 已記錄第 {_reg_count} 筆 → {TRIAL_REGISTRY_PATH}"
                      f"（DSR n_trials = {_reg_count} + {HISTORICAL_TRIALS_BASE} = {_n_trials}）")
            except Exception as _reg_exc:
                print(f"[trial-registry] 寫入失敗（統計區塊改用歷史基數）: {_reg_exc}")
                _n_trials = registry_trial_count() + HISTORICAL_TRIALS_BASE

            try:
                _stats = compute_statistics_block(result, n_trials=_n_trials, risk_free_rate=risk_free)
                result["summary"]["statistics"] = _stats
                if _stats:
                    _ci = _stats["sharpe_ci_95"]
                    _dsr = _stats["deflated_sharpe"]
                    print(f"\n── 統計紀律（缺陷 6 規則 1）──")
                    print(f"  Sharpe 95% CI（block-bootstrap, block={_ci['block_size']}, "
                          f"B={_ci['n_boot']}）: {_ci['sharpe_observed']:.3f} "
                          f"[{_ci['ci_low']:.3f}, {_ci['ci_high']:.3f}]")
                    print(f"  Excess Sharpe 95% CI: {_ci['excess_sharpe_observed']:.3f} "
                          f"[{_ci['excess_ci_low']:.3f}, {_ci['excess_ci_high']:.3f}]")
                    print(f"  DSR p-value: {_dsr['p_value']:.4f} "
                          f"(n_trials={_dsr['n_trials']}, "
                          f"{'SIGNIFICANT' if _dsr['is_significant_5pct'] else 'NOT significant'} @5%)")
            except Exception as _stats_exc:
                print(f"[statistics] 統計區塊計算失敗（欄位為 null，不阻斷）: {_stats_exc}")
                result["summary"]["statistics"] = None

            # ── TR 指數對照臂（機會成本；fetch 失敗 → null，不阻斷）──
            # 口徑前提（skills/taiex_tr.py docstring）：策略側必須是含息 P&L
            # （BACKTEST_ADJ_PRICE_PARQUET 還原價 overlay）。未設定時策略側少掉
            # 全部股息（台股約年化 3-4pp）卻與含息 TR 指數相減 → 混合口徑，
            # 系統性低估策略——直接設 null 拒算，不產生看似可比的假數字。
            _adj_parquet_env = os.getenv("BACKTEST_ADJ_PRICE_PARQUET")
            if not _adj_parquet_env:
                print("[taiex_tr] vs_taiex_tr = null：未設 BACKTEST_ADJ_PRICE_PARQUET，"
                      "策略 P&L 為 raw close（不含息），與含息 TR 指數混合口徑不可比；"
                      "要啟用對照請帶 BACKTEST_ADJ_PRICE_PARQUET=artifacts/adj_prices/adj_prices_10y.parquet")
                _vs_tr = None
            else:
                try:
                    from skills.taiex_tr import compute_vs_taiex_tr
                    _vs_tr = compute_vs_taiex_tr(result.get("equity_curve") or [])
                    if _vs_tr:
                        # 口徑標記：事後讀 JSON 的人可直接確認兩側皆含息
                        _vs_tr["pnl_convention"] = "adjusted_close_total_return"
                        _vs_tr["adj_price_parquet"] = _adj_parquet_env
                except Exception as _tr_exc:
                    print(f"[taiex_tr] vs_taiex_tr 失敗（欄位為 null）: {_tr_exc}")
                    _vs_tr = None
            result["summary"]["vs_taiex_tr"] = _vs_tr
            if _vs_tr:
                print(f"\n── vs 發行量加權股價報酬指數（TAIEX TR，含息機會成本）──")
                print(f"  策略年化 {_vs_tr['strategy_annualized_return']:+.2%}  "
                      f"TR 指數年化 {_vs_tr['taiex_tr_annualized_return']:+.2%}  "
                      f"年化超額 {_vs_tr['excess_annualized_vs_tr']:+.2%}")
                print(f"  （窗口 {_vs_tr['window']['start']} ~ {_vs_tr['window']['end']}，"
                      f"TR 取值 {_vs_tr['window']['tr_start_date']} ~ {_vs_tr['window']['tr_end_date']}）")
            else:
                print("[taiex_tr] vs_taiex_tr = null（未設 BACKTEST_ADJ_PRICE_PARQUET / "
                      "fetch 失敗 / 快取不涵蓋回測窗——原因見上方訊息）")

        # ── Stage 10.6：beta-hedge 後處理 ──
        _hedge_ratio = getattr(args, "hedge_ratio", 0.0)
        if _hedge_ratio > 0 and result.get("periods"):
            _hm = compute_hedged_metrics(result, _hedge_ratio)
            print(f"\n── Stage 10.6 Beta-Hedge (ratio={_hedge_ratio:.2f}) ──")
            print(f"  策略 OLS beta vs 大盤 = {_hm.get('ols_beta'):.4f}, corr = {_hm.get('ols_corr'):.4f}")
            print(f"  Hedged Sharpe:    {_hm.get('hedged_sharpe'):+.4f}")
            print(f"  Hedged MDD:       {_hm.get('hedged_mdd'):+.4f}")
            print(f"  Hedged Calmar:    {_hm.get('hedged_calmar'):+.4f}")
            print(f"  Hedged Annual:    {_hm.get('hedged_annual'):+.2%}")
            print(f"  Hedged Cum:       {_hm.get('hedged_cum'):+.2%}")
            # 寫進 result 供 mlflow + JSON
            result["hedge_metrics"] = _hm

        # ── Stage 9.1：寫入 mlflow metrics / artifacts ──
        log_backtest_result(result, mlflow_run=_mlflow_run)

    # ── 輸出 JSON ──
    output_path = args.output
    if output_path is None:
        artifacts_dir = ROOT / "artifacts" / "backtest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(artifacts_dir / f"backtest_{ts}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果已儲存: {output_path}")

    # ── 實驗紀錄 ──
    if args.log_experiment:
        _log_experiment(result, args, sys.argv)


def _log_experiment(result: dict, args, argv: list) -> None:
    """將實驗結果寫入 artifacts/experiments/ 目錄，供跨次比較。

    檔案格式：artifacts/experiments/<YYYYMMDD_HHMMSS>_<name>.json
    """
    summary = result.get("summary", {})
    periods = result.get("periods", [])

    # ── 逐年報酬（依 period exit_date 年份歸組）──
    yearly: dict[str, dict] = {}
    for p in periods:
        try:
            yr = str(p.get("exit_date", ""))[:4]
            if not yr.isdigit():
                continue
            strat_r = float(p.get("return", 0.0))
            bm_r    = float(p.get("benchmark_return", 0.0))
            if yr not in yearly:
                yearly[yr] = {"strategy_cum": 1.0, "benchmark_cum": 1.0}
            yearly[yr]["strategy_cum"]  *= (1 + strat_r)
            yearly[yr]["benchmark_cum"] *= (1 + bm_r)
        except Exception:
            continue

    yearly_returns = {
        yr: {
            "strategy":  round(v["strategy_cum"] - 1, 4),
            "benchmark": round(v["benchmark_cum"] - 1, 4),
            "excess":    round(v["strategy_cum"] - v["benchmark_cum"], 4),
        }
        for yr, v in sorted(yearly.items())
    }

    # ── CLI 摘要（去掉程式路徑，只留關鍵 flags）──
    cli_str = " ".join(str(a) for a in argv[1:])   # 去掉 scripts/run_backtest.py

    # ── 組裝 experiment record ──
    exp_name = args.experiment_name or "exp"
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    record   = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "name":            exp_name,
        "cli":             cli_str,
        "params": {
            "months":                  args.months,
            "stoploss":                summary.get("config", {}).get("stoploss_pct"),
            "seasonal_filter":         args.enable_seasonal_filter,
            "market_filter_tiers":     args.market_filter_tiers,
            "market_filter_min_pos":   args.market_filter_min_positions,
            "liq_weighted":            args.liquidity_weighting,
            "pruned_features":         getattr(args, "pruned_features", False),
            "rebalance_freq":          args.rebalance_freq or "M",
            "entry_delay_days":        summary.get("config", {}).get("entry_delay_days"),
            "breakthrough_entry":      getattr(args, "enable_breakthrough_entry", False),
        },
        "metrics": {
            "total_return":            summary.get("total_return"),
            "annualized_return":       summary.get("annualized_return"),
            "benchmark_total_return":  summary.get("benchmark_total_return"),
            "excess_return":           summary.get("excess_return"),
            "max_drawdown":            summary.get("max_drawdown"),
            "sharpe_ratio":            summary.get("sharpe_ratio"),
            "calmar_ratio":            summary.get("calmar_ratio"),
            "win_rate":                summary.get("win_rate"),
            "profit_factor":           summary.get("profit_factor"),
            "total_trades":            summary.get("total_trades"),
            "total_periods":           summary.get("total_periods"),
            "backtest_start":          str(summary.get("backtest_start", "")),
            "backtest_end":            str(summary.get("backtest_end", "")),
        },
        "yearly_returns": yearly_returns,
    }

    exp_dir = ROOT / "artifacts" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = exp_name.replace("/", "_").replace(" ", "_")[:40]
    exp_path  = exp_dir / f"{ts}_{safe_name}.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    # ── 終端摘要列印 ──
    m = record["metrics"]
    print(f"\n{'═'*55}")
    print(f"📋 實驗紀錄：{exp_name}")
    print(f"   期間：{m['backtest_start']} ~ {m['backtest_end']}")
    print(f"   累積  {m['total_return']:>+8.2%}  大盤 {m['benchmark_total_return']:>+8.2%}"
          f"  超額 {m['excess_return']:>+8.2%}")
    print(f"   年化  {m['annualized_return']:>+8.2%}  MDD  {m['max_drawdown']:>+8.2%}"
          f"  Sharpe {m['sharpe_ratio']:>6.3f}")
    print(f"   Calmar {(m['calmar_ratio'] or 0):>6.3f}  勝率 {m['win_rate']:>6.2%}")
    print(f"\n   逐年報酬（策略 vs 大盤）：")
    for yr, v in yearly_returns.items():
        sign = "▲" if v["strategy"] >= 0 else "▼"
        print(f"   {yr}: {sign} {v['strategy']:>+8.2%}  大盤 {v['benchmark']:>+8.2%}"
              f"  超額 {v['excess']:>+8.2%}")
    print(f"\n   紀錄已儲存: {exp_path}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
