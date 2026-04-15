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
    python scripts/run_backtest.py --no-benchmark-cost          # Benchmark 不含成本

10y 逐步優化實驗（每次只改一個變數）:
    python scripts/run_backtest.py --months 120 --baseline      # 乾淨基準（無時間加權/無複雜過濾）
    python scripts/run_backtest.py --months 120 --baseline --change-a   # Change A: +IC 特徵
    python scripts/run_backtest.py --months 120 --baseline --topn-floor 5  # Change B: topN floor=5
    python scripts/run_backtest.py --months 120 --baseline --slippage  # Change C: 滑價模型
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from skills.backtest import run_backtest
from skills.build_features import BASELINE_FEATURE_COLS, CHANGE_A_FEATURE_COLS, PRUNED_FEATURE_COLS


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward 回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 基礎參數 ──
    parser.add_argument("--months", type=int, default=24, help="回測月數 (預設 24)")
    parser.add_argument("--retrain-freq", type=int, default=3, help="模型重訓頻率（月，預設 3）")
    parser.add_argument("--topn", type=int, default=None, help="每期選股數量 (預設使用 config)")
    parser.add_argument("--cost", type=float, default=None, help="來回交易成本 (如 0.00585)")

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
                        help="Benchmark 不套用交易成本（舊行為，不建議）")

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

    # ── 解析參數（優先命令列，其次 config.yaml）──
    topn = args.topn or config.topn
    cost = args.cost if args.cost is not None else (
        config.transaction_cost_pct * 4.1  # 單邊→來回（0.1425%×2 + 0.3%稅）
        if config.transaction_cost_pct < 0.005 else config.transaction_cost_pct
    )

    if args.no_stoploss:
        stoploss = 0.0
    elif args.stoploss is not None:
        stoploss = args.stoploss
    else:
        stoploss = config.stoploss_pct

    entry_delay = args.entry_delay_days if args.entry_delay_days is not None else 0  # 原始：當日收盤進場
    risk_free = args.risk_free_rate if args.risk_free_rate is not None else config.backtest_risk_free_rate
    benchmark_with_cost = not args.no_benchmark_cost and config.backtest_benchmark_with_cost
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

    with get_session() as session:
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
            benchmark_with_cost=benchmark_with_cost,
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
        )

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
