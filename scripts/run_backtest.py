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

    # ── 速度 ──
    parser.add_argument("--fast", action="store_true",
                        help="快速模式：LightGBM 樹數 500→150，加速 ~3x（精度略降）")

    # ── 輸出 ──
    parser.add_argument("--output", type=str, default=None,
                        help="結果輸出 JSON 路徑")

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

    entry_delay = args.entry_delay_days if args.entry_delay_days is not None else config.backtest_entry_delay_days
    risk_free = args.risk_free_rate if args.risk_free_rate is not None else config.backtest_risk_free_rate
    benchmark_with_cost = not args.no_benchmark_cost and config.backtest_benchmark_with_cost
    sizing = args.position_sizing or config.backtest_position_sizing
    ps_method = args.position_sizing_method or getattr(config, "position_sizing_method", "risk_parity")
    trailing = args.trailing_stop_pct if args.trailing_stop_pct is not None else config.backtest_trailing_stop_pct
    atr_mult = args.atr_stoploss_multiplier if args.atr_stoploss_multiplier is not None else config.backtest_atr_stoploss_multiplier

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


if __name__ == "__main__":
    main()
