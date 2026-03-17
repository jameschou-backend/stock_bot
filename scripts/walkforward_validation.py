#!/usr/bin/env python3
"""Strategy C Label-10 嚴格 Walk-Forward 驗證

固定 24 個月訓練視窗 × 6 個月不重疊測試視窗。
每個 Fold 訓練一次模型，評估期間不再重訓（嚴格 OOS）。

Usage:
    python scripts/walkforward_validation.py                         # Folds 1-8，含大盤過濾
    python scripts/walkforward_validation.py --no-market-filter      # Folds 1-8，無大盤過濾
    python scripts/walkforward_validation.py --fold-start 9          # Folds 9-14（擴展）
    python scripts/walkforward_validation.py --output artifacts/walkforward_audit.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from scripts.backtest_rotation import run_rotation

# ── Folds 1-8（原始，2018-2021）──
FOLDS = [
    {"fold": 1, "train_start": date(2016, 1, 1),  "train_end": date(2017, 12, 31),
               "test_start":  date(2018, 1, 1),   "test_end":  date(2018, 6, 30)},
    {"fold": 2, "train_start": date(2016, 7, 1),  "train_end": date(2018, 6, 30),
               "test_start":  date(2018, 7, 1),   "test_end":  date(2018, 12, 31)},
    {"fold": 3, "train_start": date(2017, 1, 1),  "train_end": date(2018, 12, 31),
               "test_start":  date(2019, 1, 1),   "test_end":  date(2019, 6, 30)},
    {"fold": 4, "train_start": date(2017, 7, 1),  "train_end": date(2019, 6, 30),
               "test_start":  date(2019, 7, 1),   "test_end":  date(2019, 12, 31)},
    {"fold": 5, "train_start": date(2018, 1, 1),  "train_end": date(2019, 12, 31),
               "test_start":  date(2020, 1, 1),   "test_end":  date(2020, 6, 30)},
    {"fold": 6, "train_start": date(2018, 7, 1),  "train_end": date(2020, 6, 30),
               "test_start":  date(2020, 7, 1),   "test_end":  date(2020, 12, 31)},
    {"fold": 7, "train_start": date(2019, 1, 1),  "train_end": date(2020, 12, 31),
               "test_start":  date(2021, 1, 1),   "test_end":  date(2021, 6, 30)},
    {"fold": 8, "train_start": date(2019, 7, 1),  "train_end": date(2021, 6, 30),
               "test_start":  date(2021, 7, 1),   "test_end":  date(2021, 12, 31)},
    # ── Folds 9-14（擴展，2022-2024）──
    {"fold": 9,  "train_start": date(2020, 1, 1),  "train_end": date(2021, 12, 31),
                "test_start":  date(2022, 1, 1),   "test_end":  date(2022, 6, 30)},
    {"fold": 10, "train_start": date(2020, 7, 1),  "train_end": date(2022, 6, 30),
                "test_start":  date(2022, 7, 1),   "test_end":  date(2022, 12, 31)},
    {"fold": 11, "train_start": date(2021, 1, 1),  "train_end": date(2022, 12, 31),
                "test_start":  date(2023, 1, 1),   "test_end":  date(2023, 6, 30)},
    {"fold": 12, "train_start": date(2021, 7, 1),  "train_end": date(2023, 6, 30),
                "test_start":  date(2023, 7, 1),   "test_end":  date(2023, 12, 31)},
    {"fold": 13, "train_start": date(2022, 1, 1),  "train_end": date(2023, 12, 31),
                "test_start":  date(2024, 1, 1),   "test_end":  date(2024, 6, 30)},
    {"fold": 14, "train_start": date(2022, 7, 1),  "train_end": date(2024, 6, 30),
                "test_start":  date(2024, 7, 1),   "test_end":  date(2024, 12, 31)},
]

# Strategy A Exp D 逐年報酬（供同期對比用）
STRATEGY_A_YEARLY = {
    2016: 0.0529, 2017: 0.4471, 2018: -0.0274, 2019: 0.3327,
    2020: 1.3174, 2021: 0.7509, 2022: 0.0275, 2023: 1.1102,
    2024: 0.3105, 2025: 0.0870, 2026: 0.0784,
}


def _approx_strategy_a_half_year(test_start: date, test_end: date) -> str:
    """粗略估算 Strategy A 在測試期的報酬（從年度數據推算）。"""
    yr = test_start.year
    annual = STRATEGY_A_YEARLY.get(yr, 0.0)
    # 上/下半年近似為年報酬的一半（幾何平均近似）
    half = (1 + annual) ** 0.5 - 1
    return f"~{half:+.1%}（年度 {annual:+.1%} 的估算半年值）"


def main():
    parser = argparse.ArgumentParser(description="Strategy C Walk-Forward Validation")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "artifacts" / "walkforward_audit.md"))
    parser.add_argument("--transaction-cost", type=float, default=0.00585,
                        help="每筆交易成本（預設台股真實 0.585%%）")
    parser.add_argument("--fast", action="store_true", help="快速模式（減少 estimators）")
    parser.add_argument("--no-market-filter", action="store_true",
                        help="停用大盤過濾（對比用）")
    parser.add_argument("--fold-start", type=int, default=1,
                        help="開始 Fold 編號（預設 1）")
    parser.add_argument("--fold-end", type=int, default=8,
                        help="結束 Fold 編號（預設 8；擴展版用 14）")
    args = parser.parse_args()

    config = load_config()
    mf_tiers = None if args.no_market_filter else [(-0.05, 0.5), (-0.10, 0.33), (-0.15, 0.17)]

    # 根據 fold-start / fold-end 篩選要執行的 folds
    active_folds = [f for f in FOLDS if args.fold_start <= f["fold"] <= args.fold_end]
    total_folds = len(active_folds)

    results = []
    with get_session() as session:
        for fold_def in active_folds:
            fold_n = fold_def["fold"]
            train_start = fold_def["train_start"]
            train_end   = fold_def["train_end"]
            test_start  = fold_def["test_start"]
            test_end    = fold_def["test_end"]

            print(f"\n{'='*60}")
            print(f"Fold {fold_n}：訓練 {train_start}~{train_end} | 測試 {test_start}~{test_end}")
            print(f"{'='*60}")

            result = run_rotation(
                config=config,
                db_session=session,
                backtest_months=120,            # 不影響（eval_period 覆蓋）
                rank_threshold=0.20,
                max_hold_days=30,
                transaction_cost_pct=args.transaction_cost,
                train_label_horizon=10,
                label_horizon_buffer=20,
                fast_mode=args.fast,
                market_filter_tiers=mf_tiers,
                # 固定視窗參數
                fixed_train_start=train_start,
                fixed_train_end=train_end,
                eval_period_start=test_start,
                eval_period_end=test_end,
            )

            s = result["summary"]
            results.append({
                "fold": fold_n,
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start,   "test_end": test_end,
                "total_return": s["total_return"],
                "annualized_return": s["annualized_return"],
                "max_drawdown": s["max_drawdown"],
                "sharpe": s["sharpe_ratio"],
                "calmar": s["calmar_ratio"],
                "win_rate": s["win_rate"],
                "total_trades": s["total_trades"],
                "avg_hold": s["avg_hold_days"],
                "exit_reasons": s["exit_reasons"],
            })
            r = results[-1]
            print(f"  → 報酬 {r['total_return']:+.2%} | Sharpe {r['sharpe']:.3f} | "
                  f"MDD {r['max_drawdown']:.2%} | 勝率 {r['win_rate']:.1%} | {r['total_trades']} 筆")

    # ── 判定結果 ──
    n_pass = sum(1 for r in results if r["sharpe"] > 1.0)
    n_positive = sum(1 for r in results if r["total_return"] > 0)
    pass_threshold_hi = max(6, round(total_folds * 0.75))
    pass_threshold_lo = max(5, round(total_folds * 0.625))
    if n_pass >= pass_threshold_hi:
        verdict = "✅ 可信"
        verdict_detail = f"{n_pass}/{total_folds} 個 Fold Sharpe > 1.0，策略在嚴格 OOS 下表現穩定"
    elif n_pass >= pass_threshold_lo:
        verdict = "🟠 存疑"
        verdict_detail = f"{n_pass}/{total_folds} 個 Fold Sharpe > 1.0，需進一步驗證"
    else:
        verdict = "❌ 過擬合風險高"
        verdict_detail = f"僅 {n_pass}/{total_folds} 個 Fold Sharpe > 1.0，in-sample 過擬合可能性大"

    mf_label = "無大盤過濾" if args.no_market_filter else "含大盤過濾（-5%:×0.5, -10%:×0.33, -15%:×0.17）"
    avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
    avg_return = sum(r["total_return"] for r in results) / len(results)
    avg_mdd = sum(r["max_drawdown"] for r in results) / len(results)

    # ── 產出 Markdown ──
    lines = [
        "# Strategy C Label-10 嚴格 Walk-Forward 驗證報告",
        "",
        f"> 產出日期：2026-03-17",
        f"> 驗證配置：`--train-label-horizon 10 --label-horizon-buffer 20`，真實成本 {args.transaction_cost:.3%}",
        f"> 大盤過濾：{mf_label}",
        f"> Fold 範圍：{args.fold_start} ~ {args.fold_end}",
        f"> 驗證方法：固定 24 個月訓練視窗 × 6 個月不重疊測試視窗，訓練一次不重訓",
        "",
        "---",
        "",
        f"## 最終判定：{verdict}",
        "",
        f"**{verdict_detail}**",
        "",
        f"| 統計 | 數值 |",
        f"|------|------|",
        f"| 通過 Fold（Sharpe > 1.0）| {n_pass} / {total_folds} |",
        f"| 正報酬 Fold | {n_positive} / {total_folds} |",
        f"| 平均 Sharpe | {avg_sharpe:.3f} |",
        f"| 平均 6M 報酬 | {avg_return:+.2%} |",
        f"| 平均 MDD | {avg_mdd:.2%} |",
        "",
        "---",
        "",
        "## 逐 Fold 詳細結果",
        "",
        "| Fold | 訓練期 | 測試期 | 報酬 | Sharpe | MDD | 勝率 | 交易 | 判定 |",
        "|------|--------|--------|------|--------|-----|------|------|------|",
    ]

    for r in results:
        pass_icon = "✅" if r["sharpe"] > 1.0 else ("🟠" if r["sharpe"] > 0.5 else "❌")
        lines.append(
            f"| {r['fold']} "
            f"| {r['train_start']}~{r['train_end']} "
            f"| {r['test_start']}~{r['test_end']} "
            f"| {r['total_return']:+.1%} "
            f"| {r['sharpe']:.3f} "
            f"| {r['max_drawdown']:.1%} "
            f"| {r['win_rate']:.1%} "
            f"| {r['total_trades']} "
            f"| {pass_icon} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 出場原因分析",
        "",
        "| Fold | Rank Drop | Max Hold | 其他 |",
        "|------|----------|---------|------|",
    ]
    for r in results:
        er = r["exit_reasons"]
        total = max(r["total_trades"], 1)
        rd = er.get("Rank Drop", 0)
        mh = er.get("Max Hold Days", 0)
        oth = total - rd - mh
        lines.append(
            f"| {r['fold']} | {rd/total:.1%} | {mh/total:.1%} | {oth/total:.1%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Strategy A 同期對比（半年估算）",
        "",
        "| Fold | 測試期 | Strategy C | Strategy A（估算）|",
        "|------|--------|-----------|-----------------|",
    ]
    for r in results:
        a_approx = _approx_strategy_a_half_year(r["test_start"], r["test_end"])
        lines.append(
            f"| {r['fold']} | {r['test_start']}~{r['test_end']} "
            f"| {r['total_return']:+.1%} | {a_approx} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 驗證邏輯說明",
        "",
        "- **為什麼固定視窗更嚴格**：現行全量回測（expanding window）每次訓練用所有歷史，",
        "  容易隱式記住歷史特徵-報酬的整體分佈。固定 24 個月視窗強制模型只能學習",
        "  「那段時期的市場特性」，對未來 6 個月做真正的 OOS 預測。",
        "",
        "- **判定標準**：",
        "  - ✅ 可信：≥6 個 Fold Sharpe > 1.0",
        "  - 🟠 存疑：5 個 Fold Sharpe > 1.0",
        "  - ❌ 過擬合：< 4 個 Fold Sharpe > 1.0",
        "",
        "- **局限性**：此驗證涵蓋 2018-2024 市場環境（未涵蓋 2025-2026），",
        "  後段的 OOS 性能仍需持續監控。",
    ]

    output_text = "\n".join(lines)
    Path(args.output).write_text(output_text, encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Walk-Forward 驗證完成")
    print(f"{'='*60}")
    print(f"判定：{verdict}")
    print(f"{verdict_detail}")
    print(f"平均 Sharpe：{avg_sharpe:.3f} | 平均報酬：{avg_return:+.2%}")
    print(f"\n報告已存至：{args.output}")


if __name__ == "__main__":
    main()
