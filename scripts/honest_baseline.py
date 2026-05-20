"""誠實基準評估：對現行 backtest 結果跑 DSR 與 PBO。

目的：把「跑過 80+ 次實驗 + 看到 Sharpe 0.95」這種結論校正為「考慮 multiple
testing bias 後，真實 alpha 還剩多少」。建立後續所有實驗（Stage 3-9）的對照標準。

兩個分析：
  1. DSR (Deflated Sharpe Ratio)：對指定的 backtest result，給定試驗總數 N，
     計算單尾 p-value。p > 0.95 = 在 5% 顯著水準下確有正 alpha。
  2. PBO (Probability of Backtest Overfit)：對 artifacts/backtest/*.json 內
     多個策略候選的 monthly returns 合併成 matrix，計算「train 期最佳在 test
     期低於 median」的比例。pbo > 0.5 = 系統性 overfit。

用法：
  # 對最新 backtest result 跑 DSR
  python scripts/honest_baseline.py

  # 指定 backtest 檔 + 自訂試驗總數
  python scripts/honest_baseline.py --backtest artifacts/backtest/vol_filter_baseline.json --n-trials 80

  # 也跑 PBO（讀所有 artifacts/backtest/*.json）
  python scripts/honest_baseline.py --pbo

  # JSON 輸出（供 CI / dashboard 解析）
  python scripts/honest_baseline.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.statistics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfit,
    returns_moments,
    sharpe_from_returns,
)


BACKTEST_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "backtest"


def _load_backtest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _periods_to_returns(backtest_data: dict) -> np.ndarray:
    """從 backtest JSON 取 monthly returns 序列。"""
    periods = backtest_data.get("periods") or []
    return np.array([p.get("return", 0.0) for p in periods], dtype=float)


def _latest_backtest_file() -> Optional[Path]:
    if not BACKTEST_DIR.exists():
        return None
    files = sorted(BACKTEST_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def estimate_sr_std_from_results(
    min_periods: int = 100,
    glob_pattern: str = "*.json",
    exclude_patterns: Optional[list] = None,
) -> Optional[float]:
    """從 artifacts/backtest 估「同類試驗的 Sharpe 分散度」（DSR 校正用）。

    若 backtest 檔案 < 2 個則回 None，由 caller 決定 fallback 值。

    Args:
        glob_pattern: 包含哪些檔案（預設 *.json）
        exclude_patterns: 子字串清單，命中即排除（如 'rotation_' 排除 Strategy C/D）
    """
    if not BACKTEST_DIR.exists():
        return None
    exclude_patterns = exclude_patterns or []
    sharpes = []
    for f in BACKTEST_DIR.glob(glob_pattern):
        if any(p in f.name for p in exclude_patterns):
            continue
        try:
            d = _load_backtest(f)
            n = len(d.get("periods") or [])
            sr = d.get("summary", {}).get("sharpe_ratio")
            if n >= min_periods and sr is not None:
                sharpes.append(float(sr))
        except Exception:
            pass
    if len(sharpes) < 2:
        return None
    return float(np.std(sharpes, ddof=1))


def run_dsr(
    backtest_path: Path,
    n_trials: int,
    sr_estimates_std: Optional[float] = None,
    exclude_patterns: Optional[list] = None,
) -> dict:
    data = _load_backtest(backtest_path)
    returns = _periods_to_returns(data)

    if len(returns) < 24:
        return {
            "file": str(backtest_path),
            "error": f"too few periods ({len(returns)}) for reliable DSR",
        }

    moments = returns_moments(returns)
    sr_observed = sharpe_from_returns(returns, periods_per_year=12)

    # 同步呈現原始 summary（如果有）
    summary = data.get("summary", {})
    sr_from_summary = summary.get("sharpe_ratio")

    # 估 SR 標準差（若未指定，從實際 backtest 結果估）
    if sr_estimates_std is None:
        sr_estimates_std = estimate_sr_std_from_results(exclude_patterns=exclude_patterns) or 1.0
    sr_estimates_std = float(sr_estimates_std)

    result = deflated_sharpe_ratio(
        sr_observed=sr_observed,
        n_trials=n_trials,
        n_observations=len(returns),
        skewness=moments["skewness"],
        kurtosis=moments["kurtosis"],
        sr_estimates_std=sr_estimates_std,
    )

    return {
        "file": str(backtest_path),
        "n_periods": int(moments["n"]),
        "returns_mean": moments["mean"],
        "returns_std": moments["std"],
        "skewness": moments["skewness"],
        "kurtosis": moments["kurtosis"],
        "sharpe_from_periods": sr_observed,
        "sharpe_from_summary": sr_from_summary,
        "n_trials": n_trials,
        "sr_estimates_std": sr_estimates_std,
        "sr_expected_under_null": result.sr_expected_under_null,
        "dsr_p_value": result.p_value,
        "is_significant_5pct": result.is_significant_5pct,
    }


def run_pbo(min_periods: int = 24, n_splits: int = 16) -> dict:
    """對 artifacts/backtest/*.json 中所有「期數 >= min_periods」的策略合併成 matrix 跑 PBO。"""
    if not BACKTEST_DIR.exists():
        return {"error": f"{BACKTEST_DIR} 不存在"}

    candidates = []
    for f in sorted(BACKTEST_DIR.glob("*.json")):
        try:
            d = _load_backtest(f)
            r = _periods_to_returns(d)
            if len(r) >= min_periods:
                candidates.append((f.name, r))
        except Exception as exc:
            print(f"[skip] {f.name}: {exc}")

    if len(candidates) < 2:
        return {"error": f"too few backtest files (need >= 2, got {len(candidates)})"}

    # 對齊到相同最小長度
    min_len = min(len(r) for _, r in candidates)
    matrix = np.column_stack([r[-min_len:] for _, r in candidates])

    result = probability_of_backtest_overfit(matrix, n_splits=n_splits)
    return {
        "strategy_files": [name for name, _ in candidates],
        "n_strategies": result.n_strategies,
        "n_samples": result.n_samples,
        "n_splits": n_splits,
        "n_combinations": result.n_combinations,
        "overfit_count": result.overfit_count,
        "pbo": result.pbo,
        "verdict": (
            "嚴重 overfit" if result.pbo > 0.7
            else "明顯 overfit" if result.pbo > 0.5
            else "可接受" if result.pbo > 0.3
            else "良好"
        ),
    }


def _format_dsr_human(d: dict) -> str:
    if d.get("error"):
        return f"DSR error: {d['error']}"
    lines = [
        f"  檔案: {d['file']}",
        f"  期數: {d['n_periods']} (建議 >= 24)",
        f"  Returns mean / std: {d['returns_mean']:.4f} / {d['returns_std']:.4f}",
        f"  Skewness / Kurtosis: {d['skewness']:.3f} / {d['kurtosis']:.3f}",
        f"  Sharpe (from periods): {d['sharpe_from_periods']:.3f}",
    ]
    if d.get("sharpe_from_summary") is not None:
        lines.append(f"  Sharpe (from summary): {d['sharpe_from_summary']:.3f}")
    lines.extend([
        "",
        f"  N trials (試過幾個策略候選): {d['n_trials']}",
        f"  SR estimates std (DSR 校正): {d.get('sr_estimates_std', 1.0):.4f}",
        f"  期望最大 Sharpe under H0: {d['sr_expected_under_null']:.3f}",
        f"  Deflated p-value: {d['dsr_p_value']:.4f}",
        "",
        f"  {'✅ SIGNIFICANT' if d['is_significant_5pct'] else '❌ NOT significant'} @ 5%",
    ])
    if not d["is_significant_5pct"]:
        lines.append("  → 觀察到的 alpha 在校正 multiple testing 後不足以拒絕 H0（無 skill）")
    return "\n".join(lines)


def _format_pbo_human(d: dict) -> str:
    if d.get("error"):
        return f"PBO error: {d['error']}"
    lines = [
        f"  策略候選數: {d['n_strategies']} (來自 {len(d['strategy_files'])} 個 backtest 檔)",
        f"  時間樣本: {d['n_samples']}, 切 {d['n_splits']} 段",
        f"  Combinations: {d['n_combinations']}, Overfit: {d['overfit_count']}",
        f"  PBO: {d['pbo']:.1%} → {d['verdict']}",
    ]
    if d["pbo"] > 0.5:
        lines.append("  → 訓練期看似最佳的策略，在 test 期通常低於 median = 系統性 overfit")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="DSR + PBO 誠實基準評估")
    parser.add_argument("--backtest", type=Path, default=None,
                        help="指定 backtest JSON；不給用 artifacts/backtest 最新一個")
    parser.add_argument("--n-trials", type=int, default=80,
                        help="跑過幾個策略候選（DSR 校正用，預設 80）")
    parser.add_argument("--sr-std", type=float, default=None,
                        help="SR estimate 標準差（DSR 用）。不給則從實際 backtest 結果估")
    parser.add_argument("--exclude", nargs="*", default=["rotation_"],
                        help="從 sr_std 估計排除的檔名子字串（預設排 rotation_，即 Strategy C/D）")
    parser.add_argument("--pbo", action="store_true",
                        help="同時跑 PBO（讀 artifacts/backtest/*.json 全部）")
    parser.add_argument("--pbo-splits", type=int, default=16,
                        help="PBO 切幾段（必為偶數，預設 16）")
    parser.add_argument("--pbo-min-periods", type=int, default=24,
                        help="PBO 過濾掉期數 < 此值的策略（預設 24）")
    parser.add_argument("--json", action="store_true", help="輸出 JSON 而非人類可讀文字")
    args = parser.parse_args()

    output = {}

    # DSR
    bt_path = args.backtest or _latest_backtest_file()
    if bt_path is None:
        print("❌ 找不到 backtest 結果檔；先跑 make backtest 或指定 --backtest", file=sys.stderr)
        return 1

    dsr_result = run_dsr(
        bt_path, n_trials=args.n_trials,
        sr_estimates_std=args.sr_std, exclude_patterns=args.exclude,
    )
    output["dsr"] = dsr_result

    # PBO
    if args.pbo:
        output["pbo"] = run_pbo(min_periods=args.pbo_min_periods, n_splits=args.pbo_splits)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        print("=" * 60)
        print("Deflated Sharpe Ratio")
        print("=" * 60)
        print(_format_dsr_human(dsr_result))
        if "pbo" in output:
            print()
            print("=" * 60)
            print("Probability of Backtest Overfit")
            print("=" * 60)
            print(_format_pbo_human(output["pbo"]))
        print()
        print("=" * 60)
        print("解讀建議")
        print("=" * 60)
        if dsr_result.get("is_significant_5pct"):
            print("  ✅ DSR p > 0.95 = 即使校正 multiple testing 後，策略仍有顯著 alpha。")
            print("     後續 Stage 3-9 改動以「真實 Sharpe」=", round(dsr_result["sharpe_from_periods"], 3),
                  "為對照標準。")
        else:
            print("  ⚠️  DSR p <= 0.95 = 策略 alpha 可能來自 multiple testing bias。")
            print("     可選對策：(1) 降低 --n-trials 估計（若高估了試驗數）；")
            print("              (2) 接受目前是「弱 alpha」，後續改進需更大 Sharpe 提升才算進步；")
            print("              (3) 重新跑 backtest 拿更多 periods 降低 noise。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
