"""Trial registry — DSR n_trials 的唯一資料源（統計紀律，2026-07-10 缺陷 6 規則 1）。

宣稱語義：**每跑一次回測（不論結果好壞、不論從哪個工具跑）都算一次 trial**——
只記「採用的」會低估 selection bias，DSR 的 multiple-testing 折扣就失真。

因此除了 scripts/run_backtest.py main()，所有直接呼叫 skills.backtest.run_backtest
的多 trial 工具（scripts/optuna_search.py、run_grid_backtest.py、run_walkforward*.py）
也必須在每次評估後 append（用 :func:`record_backtest_trial`，失敗不阻斷掃描）。
record 帶 ``source`` 欄位 + 完整 command，重複配置可事後以 command 去重。

註：本模組刻意不 hook 進 skills.backtest.run_backtest 本體——單元測試也會呼叫
run_backtest，自動 append 會把測試灌進 registry。呼叫端顯式記錄。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

#: registry 檔案：每行一個 JSON record
TRIAL_REGISTRY_PATH = ROOT / "artifacts" / "experiments" / "trial_registry.jsonl"

# 歷史實驗基數（honest_baseline 慣例：DSR 的 n_trials 寧可高估讓折扣偏嚴）。
# registry 自 2026-07-10 才開始記錄；在此之前 docs/experiments_history.md 與
# memory/decisions.md 已記錄 Stage 6.1~10.6、Experiment A~F、Optuna 兩輪（30+ trials）、
# 圓桌 filter group、vol-target/topn sweep 等歷史實驗，合計約 80 個候選配置，
# 全數屬同一 selection process，須計入 multiple-testing 折扣。
HISTORICAL_TRIALS_BASE = 80


def append_trial_registry(record: dict, registry_path=None) -> int:
    """append 一行 JSONL 到 trial registry，回傳 append 後總行數。

    registry 是 DSR n_trials 的資料源：每跑一次回測（不論結果好壞）都算一次
    trial——只記「採用的」會低估 selection bias。
    """
    path = Path(registry_path) if registry_path else TRIAL_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return registry_trial_count(path)


def registry_trial_count(registry_path=None) -> int:
    """registry 現有行數（不存在 = 0）。"""
    path = Path(registry_path) if registry_path else TRIAL_REGISTRY_PATH
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def record_backtest_trial(
    result: dict,
    months: int,
    source: str,
    params: Optional[dict] = None,
    registry_path=None,
) -> Optional[int]:
    """把一次 run_backtest 評估記入 registry（多 trial 工具用；失敗不阻斷掃描）。

    Args:
        result: skills.backtest.run_backtest 回傳 dict（取 summary.sharpe_ratio）。
        months: 該次回測月數。
        source: 工具標籤（"optuna_search" / "grid_backtest" / "walkforward" / ...），
                供事後區分 sweep trial 與 headline run、以 command+params 去重。
        params: 該 trial 的搜尋參數（可選）。

    Returns:
        append 後 registry 總行數；寫入失敗回 None（印警告，不 raise）。
    """
    try:
        summary = (result or {}).get("summary") or {}
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(str(a) for a in sys.argv),
            "source": source,
            "sharpe": summary.get("sharpe_ratio"),
            "months": months,
        }
        if params:
            record["params"] = params
        count = append_trial_registry(record, registry_path)
        print(f"[trial-registry] {source} trial 已記錄（第 {count} 筆）")
        return count
    except Exception as exc:  # registry 故障不可拖垮長時間掃描
        print(f"[trial-registry] {source} trial 記錄失敗（不阻斷）: {exc}")
        return None
