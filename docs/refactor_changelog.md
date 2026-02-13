# Refactor Changelog (M0~M4)

## M0: 規格書

- 做了什麼
  - 新增 `docs/refactor_plan.md`，明確列出現況問題、目標、M1~M4 範圍、驗收與回滾策略。
- 如何驗收
  - 確認文件存在且內容涵蓋 `raw_fundamentals/raw_theme_flow=0` 與 `labels.max_date` 落後問題。
- 如何回滾
  - `git revert <M0 commit>`（`docs: add refactor plan for risk/regime/data-quality`）

## M1: Data Quality Reports + Dashboard

- 做了什麼
  - 新增 migration：`storage/migrations/005_data_quality_reports.sql`。
  - `skills/data_quality.py` 新增每日報表寫入（upsert）、`report_date` 交易日決策與 notes 規則。
  - `app/dashboard.py` 新增 Data Quality 區塊（最近 30 天 + 紅黃綠灰燈）。
  - 新增 `tests/test_data_quality_reports.py`。
- 如何驗收
  - `pytest -q tests/test_data_quality_reports.py`
  - `python scripts/migrate.py`
  - `python scripts/run_daily.py --skip-ingest`（目前環境會因既有 data quality lag fail-fast）
- 如何回滾
  - `git revert <M1 commit>`（`feat: persist data quality reports + show in dashboard`）

## M2: Risk Layer 抽離

- 做了什麼
  - 新增 `skills/risk.py`：`get_universe`、`apply_liquidity_filter`、`pick_topn`、`apply_stoploss`。
  - `skills/daily_pick.py` 與 `skills/backtest.py` 改為呼叫 risk layer。
  - 新增 `tests/test_risk.py`，並更新 `tests/test_daily_pick_utils.py`。
- 如何驗收
  - `pytest -q tests/test_risk.py tests/test_daily_pick_utils.py`
  - `python scripts/run_daily.py --skip-ingest`（目前環境會因既有 data quality lag fail-fast）
- 如何回滾
  - `git revert <M2 commit>`（`refactor: introduce risk layer and reuse in daily_pick/backtest`）

## M3: Regime 模組化

- 做了什麼
  - 新增 `skills/regime.py`（`BaseRegimeDetector` + `MovingAverageRegimeDetector`）。
  - `skills/daily_pick.py` 改為 detector 判斷市場多空，保留 `bear_topn` 行為。
  - `app/dashboard.py` 改用 detector meta 顯示 `ma_days/current_price/ma_value/diff_pct`。
  - `app/config.py`、`config.yaml`、`.env.example` 新增 `REGIME_DETECTOR`（預設 `ma`）。
  - 新增 `tests/test_regime.py`。
- 如何驗收
  - `pytest -q tests/test_regime.py`
  - `python scripts/run_daily.py --skip-ingest`（目前環境會因既有 data quality lag fail-fast）
- 如何回滾
  - `git revert <M3 commit>`（`refactor: modularize market regime detector (default MA)`）

## M4: Ranker Metrics 標準化

- 做了什麼
  - `skills/train_ranker.py` 新增版本化評估指標：
    - `ic_spearman`
    - `topk`（例如 `k10/k20`）
    - `hitrate`（例如 `k10/k20`）
    - `pred_stats`（`mean/std/min/max`）
  - `app/config.py`、`config.yaml`、`.env.example` 新增 `EVAL_TOPK_LIST` 設定。
  - `app/dashboard.py` models 區塊新增 v1/舊版 metrics 的 graceful fallback 顯示。
  - 新增 `tests/test_train_metrics.py`。
- 如何驗收
  - `pytest -q tests/test_train_metrics.py`
  - `python scripts/run_daily.py --skip-ingest`（目前環境會因既有 data quality lag fail-fast）
- 如何回滾
  - `git revert <M4 commit>`（`feat: add ranker eval metrics (topk/hitrate) and show in dashboard`）

## 一次回滾全部改造

- 由新到舊依序執行：
  - `git revert <M4 commit>`
  - `git revert <M3 commit>`
  - `git revert <M2 commit>`
  - `git revert <M1 commit>`
  - `git revert <M0 commit>`
