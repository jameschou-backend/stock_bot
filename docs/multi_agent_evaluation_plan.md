# Multi Agent Evaluation Plan

## 現況盤點

- `skills/daily_pick.py`
  - 已支援 `selection_mode=model|multi_agent`
  - 已有 `research` 降級（`raw_institutional` 缺失時使用 `_research_score_candidates`）
  - 每次成功執行會產生 `artifacts/run_manifest_daily_pick_{job_id}.json`
- `skills/multi_agent_selector.py`
  - 已有 deterministic 5-agent（tech/flow/margin/fund/theme）
  - 已輸出 `reason_json["agents"]` 與 `_selection_meta.weights_used`
- `scripts/compare_runs.py`
  - 已可比較兩個 run manifest（overlap/correlation/新增移除/dq 差異）
- `pipelines/daily_pipeline.py`
  - 已串接 daily ingest + data quality + build + train + daily_pick + export
- `config.yaml`
  - 已有 `SELECTION_MODE`、`MULTI_AGENT_WEIGHTS`、`DATA_QUALITY_MODE` 等必要設定
- `app/models.py`
  - 已有 `strategy_runs/strategy_trades/strategy_positions` 可承接策略回測結果
  - 亦有 `picks/features/raw_prices/price_adjust_factors`

## 既有可用回測能力

- `skills/backtest.py` + `scripts/run_backtest.py`
  - 已提供 walk-forward backtest（模型重訓 + 月度再平衡 + 成本與停損）
  - 目前是「直接從 features/labels 訓練與評估」，未直接吃 `daily_pick manifest`
- `skills/strategy_factory/engine.py` + `scripts/run_strategy_backtest.py`
  - 已提供策略引擎式回測（交易層級）
  - 與現有 daily_pick/multi-agent 產物尚未直接耦合

## 本次新增內容

- `experiments/multi_agent_matrix.yaml`
  - 定義實驗矩陣（model vs multi_agent, strict vs research, baseline/tech-heavy/flow-heavy/defensive）
- `scripts/evaluate_experiment.py`
  - 以 rolling picks 方式執行單組實驗評估，輸出：
    - `artifacts/evaluation_{experiment_id}.json/.md`
    - `artifacts/experiment_picks_{experiment_id}.json`
    - `artifacts/experiment_manifests/{experiment_id}/*.json`
- `scripts/agent_attribution_report.py`
  - 基於 `reason_json["agents"]` 做 attribution，輸出：
    - `artifacts/agent_attribution_{experiment_id}.json/.md`
- `scripts/run_experiment_matrix.py`
  - 讀取 matrix 設定，批量執行實驗、支援 `--resume`，輸出：
    - `artifacts/experiment_matrix_summary.json/.md`
- `scripts/render_experiment_summary.py`
  - 聚合 matrix/evaluation/attribution 成 `artifacts/experiment_summary_latest.md`
- Makefile targets
  - `experiment-matrix`
  - `evaluate-experiment`
  - `agent-attribution`
  - `experiment-summary`
  - `compare-runs`（便捷）

## artifacts 命名規範

- 單組實驗：
  - `evaluation_{experiment_id}.json/.md`
  - `agent_attribution_{experiment_id}.json/.md`
  - `experiment_picks_{experiment_id}.json`
  - `experiment_manifests/{experiment_id}/run_manifest_{experiment_id}_{date}.json`
- 矩陣總覽：
  - `experiment_matrix_summary.json/.md`
  - `experiment_summary_latest.md`
