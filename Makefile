.PHONY: migrate pipeline pipeline-build api test dashboard ai-prompt report cron-daily backfill backfill-10y backfill-listed backfill-10y-listed backfill-status backfill-estimate backfill-estimate-listed backtest backtest-long rebuild-features research-factors research-grid research-walkforward research-all backfill-prices backfill-institutional dq-report experiment-matrix evaluate-experiment agent-attribution experiment-summary compare-runs

migrate:
	python scripts/migrate.py

pipeline:
	python scripts/run_daily.py

# 跳過抓資料，只跑 data_quality + features/labels/train/pick
pipeline-build:
	python scripts/run_daily.py --skip-ingest

api:
	python scripts/run_api.py

test:
	python scripts/run_tests.py

dashboard:
	streamlit run app/dashboard.py

report:
	python scripts/export_report.py

cron-daily:
	bash scripts/cron_daily.sh

ai-prompt:
	AI_ASSIST_ENABLED=0 python scripts/ai_prompt_demo.py

# === 回測 ===

# Walk-forward 回測（預設 24 個月）
backtest:
	python scripts/run_backtest.py

# 回測 36 個月 + 自訂參數
backtest-long:
	python scripts/run_backtest.py --months 36 --topn 20

# 因子研究（10 年）
research-factors:
	python scripts/research_factors.py

# 參數網格回測（10 年）
research-grid:
	python scripts/run_grid_backtest.py --months 120

# Train5Y-Test1Y walk-forward
research-walkforward:
	python scripts/run_walkforward.py --train-years 5 --test-years 1

research-all: research-factors research-grid research-walkforward

# 重建 features（特徵欄位變更後需要執行）
# 會清空 features/labels/model_versions/picks 並重新建置
rebuild-features:
	@echo "⚠️  即將清空 features, labels, model_versions, picks 表並重建..."
	@echo "按 Ctrl+C 取消，或 Enter 繼續"
	@read _confirm
	python -c "from app.db import get_session; from app.models import Feature, Label, ModelVersion, Pick; s = get_session().__enter__(); [s.execute(t.__table__.delete()) for t in [Pick, ModelVersion, Label, Feature]]; s.commit(); print('已清空 4 張表')"
	python scripts/run_daily.py --skip-ingest

# === 歷史資料回補 ===

# 10 年完整回補（初始化 DB 用，支援中斷續傳）
# 包含：prices, institutional, margin
backfill-10y:
	python scripts/backfill_history.py --years 10 --datasets prices,institutional,margin

# 5 年回補（較快）
backfill:
	python scripts/backfill_history.py --years 5 --datasets prices,institutional

# 只抓上市櫃股票（排除下市、ETF、權證）- 推薦使用
backfill-listed:
	python scripts/backfill_history.py --years 5 --datasets prices,institutional,margin --listed-only

# 10 年上市櫃股票
backfill-10y-listed:
	python scripts/backfill_history.py --years 10 --datasets prices,institutional,margin --listed-only

# 只回補融資融券
backfill-margin:
	python scripts/backfill_history.py --years 10 --datasets margin

# 顯示回補進度
backfill-status:
	python scripts/backfill_history.py --status

# 估算 API 用量
backfill-estimate:
	python scripts/backfill_history.py --estimate --years 10 --datasets prices,institutional,margin

# 估算 API 用量（僅上市櫃）
backfill-estimate-listed:
	python scripts/backfill_history.py --estimate --years 5 --datasets prices,institutional,margin --listed-only

backfill-prices:
	python scripts/backfill_prices.py --days $${DAYS:-180}

backfill-institutional:
	python scripts/backfill_institutional.py --days $${DAYS:-180}

dq-report:
	python scripts/data_quality_report.py --days $${DAYS:-180}

experiment-matrix:
	python scripts/run_experiment_matrix.py --matrix experiments/multi_agent_matrix.yaml --resume

evaluate-experiment:
	python scripts/evaluate_experiment.py --experiment-id "$${EXPERIMENT_ID}" --start-date "$${START_DATE}" --end-date "$${END_DATE}" --selection-mode "$${SELECTION_MODE}" --dq-mode "$${DQ_MODE}" --topn "$${TOPN:-20}" $${WEIGHTS_JSON:+--weights-json "$${WEIGHTS_JSON}"}

agent-attribution:
	python scripts/agent_attribution_report.py --experiment-id "$${EXPERIMENT_ID}"

experiment-summary:
	python scripts/render_experiment_summary.py

compare-runs:
	python scripts/compare_runs.py --a "$${A}" --b "$${B}"
