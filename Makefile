.PHONY: migrate pipeline pipeline-build pipeline-dag pipeline-dag-build migrate-features daily daily-c api test dashboard trade-dashboard ai-prompt report cron-daily backfill backfill-10y backfill-listed backfill-10y-listed backfill-status backfill-estimate backfill-estimate-listed backtest backtest-long rebuild-features research-factors research-grid research-walkforward research-topn-sweep research-all backfill-prices backfill-institutional dq-report experiment-matrix evaluate-experiment agent-attribution experiment-summary compare-runs profile profile-live slow-queries check-index backfill-fear-greed backfill-gov-bank backfill-holding-dist backfill-broker-trades backfill-kbar backfill-sponsor backfill-per backfill-securities-lending backfill-quarterly-fundamental backfill-value-factors

migrate:
	python scripts/migrate.py

pipeline:
	python scripts/run_daily.py

# 跳過抓資料，只跑 data_quality + features/labels/train/pick
pipeline-build:
	python scripts/run_daily.py --skip-ingest

# DAG 版 pipeline（並行 ingest，預估快 30-40%）
pipeline-dag:
	python scripts/run_daily_dag.py

# DAG 版 pipeline（跳過 ingest，資料已最新時使用）
pipeline-dag-build:
	python scripts/run_daily_dag.py --skip-ingest

# 一次性遷移：MySQL Feature table → 年份 Parquet Feature Store
migrate-features:
	python scripts/migrate_features_to_parquet.py

# 每日選股（Strategy A：月頻 + Strategy C：日頻輪動 + Telegram 推送）
daily:
	python scripts/run_daily.py
	python scripts/strategy_c_pick.py
	python scripts/daily_signal.py
	python scripts/telegram_bot.py --push

# 只跑 Strategy C 選股（資料已是最新時使用）
daily-c:
	python scripts/strategy_c_pick.py

# Strategy C 回測對比：MSE baseline vs RankLabel vs LambdaRank
bt-rank-compare:
	@echo "=== Baseline (excess label only) ==="
	python scripts/backtest_rotation.py --months 36 --max-positions 4 \
	  --exit-mode rank --excess-label \
	  --output artifacts/bt_rank_baseline.json
	@echo "=== Rank Label (MSE on rank) ==="
	python scripts/backtest_rotation.py --months 36 --max-positions 4 \
	  --exit-mode rank --excess-label --rank-label \
	  --output artifacts/bt_rank_label.json
	@echo "=== LambdaRank (direct ranking obj) ==="
	python scripts/backtest_rotation.py --months 36 --max-positions 4 \
	  --exit-mode rank --excess-label --rank-label --ranking-obj \
	  --output artifacts/bt_lambdarank.json

# Strategy D 選股 + 推送 TG（label=5d + trailing stop -25%）
daily-d:
	python scripts/strategy_d_pick.py
	python scripts/telegram_bot.py --push --strategy d

# 啟動 Telegram Bot 監聽模式
bot:
	python scripts/telegram_bot.py --listen

api:
	python scripts/run_api.py

test:
	python scripts/run_tests.py

dashboard:
	streamlit run app/dashboard.py

trade-dashboard:
	python app/trade_dashboard.py

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

# Walk-forward 掃描 TopN（預設 3,5,8,12,20）
research-topn-sweep:
	python scripts/run_walkforward_topn_sweep.py --train-years 5 --test-years 1

research-all: research-factors research-grid research-walkforward research-topn-sweep

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

# === Sponsor 資料集回補 ===
# 注意：需要 FinMind Sponsor 計劃才能存取這些資料集

# CNN 恐懼貪婪指數（最輕量，先跑這個測試 Sponsor token）
backfill-fear-greed:
	python scripts/backfill_sponsor.py --dataset fear_greed

# 官股銀行買賣超（輕量，全市場每日）
backfill-gov-bank:
	python scripts/backfill_sponsor.py --dataset gov_bank

# 持股分級週報（每週一次，資料量小）
backfill-holding-dist:
	python scripts/backfill_sponsor.py --dataset holding_dist

# 分點券商聚合（資料量大，需要較長時間）
backfill-broker-trades:
	python scripts/backfill_sponsor.py --dataset broker_trades

# 分鐘K線日內特徵（資料量最大，需要最長時間）
backfill-kbar:
	python scripts/backfill_sponsor.py --dataset kbar_features

# 回補所有 Sponsor 資料集（依優先序，含舊版 P1-P5）
backfill-sponsor:
	python scripts/backfill_sponsor.py --dataset all

# === 價值因子 + 借券 + 季報（Sponsor，2026-04-23 新增）===

# 本益比/殖利率/本淨比（最推薦先跑，每股一次 API call，每日資料）
backfill-per:
	python scripts/backfill_sponsor.py --dataset per

# 借券餘額聚合（逐筆資料量大，時間較長）
backfill-securities-lending:
	python scripts/backfill_sponsor.py --dataset securities_lending

# 季報財務摘要（三個 dataset 合併，每季一筆，時間較長）
backfill-quarterly-fundamental:
	python scripts/backfill_sponsor.py --dataset quarterly_fundamental

# 一次回補 PER + 借券 + 季報（推薦在 backfill-sponsor 後執行）
backfill-value-factors:
	python scripts/backfill_sponsor.py --dataset per && \
	python scripts/backfill_sponsor.py --dataset securities_lending && \
	python scripts/backfill_sponsor.py --dataset quarterly_fundamental

dq-report:
	python scripts/data_quality_report.py --days $${DAYS:-180}

# 評估歷史訊號績效（IC / 命中率 / 超額報酬）
eval-signal:
	python scripts/eval_signal_perf.py --strategy $${STRATEGY:-c} --topn $${TOPN:-10}

# 評估 Strategy D 訊號績效
eval-signal-d:
	python scripts/eval_signal_perf.py --strategy d --topn 4

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

# === 效能分析與監控 ===

# 使用 pyinstrument 對 pipeline-build 進行 CPU profiling，輸出 HTML 報表
profile:
	DQ_MAX_LAG_TRADING_DAYS=2 python -m pyinstrument -r html -o artifacts/profile_report.html scripts/run_daily.py --skip-ingest
	@echo "Profile report saved to artifacts/profile_report.html"

# 使用 py-spy 即時監控已執行中的 run_daily.py（需先啟動 make pipeline）
profile-live:
	@PID=$$(pgrep -f "run_daily.py" | head -1); \
	if [ -z "$$PID" ]; then \
		echo "Error: no run_daily.py process found. Run 'make pipeline' first."; exit 1; \
	fi; \
	sudo py-spy top --pid $$PID

# 顯示最近 50 筆 slow query 記錄（由 API 自動記錄至 artifacts/slow_queries.jsonl）
slow-queries:
	@if [ -f artifacts/slow_queries.jsonl ]; then \
		python3 -c "import json; lines=[l.strip() for l in open('artifacts/slow_queries.jsonl') if l.strip()]; [print(json.dumps(json.loads(l), ensure_ascii=False, indent=2)) for l in lines[-50:]]"; \
	else \
		echo "No slow queries recorded yet. Start the API and run some requests."; \
	fi

# 檢查 DB 重要索引是否存在
check-index:
	python scripts/check_db_indexes.py
