.PHONY: migrate pipeline api test dashboard ai-prompt report cron-daily backfill backfill-10y backfill-status

migrate:
	python scripts/migrate.py

pipeline:
	python scripts/run_daily.py

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

# === 歷史資料回補 ===

# 10 年完整回補（初始化 DB 用，支援中斷續傳）
# 包含：prices, institutional, margin
backfill-10y:
	python scripts/backfill_history.py --years 10 --datasets prices,institutional,margin

# 5 年回補（較快）
backfill:
	python scripts/backfill_history.py --years 5 --datasets prices,institutional

# 只回補融資融券
backfill-margin:
	python scripts/backfill_history.py --years 10 --datasets margin

# 顯示回補進度
backfill-status:
	python scripts/backfill_history.py --status

# 估算 API 用量
backfill-estimate:
	python scripts/backfill_history.py --estimate --years 10 --datasets prices,institutional,margin
