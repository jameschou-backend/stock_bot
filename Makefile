.PHONY: migrate pipeline api test dashboard ai-prompt report cron-daily

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
