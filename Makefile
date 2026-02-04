.PHONY: migrate pipeline api test dashboard ai-prompt

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

ai-prompt:
	AI_ASSIST_ENABLED=0 python scripts/ai_prompt_demo.py
