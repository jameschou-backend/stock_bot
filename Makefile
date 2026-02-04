.PHONY: migrate pipeline api test

migrate:
	python scripts/migrate.py

pipeline:
	python scripts/run_daily.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000

test:
	pytest -q
