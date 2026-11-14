.PHONY: env install up down logs test lint benchmark-milan benchmark-fred

env:
	@test -f .env || cp .env.example .env

install:
	pip install -e ".[dev,forecast]"

up: env
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f airflow-scheduler

test:
	python -m pytest -q

lint:
	ruff check src tests airflow/dags

# Benchmarks read from ClickHouse on localhost; run the ingestion DAGs first.
benchmark-milan:
	python -m daf.forecasting.benchmark --source telecom_milan \
		--prediction-length 24 --context-length 512 --n-windows 4

benchmark-fred:
	python -m daf.forecasting.benchmark --source fred --frequency M \
		--prediction-length 12 --context-length 256 --n-windows 3
