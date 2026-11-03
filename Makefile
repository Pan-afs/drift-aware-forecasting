.PHONY: env up down logs test lint

env:
	@test -f .env || cp .env.example .env

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
