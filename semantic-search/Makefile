PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
# Offline-only environment (torch, model2vec[distill], BGE-M3): never used at runtime.
BENCH_VENV := .venv-bench
BENCH_PY := $(BENCH_VENV)/bin/python
BENCH_VENV_FLAGS ?=
PORT ?= 8000

.PHONY: install install-bench models lint format test test-db audit gazetteer eval distill benchmark db-up migrate import api bot up down

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: $(PY)  ## runtime + dev tools (pytest, ruff)
	$(PY) -m pip install -r requirements-dev.txt

$(BENCH_PY):
	$(PYTHON) -m venv $(BENCH_VENV_FLAGS) $(BENCH_VENV)
	$(BENCH_PY) -m pip install --upgrade pip

install-bench: $(BENCH_PY)  ## separate .venv-bench with torch; never used at runtime
	$(BENCH_PY) -m pip install -r requirements-bench.txt

models:  ## download the static embedding model into models/ (~530 MB)
	$(PY) -m scripts.download_models

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

test-db:  ## DB tests against the compose db (own database qa_test)
	TEST_DATABASE_URL=$${TEST_DATABASE_URL:-postgresql+psycopg://qa:qa@localhost:5440/qa_test} $(PY) -m pytest -m db

audit:
	$(PY) -m scripts.audit_data

gazetteer:  ## candidates only; data/gazetteer.yaml is curated by hand
	$(PY) -m scripts.build_gazetteer

eval:
	$(PY) -m scripts.evaluate

distill:
	$(BENCH_PY) -m scripts.distill_bge_m3

benchmark:
	$(BENCH_PY) -m scripts.benchmark_bge_m3

db-up:  ## PostgreSQL 16 on localhost:$${POSTGRES_PORT:-5440}
	docker compose up -d --wait db

migrate:
	$(PY) -m alembic upgrade head

import:
	$(PY) -m scripts.import_xlsx

import-if-empty:
	$(PY) -m scripts.import_xlsx --if-empty

api:  ## PORT=8010 make api if 8000 is taken
	$(PY) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port $(PORT)

bot:
	$(PY) -m app.bot.main

up:  ## BOT_TOKEN from the shell or .env, else from bot_token.txt (never printed or baked in)
	@if [ -z "$$BOT_TOKEN" ] && ! grep -qs '^BOT_TOKEN=.' .env && [ -f bot_token.txt ]; then \
		export BOT_TOKEN="$$(cat bot_token.txt)"; fi; \
	docker compose up --build

down:
	docker compose down
