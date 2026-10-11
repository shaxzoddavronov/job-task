PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
PORT ?= 8000

.PHONY: install models latency lint format test test-db audit gazetteer eval db-up migrate import api bot up down

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: $(PY)  ## runtime + dev tools (pytest, ruff)
	$(PY) -m pip install -r requirements-dev.txt

models:  ## optional: pre-download the transformer models (~4.5 GB)
	$(PY) -m scripts.download_models

latency:  ## per-question latency + GPU memory on this machine
	$(PY) -m scripts.latency

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

up:  ## db + api + bot; GPU mode automatically when Docker can use an NVIDIA GPU
	sh start.sh

down:
	docker compose down
