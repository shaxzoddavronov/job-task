PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install install-bench lint format test audit gazetteer eval distill benchmark db-up migrate import api bot up down

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: $(PY)  ## runtime + dev tools (pytest, ruff)
	$(PY) -m pip install -r requirements-dev.txt

install-bench: $(PY)  ## offline distill/benchmark deps (torch); never used at runtime
	$(PY) -m pip install -r requirements-bench.txt

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

audit:
	$(PY) -m scripts.audit_data

gazetteer:  ## candidates only; data/gazetteer.yaml is curated by hand
	$(PY) -m scripts.build_gazetteer

eval:
	$(PY) -m scripts.evaluate

distill:
	$(PY) -m scripts.distill_bge_m3

benchmark:
	$(PY) -m scripts.benchmark_bge_m3

db-up:
	docker compose up -d db

migrate:
	$(PY) -m alembic upgrade head

import:
	$(PY) -m scripts.import_xlsx

api:
	$(PY) -m uvicorn app.api.main:app --reload --port 8000

bot:
	$(PY) -m app.bot.main

up:
	docker compose up --build

down:
	docker compose down
