# One image for the api and bot services. Runtime dependencies only: no torch / transformers.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/app

RUN useradd --create-home --uid 1000 app
WORKDIR /app
RUN chown app /app

COPY requirements.txt .
RUN pip install -r requirements.txt

USER app

# Static embedding table (~530 MB, numpy only at inference). Own layer so it is cached across
# code changes; downloaded at build time so `docker compose up` needs no manual step.
COPY --chown=app app/__init__.py app/__init__.py
COPY --chown=app app/core/__init__.py app/core/config.py app/core/
COPY --chown=app scripts/__init__.py scripts/download_models.py scripts/
RUN python -m scripts.download_models && rm -rf /home/app/.cache

COPY --chown=app app app
COPY --chown=app scripts scripts
COPY --chown=app alembic alembic
COPY --chown=app alembic.ini .
COPY --chown=app docker/entrypoint.sh docker/entrypoint.sh
# Committed, non-private data and the last evaluation metrics (shown by /stats and /about)
COPY --chown=app data/gazetteer.yaml data/synonyms.yaml data/
COPY --chown=app reports/eval_results.json reports/benchmark_bge_m3.json reports/thresholds.json reports/

EXPOSE 8000
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["api"]
