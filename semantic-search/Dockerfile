# One image for the api and bot services, on GPU and CPU machines alike: the CUDA build of
# PyTorch falls back to CPU when no GPU is passed in. Models are not baked in; they are
# downloaded on first start into the `models` volume (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/app \
    HF_HOME=/models/hf \
    EMBEDDING_CACHE_DIR=/models/embeddings

RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app /models
WORKDIR /app
RUN chown app /app

COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 \
 && pip install -r requirements.txt

USER app

COPY --chown=app app app
COPY --chown=app scripts scripts
COPY --chown=app alembic alembic
COPY --chown=app alembic.ini .
COPY --chown=app docker/entrypoint.sh docker/entrypoint.sh
# Strip Windows line endings in case the repo was checked out with CRLF
RUN sed -i 's/\r$//' docker/entrypoint.sh
# Committed, non-private data and the last evaluation metrics (shown by /stats and /about)
COPY --chown=app data/gazetteer.yaml data/synonyms.yaml data/
COPY --chown=app reports/eval_results.json reports/thresholds_rerank.json reports/thresholds_dense.json reports/

EXPOSE 8000
ENTRYPOINT ["sh", "docker/entrypoint.sh"]
CMD ["api"]
