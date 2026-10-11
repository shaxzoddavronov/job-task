#!/bin/sh
# api: wait for PostgreSQL → migrate → import the data files if the DB is empty → uvicorn.
# bot: start the Telegram bot (talks to the api over HTTP).
set -e

case "$1" in
  api)
    python - <<'PY'
import time
from sqlalchemy import create_engine, text
from app.core.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True)
for attempt in range(60):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Database not reachable after 60 s")
PY
    python -m alembic upgrade head
    python -m scripts.import_xlsx --if-empty
    exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000
    ;;
  bot)
    exec python -m app.bot.main
    ;;
  *)
    exec "$@"
    ;;
esac
