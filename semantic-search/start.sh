#!/bin/sh
# Start db + api + bot with Docker. Uses the NVIDIA GPU automatically when Docker can reach one,
# otherwise runs on CPU. Extra arguments go to `docker compose up` (e.g. ./start.sh -d).
set -e
cd "$(dirname "$0")"

files="-f docker-compose.yml"
if docker run --rm --gpus all busybox:1.36 true >/dev/null 2>&1; then
  echo "NVIDIA GPU available to Docker: starting in GPU mode (BGE-M3 + reranker)."
  files="$files -f docker-compose.gpu.yml"
else
  echo "No GPU available to Docker: starting in CPU mode (BGE-M3 only, slower)."
fi

# Bot token: BOT_TOKEN from the shell or .env, else bot_token.txt (never printed)
if [ -z "$BOT_TOKEN" ] && ! grep -qs '^BOT_TOKEN=.' .env && [ -f bot_token.txt ]; then
  BOT_TOKEN="$(cat bot_token.txt)"
  export BOT_TOKEN
fi

# shellcheck disable=SC2086
exec docker compose $files up --build "$@"
