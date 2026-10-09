#!/usr/bin/env bash
# Launcher for Linux and macOS (plan section 10).
#
# Detects an NVIDIA GPU, reports its free VRAM to the app, and starts the right
# compose configuration. Nothing here needs editing: the app picks its model
# from the number this script exports.
set -euo pipefail
cd "$(dirname "$0")"

compose_files=(-f docker-compose.yml)

if command -v nvidia-smi >/dev/null 2>&1 \
   && free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1) \
   && [ -n "${free_mb:-}" ]; then
  # Largest GPU, if there is more than one: Ollama will use a single card.
  export HOST_GPU_FREE_MB="$free_mb"
  export OLLAMA_NUM_PARALLEL=4
  compose_files+=(-f docker-compose.gpu.yml)
  echo "NVIDIA GPU detected: ${free_mb} MB free VRAM."
else
  export HOST_GPU_FREE_MB=0
  export OLLAMA_NUM_PARALLEL=1
  echo "No NVIDIA GPU detected — running on CPU (slower, but fully supported)."
fi

echo
echo "Starting. The first run downloads the language model (a few GB); later"
echo "runs reuse it. Progress appears in the logs below."
echo "When it is ready, open http://localhost:8000"
echo

exec docker compose "${compose_files[@]}" up --build
