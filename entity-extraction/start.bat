@echo off
REM Launcher for Windows (plan section 10).
REM
REM Detects an NVIDIA GPU, reports its free VRAM to the app, and starts the
REM right compose configuration. Nothing here needs editing: the app picks its
REM model from the number this script sets.
REM
REM Requires Docker Desktop with the WSL2 backend. GPU support in containers
REM also needs a recent NVIDIA driver.
setlocal
cd /d "%~dp0"

set "COMPOSE_FILES=-f docker-compose.yml"
set "HOST_GPU_FREE_MB=0"
set "OLLAMA_NUM_PARALLEL=1"

where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  for /f "usebackq delims=" %%m in (
    `nvidia-smi --query-gpu^=memory.free --format^=csv,noheader,nounits 2^>nul`
  ) do (
    if not defined GPU_FREE set "GPU_FREE=%%m"
  )
)

if defined GPU_FREE (
  set "HOST_GPU_FREE_MB=%GPU_FREE%"
  set "OLLAMA_NUM_PARALLEL=4"
  set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
  echo NVIDIA GPU detected: %GPU_FREE% MB free VRAM.
) else (
  echo No NVIDIA GPU detected - running on CPU ^(slower, but fully supported^).
)

echo.
echo Starting. The first run downloads the language model ^(a few GB^); later
echo runs reuse it. Progress appears in the logs below.
echo When it is ready, open http://localhost:8000
echo.

docker compose %COMPOSE_FILES% up --build
endlocal
