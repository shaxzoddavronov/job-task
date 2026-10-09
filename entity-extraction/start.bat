@echo off
REM Launcher for Windows (plan section 10).
REM
REM Detects an NVIDIA GPU, reports its free VRAM to the app, and starts the
REM right compose configuration. Nothing here needs editing: the app picks its
REM model from the number this script sets.
REM
REM Requires Docker Desktop with the WSL2 backend. GPU support in containers
REM also needs a recent NVIDIA driver.
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- Docker has to be running before anything else is worth trying --------
docker info >nul 2>&1
if errorlevel 1 (
  echo.
  echo ERROR: Docker is not responding.
  echo.
  echo Start Docker Desktop, wait until its icon stops animating, then run
  echo this script again.
  echo.
  exit /b 1
)

REM Host port, in case 8000 is already taken:  set APP_PORT=8080 && .\start.bat
if not defined APP_PORT set "APP_PORT=8000"

set "COMPOSE_FILES=-f docker-compose.yml"
set "HOST_GPU_FREE_MB=0"
set "OLLAMA_NUM_PARALLEL=1"
set "GPU_FREE="

REM --- Free VRAM, via a temp file -------------------------------------------
REM nvidia-smi output is read from a file rather than a FOR /F backtick block:
REM cmd splits "--format=csv,noheader,nounits" on its commas, so the options
REM arrive mangled.
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits > "%TEMP%\uzdoc_gpu.txt" 2>nul
  if exist "%TEMP%\uzdoc_gpu.txt" (
    for /f "usebackq tokens=1" %%m in ("%TEMP%\uzdoc_gpu.txt") do (
      if not defined GPU_FREE (
        REM Keep it only if it is a plain number; an error message is not.
        set /a "CHECK=%%m" >nul 2>&1
        if !errorlevel! equ 0 if %%m gtr 0 set "GPU_FREE=%%m"
      )
    )
    del "%TEMP%\uzdoc_gpu.txt" >nul 2>&1
  )
)

if defined GPU_FREE (
  set "HOST_GPU_FREE_MB=!GPU_FREE!"
  set "OLLAMA_NUM_PARALLEL=4"
  set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
  echo NVIDIA GPU detected: !GPU_FREE! MB free VRAM.
) else (
  echo No NVIDIA GPU detected - running on CPU ^(slower, but fully supported^).
)

echo.
echo Starting. The first run builds the image and downloads the language model
echo ^(a few GB^), so expect 10-20 minutes. Later runs reuse both.
echo When it is ready, open http://localhost:%APP_PORT%
echo.

docker compose %COMPOSE_FILES% up --build
endlocal
