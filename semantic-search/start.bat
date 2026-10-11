@echo off
rem Windows launcher: runs start.ps1 (GPU auto-detection). Extra arguments go to docker compose up.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
