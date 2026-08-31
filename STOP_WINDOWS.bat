@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Stop Server

if not exist "%~dp0runtime\server.pid" (
  echo [INFO] No running Football Insight service is recorded.
  pause
  exit /b 0
)

set "PID="
set /p "PID=" < "%~dp0runtime\server.pid"
if not defined PID (
  del /q "%~dp0runtime\server.pid" >nul 2>nul
  echo [INFO] Empty PID file removed.
  pause
  exit /b 0
)

echo [STOP] Stopping Football Insight PID %PID% ...
taskkill /PID %PID% /T /F >nul 2>nul
if errorlevel 1 (
  echo [WARN] Process was not found. It may already be stopped.
) else (
  echo [OK] Service stopped.
)

del /q "%~dp0runtime\server.pid" >nul 2>nul
del /q "%~dp0runtime\server.url" >nul 2>nul
pause
endlocal & exit /b 0
