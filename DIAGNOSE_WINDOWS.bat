@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Windows Diagnostics

echo ============================================================
echo Football Insight V2.3.3 - Windows Diagnostics
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv is not installed yet.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" scripts\windows_torch_probe.py --expect-cuda
set "RC=%ERRORLEVEL%"
echo.
echo Diagnostic JSON:
echo runtime\diagnostics\windows_torch_probe.json
echo.
pause
endlocal & exit /b %RC%
