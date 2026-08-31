@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - System Check

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Runtime environment is not installed.
  echo Run RUN_WINDOWS.bat first.
  pause
  exit /b 1
)

set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" "%~dp0scripts\system_check.py"
if errorlevel 1 goto :fail

echo.
echo ===== Engine chain audit =====
"%PY%" "%~dp0scripts\audit_engine_chain.py"
if errorlevel 1 goto :fail

echo.
echo ===== Product / API acceptance =====
"%PY%" "%~dp0scripts\acceptance_check.py"
if errorlevel 1 goto :fail

echo.
echo [OK] All checks passed.
pause
endlocal & exit /b 0

:fail
echo.
echo [ERROR] One or more checks failed.
pause
endlocal & exit /b 1
