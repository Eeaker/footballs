@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - One-click Deploy

echo ============================================================
echo Football Insight V3.0.0 - One-click online deploy
echo No Docker. No system Python. No pip.
echo This window checks the PC from zero, installs anything missing,
echo then starts the app. Do not run any other setup file.
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" goto :install
if not exist ".venv\football_insight_install.mode" goto :install
goto :start

:install
echo [DEPLOY] First run: downloading Python, pip, runtime, and model...
echo Internet is required. This can take several minutes.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_install.ps1" -Mode AUTO
if errorlevel 1 (
  echo.
  echo [ERROR] Automatic deployment failed. Keep this window open.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe was not found after installation.
  pause
  exit /b 1
)
if not exist ".venv\football_insight_install.mode" (
  echo [ERROR] Installation completion marker was not created.
  pause
  exit /b 1
)

:start
echo [START] Opening Football Insight...
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\windows_launcher.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [ERROR] Football Insight exited with code %RC%.
  pause
)
endlocal & exit /b %RC%
