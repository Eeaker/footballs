@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Repair Runtime

echo ============================================================
echo Football Insight - Repair Runtime
echo ============================================================
echo This will keep project data and reinstall the Python runtime.
echo.

if exist ".venv\football_insight_install.mode" del /q ".venv\football_insight_install.mode" >nul 2>nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_install.ps1" -Mode AUTO
if errorlevel 1 (
  echo.
  echo [ERROR] Repair failed. Keep this window open.
  pause
  exit /b 1
)
echo.
echo [OK] Repair completed. Double-click DEPLOY_ONE_CLICK_WINDOWS.bat to start.
pause
endlocal & exit /b 0
