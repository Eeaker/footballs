@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Model Installer

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Runtime environment is not installed.
  echo Run RUN_WINDOWS.bat first.
  pause
  exit /b 1
)

echo ============================================================
echo Football Insight - Install default YOLOv8x model
echo Internet access is required for the first download.
echo You can also upload a .pt model from the System Status page.
echo ============================================================

"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\install_default_model.py"
if errorlevel 1 (
  echo.
  echo [ERROR] Model installation did not complete.
  pause
  exit /b 1
)

echo.
echo [OK] Model installation completed.
echo Run CHECK_WINDOWS.bat before starting a new full analysis.
pause
endlocal & exit /b 0
