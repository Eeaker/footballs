@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Server

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Runtime environment is not installed.
  echo Run RUN_WINDOWS.bat first.
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\windows_launcher.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [ERROR] Football Insight exited with code %RC%.
  pause
)
endlocal & exit /b %RC%
