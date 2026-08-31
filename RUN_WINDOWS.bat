@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Windows Launcher

echo ============================================================
echo Football Insight - Windows Launcher
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" goto :install
if not exist ".venv\football_insight_install.mode" goto :install
goto :start

:install
echo [FIRST RUN / REPAIR] Runtime environment is not fully installed.
echo Starting the installer...
echo.
call "%~dp0INSTALL_WINDOWS.bat"
if errorlevel 1 (
  echo.
  echo [ERROR] Installation did not complete successfully.
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
call "%~dp0START_WINDOWS.bat"
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
