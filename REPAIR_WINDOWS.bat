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
call "%~dp0INSTALL_WINDOWS.bat"
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
