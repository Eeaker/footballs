@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Installer

echo ============================================================
echo Football Insight V2.3.3 - Windows Installer
echo ============================================================
echo.
echo 1. Auto detect NVIDIA / CPU
echo 2. Force NVIDIA GPU mode
echo 3. CPU mode
echo 4. Web / presentation only
echo.

set "MODE="
set "ARG=AUTO"
set /p "MODE=Select [1-4], press Enter for 1: "
if "%MODE%"=="2" set "ARG=GPU"
if "%MODE%"=="3" set "ARG=CPU"
if "%MODE%"=="4" set "ARG=WEB"

echo.
echo [INFO] Install mode: %ARG%
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_install.ps1" -Mode "%ARG%"
if errorlevel 1 (
  echo.
  echo [ERROR] Installation failed. Keep this window open and review the message above.
  pause
  exit /b 1
)

echo.
echo [OK] Installation completed.
pause
endlocal & exit /b 0
