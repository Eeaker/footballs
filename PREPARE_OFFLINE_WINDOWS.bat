@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Offline Package Builder

echo ============================================================
echo Football Insight V3.0.0 - Prepare Offline Dependencies
echo ============================================================
echo.
echo This step requires Internet access on this Windows PC.
echo It downloads Python 3.12, VC++ runtime, and pip wheels.
echo Copy the whole system folder to the offline PC afterwards.
echo The offline PC does not need Docker or a preinstalled Python.
echo.
echo 1. NVIDIA GPU / CUDA wheel set
echo 2. CPU wheel set
echo.

set "MODE="
set "ARG=GPU"
set /p "MODE=Select [1-2], press Enter for 1: "
if "%MODE%"=="2" set "ARG=CPU"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_prepare_offline.ps1" -Mode "%ARG%"
if errorlevel 1 (
  echo.
  echo [ERROR] Offline dependency preparation failed.
  pause
  exit /b 1
)

echo.
echo [OK] Offline dependency cache completed.
pause
endlocal & exit /b 0
