@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Offline Installer

echo ============================================================
echo Football Insight V2.3.3 - Windows Offline Installer
echo ============================================================

if not exist "%~dp0wheelhouse\" (
  echo [ERROR] wheelhouse folder was not found.
  echo Run PREPARE_OFFLINE_WINDOWS.bat on an online Windows PC first.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_install_offline.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Offline installation failed.
  pause
  exit /b 1
)

echo.
echo [OK] Offline installation completed.
pause
endlocal & exit /b 0
