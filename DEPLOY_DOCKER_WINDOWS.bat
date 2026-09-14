@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Football Insight - Docker Deploy
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1"
if errorlevel 1 pause
