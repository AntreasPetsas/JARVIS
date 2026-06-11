@echo off
REM Jarvis Launcher Batch Wrapper
REM This script launches the PowerShell launcher while handling execution policy issues

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
for /f "delims=" %%i in ('cd') do set "scriptDir=%%i"

REM Change to script directory
cd /d "%~dp0"

REM Launch PowerShell with the launcher script, bypassing execution policy
powershell -NoProfile -ExecutionPolicy Bypass -File "launcher.ps1"

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit...
    pause >nul
)

endlocal
