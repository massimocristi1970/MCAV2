@echo off
title MCA v2 Batch Input Sync
echo ==========================================
echo    MCA v2 Batch Input Sync
echo ==========================================
echo.

cd /d "%~dp0"

set "PYTHON_EXE="
if exist "..\.venv\Scripts\python.exe" set "PYTHON_EXE=..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "..\venv\Scripts\python.exe" set "PYTHON_EXE=..\venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=py -3"
)
if not defined PYTHON_EXE (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo ERROR: Python was not found.
    echo Install Python 3.10 or newer, then run this launcher again.
    goto :end
)

echo Using Python: %PYTHON_EXE%
echo.

if /I "%~1"=="--dry-run" (
    %PYTHON_EXE% sync_batch_inputs.py --dry-run
) else (
    %PYTHON_EXE% sync_batch_inputs.py
)

:end
echo.
pause
