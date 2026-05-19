@echo off
title MCA v2 Batch Processor
echo ==========================================
echo    MCA v2 Batch Processor Launcher
echo ==========================================
echo.
echo Starting application...

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo Current directory: %CD%
echo.

set "PYTHON_EXE="
if exist "..\.venv\Scripts\python.exe" set "PYTHON_EXE=..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
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
echo Installing/upgrading batch processor requirements...
%PYTHON_EXE% -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo ERROR: Requirement installation failed.
    goto :end
)

if exist "sync_batch_inputs.py" (
    echo.
    choice /C YN /N /M "Sync new OneDrive batch input files before starting? [Y/N]: "
    if errorlevel 2 (
        echo Skipping sync.
    ) else (
        echo.
        echo Syncing batch input files...
        %PYTHON_EXE% sync_batch_inputs.py
        if errorlevel 1 (
            echo ERROR: Batch input sync failed.
            goto :end
        )
    )
)

set "PORT="
for %%P in (8502 8503 8504 8505 8506 8507 8508 8509 8510) do (
    if not defined PORT (
        netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul
        if errorlevel 1 (
            set "PORT=%%P"
        ) else (
            echo Port %%P is already in use, trying next port...
        )
    )
)

if not defined PORT (
    echo ERROR: No available Streamlit port found between 8502 and 8510.
    echo Close one of the running Streamlit windows and try again.
    goto :end
)

REM Check if the main Python file exists
if exist "batch_processor_standalone.py" (
    echo Found batch_processor_standalone.py
    echo Starting Streamlit application on port %PORT%...
    echo URL: http://localhost:%PORT%
    %PYTHON_EXE% -m streamlit run batch_processor_standalone.py --server.port %PORT% --server.headless false
) else (
    echo ERROR: batch_processor_standalone.py not found in current directory
    echo Please make sure this batch file is in the same folder as the Python files
    echo Current directory: %CD%
    dir *.py
)

:end
echo.
echo Application stopped.
pause
