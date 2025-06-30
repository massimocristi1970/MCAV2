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

REM Check if the main Python file exists
if exist "batch_processor_standalone.py" (
    echo Found batch_processor_standalone.py
    echo Starting Streamlit application...
    streamlit run batch_processor_standalone.py --server.port 8502 --server.headless false
) else (
    echo ERROR: batch_processor_standalone.py not found in current directory
    echo Please make sure this batch file is in the same folder as the Python files
    echo Current directory: %CD%
    dir *.py
)

echo.
echo Application stopped.
pause