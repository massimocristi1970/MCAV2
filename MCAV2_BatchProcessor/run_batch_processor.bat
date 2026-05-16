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
    streamlit run batch_processor_standalone.py --server.port %PORT% --server.headless false
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
