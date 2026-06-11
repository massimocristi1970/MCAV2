@echo off
setlocal

echo Starting MCAv2 Business Finance Scorecard...
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PYTHON=%CD%\%VENV_DIR%\Scripts\python.exe"
set "BOOTSTRAP_PYTHON="
set "PYTHON_EXE="

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%VENV_PYTHON%"
    )
)

if not defined BOOTSTRAP_PYTHON (
    if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
        set "BOOTSTRAP_PYTHON=%LocalAppData%\Programs\Python\Python313\python.exe"
    )
)

if not defined BOOTSTRAP_PYTHON (
    py -3.13 -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%I in ('py -3.13 -c "import sys; print(sys.executable)"') do set "BOOTSTRAP_PYTHON=%%I"
    )
)

if not defined BOOTSTRAP_PYTHON (
    python -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "BOOTSTRAP_PYTHON=%%I"
    )
)

if not defined BOOTSTRAP_PYTHON (
    echo No working Python 3.13 interpreter found.
    echo Install Python 3.13 and rerun this launcher.
    pause
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo Creating local virtual environment in %VENV_DIR%...
    call "%BOOTSTRAP_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%VENV_PYTHON%"

echo Using Python: %PYTHON_EXE%
call "%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

call "%PYTHON_EXE%" -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.txt.
    pause
    exit /b 1
)

if not exist "logs" mkdir logs
if not exist "data" mkdir data

if exist "app\main.py" (
    call "%PYTHON_EXE%" -m streamlit run app\main.py --server.port 8624 --server.address localhost
) else if exist "main.py" (
    call "%PYTHON_EXE%" -m streamlit run main.py --server.port 8624 --server.address localhost
) else (
    echo Cannot find main.py or app\main.py
    pause
    exit /b 1
)

pause

