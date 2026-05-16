@echo off
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo   MCA v2 - Install / refresh local requirements
echo ============================================================
echo   This will create or update the local .venv folder and
echo   install packages from requirements.txt.
echo ============================================================
echo.

set "VENV_DIR=.venv"
set "VENV_PYTHON=%CD%\%VENV_DIR%\Scripts\python.exe"
set "BOOTSTRAP_PYTHON="

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "BOOTSTRAP_PYTHON=%LocalAppData%\Programs\Python\Python313\python.exe"
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
    echo No working Python interpreter found.
    echo Install Python 3.13, then run this file again.
    pause
    exit /b 1
)

echo Using Python:
"%BOOTSTRAP_PYTHON%" -c "import sys; print(sys.executable); print(sys.version)"
echo.

if not exist "%VENV_PYTHON%" (
    echo Creating local virtual environment in %VENV_DIR%...
    "%BOOTSTRAP_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing requirements.txt...
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.txt.
    pause
    exit /b 1
)

echo.
echo Requirements installed successfully in %VENV_DIR%.
echo You can now run the main app launcher or the batch processor launcher.
echo.
pause
