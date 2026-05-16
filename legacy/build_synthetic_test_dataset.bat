@echo off
cd /d "%~dp0.."

echo.
echo ============================================================
echo   SYNTHETIC TEST DATA ENGINE - File / folder selection
echo ============================================================
echo   Synthetic data is for testing, scenario analysis, demos,
echo   and pipeline validation only.
echo.
echo   Do not mix synthetic data into production model calibration.
echo ============================================================
echo.

set /p CONFIRM=Type SYNTHETIC to run the synthetic data picker: 
if /I not "%CONFIRM%"=="SYNTHETIC" (
    echo Cancelled.
    pause
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" build_synthetic_launcher.py
) else (
    python build_synthetic_launcher.py
)

echo.
pause
