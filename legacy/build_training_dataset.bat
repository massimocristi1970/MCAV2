@echo off
cd /d "%~dp0.."

echo.
echo ============================================================
echo   LEGACY - Build training dataset
echo ============================================================
echo   This uses the older fixed-folder workflow:
echo     data\JsonExport\
echo     data\training_dataset.xlsx
echo.
echo   The MCAV2_BatchProcessor app is now the preferred workflow
echo   for scorecard development.
echo ============================================================
echo.
set /p CONFIRM=Type LEGACY to run this older process: 
if /I not "%CONFIRM%"=="LEGACY" (
    echo Cancelled.
    pause
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" build_training_dataset.py
) else (
    python build_training_dataset.py
)

pause
