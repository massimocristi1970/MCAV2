@echo off
cd /d "%~dp0.."

echo.
echo ============================================================
echo   LEGACY - JSON banking data -^> reference CSV -^> synthetic data
echo ============================================================
echo   Step 1: Build ml_training_dataset.csv from JSON + spreadsheet
echo   Step 2: Run synthetic engine on that CSV using file/folder picker
echo.
echo   This is not the preferred scorecard-development workflow.
echo   Use MCAV2_BatchProcessor for real paid/not-paid calibration.
echo ============================================================
echo.
echo Make sure you have:
echo   - JSON files in  data\JsonExport\
echo   - Spreadsheet    data\training_dataset.xlsx  (company_name, Outcome)
echo.
set /p CONFIRM=Type LEGACY to run this older chained process: 
if /I not "%CONFIRM%"=="LEGACY" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo --- Step 1: Building reference CSV from JSON + spreadsheet ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" build_training_dataset.py
) else (
    python build_training_dataset.py
)
if errorlevel 1 (
  echo.
  echo Step 1 failed. Check data\JsonExport\ and data\training_dataset.xlsx
  pause
  exit /b 1
)

echo.
echo --- Step 2: Synthetic data engine (select CSV and output folder) ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" build_synthetic_launcher.py
) else (
    python build_synthetic_launcher.py
)

echo.
pause
