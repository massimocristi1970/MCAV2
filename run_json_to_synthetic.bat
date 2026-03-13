@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo   JSON banking data  -^>  reference CSV  -^>  synthetic data
echo ============================================================
echo   Step 1: Build ml_training_dataset.csv from JSON + spreadsheet
echo   Step 2: Run synthetic engine on that CSV (file/folder picker)
echo ============================================================
echo.
echo Make sure you have:
echo   - JSON files in  data\JsonExport\
echo   - Spreadsheet    data\training_dataset.xlsx  (company_name, Outcome)
echo.
pause

echo.
echo --- Step 1: Building reference CSV from JSON + spreadsheet ---
python build_training_dataset.py
if errorlevel 1 (
  echo.
  echo Step 1 failed. Check data\JsonExport\ and data\training_dataset.xlsx
  pause
  exit /b 1
)

echo.
echo --- Step 2: Synthetic data engine (select CSV and output folder) ---
python build_synthetic_launcher.py

echo.
pause
