@echo off
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo   MCA v2 - Train improved ML model
echo ============================================================
echo   This updates:
echo     app\models\model_artifacts\model.pkl
echo     app\models\model_artifacts\scaler.pkl
echo.
echo   Only run this after you have reviewed the training dataset.
echo ============================================================
echo.

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
set "TRAINING_DATA=data\ml_training_dataset.csv"
set "ARTIFACT_DIR=app\models\model_artifacts"

if not exist "%VENV_PYTHON%" (
    echo Local virtual environment was not found.
    echo Run install_requirements.bat first.
    pause
    exit /b 1
)

if not exist "%TRAINING_DATA%" (
    echo Training data not found: %TRAINING_DATA%
    echo Build/review the dataset first, then rerun this file.
    pause
    exit /b 1
)

echo Training data:
echo   %TRAINING_DATA%
echo.
set /p CONFIRM=Type TRAIN to back up current artifacts and train a new model: 
if /I not "%CONFIRM%"=="TRAIN" (
    echo Cancelled. No model files changed.
    pause
    exit /b 0
)

for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "BACKUP_DIR=%ARTIFACT_DIR%\backups\%STAMP%"
mkdir "%BACKUP_DIR%" >nul 2>&1

if exist "%ARTIFACT_DIR%\model.pkl" copy "%ARTIFACT_DIR%\model.pkl" "%BACKUP_DIR%\model.pkl" >nul
if exist "%ARTIFACT_DIR%\scaler.pkl" copy "%ARTIFACT_DIR%\scaler.pkl" "%BACKUP_DIR%\scaler.pkl" >nul

echo Backed up existing artifacts to:
echo   %BACKUP_DIR%
echo.

"%VENV_PYTHON%" train_improved_model.py --data "%TRAINING_DATA%" --output-dir "%ARTIFACT_DIR%"
if errorlevel 1 (
    echo.
    echo Training failed. Existing backups are in:
    echo   %BACKUP_DIR%
    pause
    exit /b 1
)

echo.
echo Training complete. Review the console metrics before relying on the new model.
echo Backup folder:
echo   %BACKUP_DIR%
echo.
pause
