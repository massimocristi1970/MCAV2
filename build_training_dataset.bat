@echo off
setlocal

echo ============================================
echo Build MCA Training Dataset
echo ============================================

REM Go to repo root (where this BAT lives)
cd /d "%~dp0"

REM Use venv python if it exists, else fallback to system python
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" build_training_dataset.py
) else (
    python build_training_dataset.py
)

echo.
echo ============================================
echo DONE
echo ============================================
pause
endlocal
