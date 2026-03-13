@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo   SYNTHETIC DATA ENGINE - File / folder selection
echo ============================================================
echo   A window will open to select reference CSV file(s).
echo   Then choose the output folder.
echo   (Use Ctrl+Click to select multiple CSV files.)
echo ============================================================
echo.

python build_synthetic_launcher.py

echo.
pause
