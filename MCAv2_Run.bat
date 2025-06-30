@echo off
@echo off
echo Starting MCAv2 Business Finance Scorecard...

REM Go to the directory where this batch file is located
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python first.
    pause
    exit /b
)

REM Install packages if needed (runs quickly if already installed)
pip install streamlit pandas plotly numpy scikit-learn rapidfuzz python-dotenv openpyxl arrow pydantic

REM Create directories if they don't exist
if not exist "logs" mkdir logs
if not exist "data" mkdir data

REM Find and run the main app
if exist "app\main.py" (
    streamlit run app\main.py
) else if exist "main.py" (
    streamlit run main.py
) else (
    echo Cannot find main.py or app\main.py
    pause
)

pause
