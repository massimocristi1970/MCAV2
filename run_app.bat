@echo off
cd /d "%~dp0"

echo Checking Python version...
python --version || (
    echo ❌ Python is not installed or not in PATH.
    pause
    exit /b
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages from requirements.txt...
pip install -r requirements.txt

echo Launching Streamlit app...
streamlit run app\main.py

echo App has stopped. Press any key to exit.
pause
