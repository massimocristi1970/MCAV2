@echo off
cd /d "C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2"
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pause
