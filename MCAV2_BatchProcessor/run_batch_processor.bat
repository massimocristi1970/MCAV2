@echo off
title MCA v2 Batch Processor
echo ==========================================
echo    MCA v2 Batch Processor Launcher
echo ==========================================
echo.
echo Starting application...
cd /d "C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2_BatchProcessor"
echo Current directory: %CD%
echo.
streamlit run batch_processor_standalone.py --server.port 8502 --server.headless false
echo.
echo Application stopped.
pause