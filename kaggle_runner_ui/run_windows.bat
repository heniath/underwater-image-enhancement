@echo off
title Kaggle Multi-Account Control Hub
cd /d "%~dp0"
echo =======================================================
echo  Starting Kaggle Multi-Account & Multi-Thread Web Hub
echo =======================================================
python server.py --port 8080 --open-browser
pause
