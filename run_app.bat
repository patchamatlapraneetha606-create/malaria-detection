@echo off
cd /d "%~dp0"
set "PYTHON_DIR=%LocalAppData%\Programs\Python\Python312"
if exist "%PYTHON_DIR%\python.exe" set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
if not exist venv\Scripts\streamlit.exe (
    echo Installing dependencies...
    venv\Scripts\pip.exe install -r requirements.txt -q
)
echo Starting Malaria Detection System...
set ENABLE_LOCAL_TRAINING=1
start http://localhost:8501
venv\Scripts\streamlit.exe run app.py
