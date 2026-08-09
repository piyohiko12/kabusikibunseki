@echo off
rem US Stock Analyzer launcher (Windows). Double-click to start; close this window to stop.
rem (Keep this file ASCII-only: cmd.exe parses batch files with the OEM codepage.)
cd /d "%~dp0"

rem Find a Python 3.10+ interpreter (an older Python creates a venv that fails at runtime).
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo Python 3.10 or newer was not found. Install it first:
    echo     winget install -e --id Python.Python.3.12
    echo Then open a NEW window and run this file again.
    pause
    exit /b 1
)

rem Set up on the first run, and again whenever requirements.txt changes (e.g. after git pull).
set "STAMP=.venv\requirements.installed.txt"
if not exist ".venv\Scripts\python.exe" goto setup
fc /b requirements.txt "%STAMP%" >nul 2>&1
if errorlevel 1 goto setup
goto run

:setup
echo Setting up. This may take a few minutes...
if exist ".venv\Scripts\python.exe" goto deps
%PY% -m venv .venv
if errorlevel 1 (
    echo Failed to create the virtual environment
    pause
    exit /b 1
)

:deps
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)
copy /y requirements.txt "%STAMP%" >nul
echo Setup complete.

:run

echo ============================================
echo  US Stock Analyzer  -  http://localhost:8501
echo  The browser will open automatically.
echo  To stop the app, close this window (or Ctrl+C).
echo ============================================
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
echo.
echo The app has stopped. If it failed to start, the port may be in use
echo (the app might already be running in another window).
pause
