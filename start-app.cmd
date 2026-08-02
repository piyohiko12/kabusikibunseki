@echo off
rem US Stock Analyzer launcher (Windows). Double-click to start; close this window to stop.
rem (Keep this file ASCII-only: cmd.exe parses batch files with the OEM codepage.)
cd /d "%~dp0"

rem First run on a new PC: create the virtual environment automatically.
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup. This may take a few minutes...
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python was not found. Install it first:  winget install -e --id Python.Python.3.12
        pause
        exit /b 1
    )
    python -m venv .venv || (echo Failed to create the virtual environment & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Failed to install dependencies & pause & exit /b 1)
    echo Setup complete.
)

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
