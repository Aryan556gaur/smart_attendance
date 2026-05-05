@echo off
chcp 65001 > nul
echo ============================================
echo   NeuralAttend - Smart Attendance System
echo ============================================
echo.

:: Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

:: Go to project root
cd /d "%~dp0"

:: Install/verify dependencies
echo Installing / verifying dependencies...
pip install -r backend/requirements.txt -q
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting backend on http://localhost:5000
echo Opening frontend in your default browser...
echo.
echo Press Ctrl+C to stop the server.
echo ============================================

:: Open frontend after a short delay
start "" /b cmd /c "timeout /t 2 > nul && start http://localhost:5000/api/health > nul 2>&1 && start frontend\index.html"

:: Start backend (blocking)
python backend\app.py

pause
