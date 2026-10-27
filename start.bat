@echo off
echo ============================================================
echo   Emotion + Intent Engine - Quick Start (Windows)
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists. Skipping.
)

echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo [3/3] Starting server...
echo   Open your browser at: http://localhost:8000
echo   API Docs at:          http://localhost:8000/docs
echo   Press Ctrl+C to stop.
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000
