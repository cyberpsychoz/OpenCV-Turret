@echo off
REM Windows setup script for OpenCV Turret

echo === OpenCV Turret Windows Setup ===

REM Check Python
python --version || (echo Python is required && exit /b 1)

REM Create virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Upgrade pip
pip install --upgrade pip

REM Install dependencies
echo Installing Python dependencies...
pip install ultralytics opencv-python numpy scipy filterpy
pip install mediapipe==0.10.14
pip install torch torchvision

echo.
echo === Setup complete ===
echo Activate with: .venv\Scripts\activate
echo Run with: python main.py --mode video --video test_videos\test_11.mp4
