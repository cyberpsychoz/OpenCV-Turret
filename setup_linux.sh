#!/bin/bash
# Linux setup script for OpenCV Turret

set -e

echo "=== OpenCV Turret Linux Setup ==="

# Check Python version
python3 --version || { echo "Python 3 is required"; exit 1; }

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install ultralytics opencv-python numpy scipy filterpy

# Install mediapipe (Linux specific)
pip install mediapipe==0.10.14

# Install PyTorch CPU version for MiDaS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=== Setup complete ==="
echo "Activate with: source venv/bin/activate"
echo "Run with: python main.py --mode video --video test_videos/test_11.mp4"
