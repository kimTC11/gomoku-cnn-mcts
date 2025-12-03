#!/bin/bash

# Quick setup script for Linux
# This script automates the setup process for the Gomoku AI game

set -e  # Exit on error

echo "=================================="
echo "🎮 Gomoku AI Game - Linux Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""√
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"

# Install PyTorch (CPU version)
echo ""
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
echo "✓ PyTorch installed"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install flask flask-cors numpy pyyaml requests > /dev/null 2>&1
echo "✓ Core dependencies installed"

# Install ML dependencies for 3x3 mode
echo ""
echo "Installing ML dependencies (optional, for 3x3 mode)..."
pip install pandas scikit-learn xgboost catboost joblib > /dev/null 2>&1
echo "✓ ML dependencies installed"

# Make startup script executable
echo ""
echo "Making startup script executable..."
chmod +x web/start_all_servers.sh
echo "✓ Startup script is executable"

# Check for model files
echo ""
echo "Checking for model files..."
model_warnings=0

if [ ! -f "5x5/temp5x5/best.pth.tar" ]; then
    echo "⚠ Warning: 5x5 model not found at 5x5/temp5x5/best.pth.tar"
    model_warnings=$((model_warnings + 1))
fi

if [ ! -f "10x10/temp/best.pth.tar" ]; then
    echo "⚠ Warning: 10x10 model not found at 10x10/temp/best.pth.tar"
    model_warnings=$((model_warnings + 1))
fi

if [ ! -d "3x3/models" ] || [ -z "$(ls -A 3x3/models 2>/dev/null)" ]; then
    echo "⚠ Warning: 3x3 models not found in 3x3/models/"
    model_warnings=$((model_warnings + 1))
fi

if [ $model_warnings -gt 0 ]; then
    echo ""
    echo "⚠ Some models are missing. The game will run but AI will make random moves."
    echo "   You need to either:"
    echo "   1. Download pre-trained models from GitHub releases"
    echo "   2. Train your own models (see SETUP_LINUX.md)"
fi

# Final message
echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To start the game:"
echo "  cd web"
echo "  ./start_all_servers.sh"
echo ""
echo "Then open your browser to: http://localhost:8080"
echo ""
echo "For more information, see SETUP_LINUX.md"
echo "=================================="
