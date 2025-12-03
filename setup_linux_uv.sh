#!/bin/bash

# Quick setup script for Linux using UV package manager
# This script automates the setup process for the Gomoku AI game

set -e  # Exit on error

echo "=================================="
echo "🎮 Gomoku AI Game - Linux Setup (UV)"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ UV installed"
    echo ""
    echo "Please restart your terminal or run:"
    echo "  source \$HOME/.cargo/env"
    echo "Then run this script again."
    exit 0
fi

echo "✓ UV is installed"

# Check Python version
echo ""
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "✓ Python version: $python_version"
else
    echo "⚠ Python not found. UV will download Python automatically."
fi

# Initialize UV project (creates virtual environment automatically)
echo ""
echo "Initializing UV environment..."
if [ ! -f "pyproject.toml" ]; then
    echo "⚠ pyproject.toml not found. Creating basic configuration..."
    cat > pyproject.toml << EOF
[project]
name = "gomoku-ai"
version = "1.0.0"
description = "Multi-board Gomoku AI with AlphaZero"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "flask>=2.3.0",
    "flask-cors>=4.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "catboost>=1.2",
    "joblib>=1.3.0",
]

[tool.uv]
dev-dependencies = []
EOF
    echo "✓ Created pyproject.toml"
else
    echo "✓ pyproject.toml exists"
fi

# Sync dependencies with UV
echo ""
echo "Installing dependencies with UV (this may take a few minutes)..."
uv sync
echo "✓ Dependencies installed"

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
echo "  uv run bash web/start_all_servers.sh"
echo ""
echo "Or activate the UV environment manually:"
echo "  source .venv/bin/activate"
echo "  cd web"
echo "  ./start_all_servers.sh"
echo ""
echo "Then open your browser to: http://localhost:8080"
echo ""
echo "For more information, see SETUP_LINUX.md"
echo "=================================="
