#!/bin/bash
# =============================================================================
# Skin Lesion XAI - Automated Setup Script for macOS/Linux
# =============================================================================
# This script sets up the Skin Lesion Classification project
# Usage: bash setup.sh
# =============================================================================

set -e  # Exit on error

echo ""
echo "========================================"
echo "Skin Lesion XAI - Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[OK] Python found"
python3 --version
echo ""

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 is not available"
    echo "Please install Python with pip support"
    exit 1
fi

echo "[OK] pip3 found"
echo ""

# Step 1: Create Virtual Environment
echo "[1/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

echo ""

# Step 2: Activate Virtual Environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"

echo ""

# Step 3: Upgrade pip, setuptools, wheel
echo "[3/5] Upgrading pip and tools..."
python -m pip install --upgrade pip setuptools wheel > /dev/null
echo "[OK] pip and tools upgraded"

echo ""

# Step 4: Install Requirements
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    deactivate
    exit 1
fi
echo "[OK] Dependencies installed"

echo ""

# Step 5: Verify Setup
echo "[5/5] Verifying installation..."
if [ -f verify_setup.py ]; then
    python verify_setup.py
    if [ $? -ne 0 ]; then
        echo "[WARNING] Setup verification found some issues"
    fi
else
    echo "[INFO] Skipping verification (verify_setup.py not found)"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. To activate the virtual environment in future sessions, run:"
echo "   source venv/bin/activate"
echo ""
echo "2. To run the Streamlit demo:"
echo "   streamlit run app.py"
echo ""
echo "3. To run the Flask web app:"
echo "   python flask_app.py"
echo ""
echo "4. To train the model:"
echo "   python src/train.py"
echo ""
echo "5. To deactivate the virtual environment:"
echo "   deactivate"
echo ""
echo "For more information, see SETUP_GUIDE.md"
echo ""
