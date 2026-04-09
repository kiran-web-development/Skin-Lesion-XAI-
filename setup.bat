@echo off
REM =============================================================================
REM Skin Lesion XAI - Automated Windows Setup Script
REM =============================================================================
REM This script sets up the Skin Lesion Classification project on Windows
REM Run as Administrator for best results
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Skin Lesion XAI - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo Please install Python with pip support
    pause
    exit /b 1
)

echo [OK] pip found
echo.

REM Step 1: Create Virtual Environment
echo [1/5] Creating virtual environment...
if exist "venv" (
    echo [INFO] Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo.

REM Step 2: Activate Virtual Environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

echo.

REM Step 3: Upgrade pip, setuptools, wheel
echo [3/5] Upgrading pip and tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)
echo [OK] pip and tools upgraded

echo.

REM Step 4: Install Requirements
echo [4/5] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.

REM Step 5: Verify Setup
echo [5/5] Verifying installation...
if exist verify_setup.py (
    python verify_setup.py
    if errorlevel 1 (
        echo [WARNING] Setup verification found some issues
    )
) else (
    echo [INFO] Skipping verification (verify_setup.py not found)
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. To activate the virtual environment in future sessions, run:
echo    venv\Scripts\activate
echo.
echo 2. To run the Streamlit demo:
echo    streamlit run app.py
echo.
echo 3. To run the Flask web app:
echo    python flask_app.py
echo.
echo 4. To train the model:
echo    python src\train.py
echo.
echo 5. To deactivate the virtual environment:
echo    deactivate
echo.
echo For more information, see SETUP_GUIDE.md
echo.
pause
