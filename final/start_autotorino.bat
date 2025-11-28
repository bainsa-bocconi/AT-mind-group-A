@echo off
title Autotorino AI Launcher
cls
cd /d "%~dp0"

echo ========================================================
echo   AUTOTORINO AI - UNIVERSAL INSTALLER
echo ========================================================
echo.

:: --- 1. CHECK PYTHON VERSION ---
:: We look for 3.11 or 3.10 to ensure compatibility
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=py -3.11
    goto CHECK_VENV
)
set PY_CMD=python

:CHECK_VENV
:: --- 2. ENVIRONMENT SETUP ---
if exist ".venv\Scripts\activate.bat" goto SKIP_INSTALL

echo [!] First run detected. Setting up environment...
echo.

:: 2.1 Create VENV
%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Could not create virtual environment.
    pause
    exit
)

:: 2.2 Activate
call .venv\Scripts\activate.bat

:: --- 2.3 SMART PYTORCH INSTALLATION ---
echo.
echo [STEP 1/2] Installing AI Engine...

:: ATTEMPT 1: High-Performance NVIDIA (Specific URL)
echo     ... Trying NVIDIA GPU Version (Recommended)...
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

:: If it worked, skip to requirements
if %errorlevel% equ 0 goto INSTALL_REQ

:: ATTEMPT 2: Fallback / Auto-Detect (Standard Repo)
echo.
echo [WARNING] NVIDIA version failed or not compatible.
echo [FALLBACK] Searching for the best compatible version for THIS device...
echo.
:: This command lets PIP decide what works for this PC (CPU, AMD, etc.)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL ERROR] Could not install PyTorch.
    echo Check internet connection.
    pause
    exit
)

:INSTALL_REQ
:: 2.4 Install other libraries
echo.
echo [STEP 2/2] Installing dependencies...
pip install -r requirements.txt

echo.
echo [OK] Installation Complete!

:SKIP_INSTALL
if not defined VIRTUAL_ENV call .venv\Scripts\activate.bat

:: --- 3. MODULAR LAUNCH ---

echo.
echo [1/3] Checking Model...
python download_model.py
if %errorlevel% neq 0 (
    echo [ERROR] Model download failed.
    pause
    exit
)

echo.
echo [2/3] Starting API Server...
start "Autotorino API Server" /min cmd /k "call .venv\Scripts\activate.bat && python app.py"

echo.
echo [INFO] Waiting 100 seconds for model to load...
timeout /t 100 /nobreak >nul

echo.
echo [3/3] Starting User Interface...
python ui.py

echo.
echo Interface closed. Remember to close the Server window!
pause