@echo off
title Autotorino AI Launcher
cls
cd /d "%~dp0"

echo ========================================================
echo   AUTOTORINO AI - AVVIO UNIVERSALE
echo ========================================================
echo.


py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=py -3.11
    goto CHECK_VENV
)
set PY_CMD=python

:CHECK_VENV

if exist ".venv\Scripts\activate.bat" goto SKIP_INSTALL

echo [!] Primo avvio rilevato. Configurazione in corso...
echo.


%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERRORE] Impossibile creare l'ambiente virtuale.
    pause
    exit
)


call .venv\Scripts\activate.bat


echo.
echo [STEP 1/2] Installazione Motore AI (Tentativo GPU)...


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% equ 0 goto INSTALL_REQ


echo.
echo [AVVISO] Versione GPU non trovata. Installazione versione standard...
pip install torch torchvision torchaudio

if %errorlevel% neq 0 (
    echo.
    echo [ERRORE CRITICO] Impossibile scaricare PyTorch.
    pause
    exit
)

:INSTALL_REQ

echo.
echo [STEP 2/2] Installazione dipendenze app...
pip install -r requirements.txt

echo.
echo [OK] Installazione completata!

:SKIP_INSTALL
if not defined VIRTUAL_ENV call .venv\Scripts\activate.bat




echo.
echo [2/2] Avvio Sistema AI...
echo     Il browser si aprira' automaticamente tra pochi secondi.
echo     Per chiudere l'app, chiudi questa finestra nera.
echo.


python app.py

pause