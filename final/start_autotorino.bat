@echo off
title Autotorino AI Launcher
cls
cd /d "%~dp0"

echo ========================================================
echo   AUTOTORINO AI - SISTEMA COMPLETO
echo ========================================================
echo.


:: Cerchiamo Python 3.11 o 3.10 per evitare problemi con la 3.13
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=py -3.11
    goto CHECK_VENV
)
:: Fallback su python di sistema
set PY_CMD=python

:CHECK_VENV

if exist ".venv\Scripts\activate.bat" goto SKIP_INSTALL

echo [!] Primo avvio rilevato (cartella .venv mancante).
echo     Inizio installazione automatica dei requisiti...
echo.


echo Creazione ambiente virtuale con %PY_CMD%...
%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERRORE] Impossibile creare l'ambiente. Controlla l'installazione di Python.
    pause
    exit
)


call .venv\Scripts\activate.bat

:: 2.3 Installazione PyTorch GPU 
echo.
echo [STEP 1/2] Scarico PyTorch GPU (versione specifica CUDA 12.1)...
echo     (Questo file e' grande ~2.5GB, porta pazienza)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo [ERRORE] Download di PyTorch fallito.
    pause
    exit
)


echo.
echo [STEP 2/2] Installazione librerie da requirements.txt...
pip install -r requirements.txt

echo.
echo [OK] Installazione completata!

:SKIP_INSTALL
:: Attivazione ambiente per l'esecuzione
if not defined VIRTUAL_ENV call .venv\Scripts\activate.bat



echo.
echo [1/3] Controllo modello...
python download_model.py
if %errorlevel% neq 0 (
    echo [ERRORE] Script download_model.py fallito o mancante.
    pause
    exit
)

echo.
echo [2/3] Avvio del Server API...
echo     Si aprira' una finestra ridotta a icona. NON CHIUDERLA.
start "Autotorino API Server" /min cmd /k "call .venv\Scripts\activate.bat && python app.py"

echo.
echo [INFO] Attendo 100 secondi che il server e la GPU siano pronti...
timeout /t 100 /nobreak >nul

echo.
echo [3/3] Avvio Interfaccia...
python ui.py

echo.
echo [FINE] Hai chiuso l'interfaccia.
echo Ricordati di chiudere la finestra del Server!
pause