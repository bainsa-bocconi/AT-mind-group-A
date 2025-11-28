@echo off
title Autotorino AI Launcher
cls
cd /d "%~dp0"

echo ========================================================
echo   AUTOTORINO AI - AVVIO MODULARE
echo ========================================================
echo.

:: --- 1. SETUP AMBIENTE ---
if not exist ".venv\Scripts\activate.bat" (
    echo [ATTENZIONE] Ambiente virtuale non trovato!
    echo Esegui prima l'installazione o controlla la cartella.
    pause
    exit
)

call .venv\Scripts\activate.bat


echo [1/3] Controllo presenza modello...
python download_model.py
if %errorlevel% neq 0 (
    echo [ERRORE] Il download del modello e' fallito.
    pause
    exit
)

echo.
echo [2/3] Avvio del Server API (Backend)...
echo     Si aprira' una finestra secondaria ridotta a icona. 
echo     NON CHIUDERLA finche' non hai finito.

start "Autotorino API Server" /min cmd /k "call .venv\Scripts\activate.bat && python app.py"


:: Dobbiamo dare tempo al modello di caricarsi nella VRAM prima di lanciare la UI
echo.
echo [INFO] Attendo 120 secondi per il caricamento del modello in GPU...
timeout /t 120 /nobreak >nul


echo.
echo [3/3] Avvio Interfaccia Gradio...
echo     Il browser si aprira' a breve.


python ui.py


echo.
echo Hai chiuso l'interfaccia. Ricordati di chiudere anche la finestra del Server!
pause