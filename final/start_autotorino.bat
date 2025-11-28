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

:: --- 2. DOWNLOAD MODELLO (Bloccante) ---
:: Questo deve finire PRIMA di lanciare il server
echo [1/3] Controllo presenza modello...
python download_model.py
if %errorlevel% neq 0 (
    echo [ERRORE] Il download del modello e' fallito.
    pause
    exit
)

:: --- 3. AVVIO SERVER API (Background) ---
:: Usiamo 'start' per aprire una nuova finestra dedicata al server.
:: Quella finestra attivera' il suo venv e lancera' app.py
echo.
echo [2/3] Avvio del Server API (Backend)...
echo     Si aprira' una finestra secondaria ridotta a icona. 
echo     NON CHIUDERLA finche' non hai finito.

start "Autotorino API Server" /min cmd /k "call .venv\Scripts\activate.bat && python app.py"

:: --- 4. ATTESA CARICAMENTO ---
:: Dobbiamo dare tempo al modello di caricarsi nella VRAM prima di lanciare la UI
echo.
echo [INFO] Attendo 50 secondi per il caricamento del modello in GPU...
timeout /t 50 /nobreak >nul

:: --- 5. AVVIO INTERFACCIA UTENTE (Frontend) ---
echo.
echo [3/3] Avvio Interfaccia Gradio...
echo     Il browser si aprira' a breve.

:: Lancia ui.py che si colleghera' al server aperto al punto 3
python ui.py

:: Quando chiudi la UI (CTRL+C), il batch arriva qui
echo.
echo Hai chiuso l'interfaccia. Ricordati di chiudere anche la finestra del Server!
pause