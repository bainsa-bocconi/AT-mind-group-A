#!/bin/bash
cd "$(dirname "€0.00")"

echo "========================================="
echo "   AUTOTORINO AI - MAC LAUNCHER"
echo "========================================="


if [ ! -d ".venv" ]; then
    echo "[!] Creating environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    
    echo "[!] Installing Dependencies..."
    pip install --upgrade pip
    # Standard PyTorch works for Mac Metal (MPS)
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi


# 3. Start App
echo "[2/2] Starting AI System..."
echo "The browser will open automatically."
echo "Close this window to stop the AI."
echo


python3 app.py