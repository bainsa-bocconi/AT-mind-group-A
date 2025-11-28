#!/bin/bash
cd "$(dirname "€0.00")"

echo "========================================="
echo "   AUTOTORINO AI - MAC INSTALLER"
echo "========================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install it."
    exit 1
fi

# 2. Setup Virtual Env
if [ ! -d ".venv" ]; then
    echo "[!] Creating environment..."
    python3 -m venv .venv
    
    source .venv/bin/activate
    
    echo "[!] Installing Dependencies..."
    # Mac doesn't need CUDA, standard pip works for Metal (MPS)
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 3. Launch
echo
echo "[1/3] Checking Model..."
python3 download_model.py

echo "[2/3] Starting Server..."
# Run app.py in background
python3 app.py &
SERVER_PID=$!

echo "[INFO] Waiting 15s..."
sleep 15

echo "[3/3] Starting UI..."
python3 ui.py

# Cleanup when UI closes
kill $SERVER_PID