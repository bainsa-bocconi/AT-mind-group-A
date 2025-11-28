import os
import sys
import torch
import uvicorn
import gradio as gr
import webbrowser
from threading import Timer
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Literal, Optional

# --- IMPORT DALLA TUA CONFIGURAZIONE ---
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download

# --- 1. SETUP E DOWNLOAD (Da download_model.py) ---
BASE_MODEL = "Canonik/Autotorino-Llama-3.1-8B-instruct_v2"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "autotorino")

def check_and_download():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print(f"Modello non trovato. Download in corso in: {MODEL_DIR}...")
        snapshot_download(repo_id=BASE_MODEL, local_dir=MODEL_DIR, ignore_patterns=["*.git*"])
    else:
        print(f"Modello trovato in {MODEL_DIR}")

check_and_download()

# --- 2. CARICAMENTO MODELLO (Copia esatta da app.py) ---
print("Caricamento modello in GPU...")

# Configurazione presa dal tuo app.py
bitsandbytes_loading = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map={"": 0},
    torch_dtype=torch.float16,
    quantization_config=bitsandbytes_loading,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    trust_remote_code=False,
    local_files_only=False
)

model.eval()
model.config.use_cache = True

# Fix Token (Copia da app.py)
EOT = tok.convert_tokens_to_ids("<|eot_id|>")
if EOT is None: EOT = tok.eos_token_id
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
pad_id = tok.pad_token_id

# --- 3. FUNZIONE DI GENERAZIONE (Bloccante come in app.py) ---
def core_generate(messages_dicts, max_tokens=256, temperature=0.2, top_p=0.5):
    # 1. Applica il template
    prompt = tok.apply_chat_template(
        messages_dicts,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 2. Prepara i tensori
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    
    # 3. Genera (BLOCCANTE: Aspetta che finisca tutto)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=[tok.eos_token_id, EOT],
            pad_token_id=pad_id,
        )

    # 4. Decodifica solo la risposta nuova
    generated_tokens = out[0][inputs.input_ids.shape[1]:]
    text = tok.decode(generated_tokens, skip_special_tokens=True)
    
    return text

# --- 4. FASTAPI SERVER (Struttura da app.py) ---
app = FastAPI(title="Autotorino GPU API")

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "autotorino"
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        # Convertiamo i messaggi Pydantic in dicts per il core
        msgs = [m.model_dump() for m in req.messages]
        text = core_generate(msgs, req.max_tokens, req.temperature, req.top_p)
        return PlainTextResponse(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. INTERFACCIA GRADIO (Senza Streamer) ---
def gradio_fn(message, history):
    # Ricostruiamo la cronologia nel formato lista di dizionari
    msgs = [{"role": "system", "content": "Sei un copilot per Autotorino, rispondi in maniera completa, senza allucinazioni."}]
    
    for u, a in history:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    
    msgs.append({"role": "user", "content": message})
    
    # Chiamata diretta (non streamata, aspetterà qualche secondo e poi apparirà il testo)
    return core_generate(msgs)

ui = gr.ChatInterface(
    fn=gradio_fn,
    title="Autotorino AI Copilot",
    examples=["Il cliente è insoddisfatto, cosa faccio?", "Dammi info sulle promozioni."]
)

# Montiamo Gradio su FastAPI
app = gr.mount_gradio_app(app, ui, path="/")

# --- 6. AVVIO ---
if __name__ == "__main__":
    def open_browser():
        webbrowser.open("http://localhost:8000")
    
    print("Sistema pronto. Apertura browser...")
    Timer(1.0, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)