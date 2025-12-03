import os
import sys
import torch
import uvicorn
import queue
import codecs
import webbrowser
import platform
from threading import Thread, Timer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "autotorino")

app = FastAPI(title="Autotorino AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


HTML_UI = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autotorino Copilot</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 80%; padding: 15px; border-radius: 15px; line-height: 1.5; font-size: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); white-space: pre-wrap; }
        .user { align-self: flex-end; background-color: #0078d4; color: white; border-bottom-right-radius: 2px; }
        .bot { align-self: flex-start; background-color: white; color: #333; border: 1px solid #ddd; border-bottom-left-radius: 2px; }
        #input-area { padding: 20px; background: white; border-top: 1px solid #ddd; display: flex; gap: 10px; }
        input { flex: 1; padding: 15px; border: 1px solid #ccc; border-radius: 30px; font-size: 16px; outline: none; }
        button { padding: 15px 30px; background-color: #0078d4; color: white; border: none; border-radius: 30px; cursor: pointer; font-weight: bold; font-size: 16px; }
        button:hover { background-color: #005a9e; }
        button:disabled { background-color: #ccc; }
    </style>
</head>
<body>
    <div id="chat-box">
        <div class="msg bot">Ciao! Sono il tuo assistente Autotorino. Come posso aiutarti oggi?</div>
    </div>
    <div id="input-area">
        <input type="text" id="user-input" placeholder="Scrivi qui la tua domanda..." onkeypress="if(event.key==='Enter') send()">
        <button onclick="send()" id="btn">Invia</button>
    </div>
    <script>
        let history = [{"role": "system", "content": "Sei un copilot per Autotorino."}];
        
        async function send() {
            const input = document.getElementById('user-input');
            const btn = document.getElementById('btn');
            const box = document.getElementById('chat-box');
            const text = input.value.trim();
            if (!text) return;

            // 1. ADD USER MESSAGE (SAFE METHOD)
            const userDiv = document.createElement('div');
            userDiv.className = 'msg user';
            userDiv.innerText = text;
            box.appendChild(userDiv);
            box.scrollTop = box.scrollHeight;

            input.value = '';
            btn.disabled = true;
            history.push({"role": "user", "content": text});
            
            // 2. ADD BOT PLACEHOLDER
            const botDiv = document.createElement('div');
            botDiv.className = 'msg bot';
            botDiv.innerText = '...';
            box.appendChild(botDiv);
            box.scrollTop = box.scrollHeight;

            try {
                const resp = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ messages: history, max_tokens: 512, temperature: 0.2 })
                });
                
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let botText = "";
                botDiv.innerText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, {stream: true});
                    if (chunk) {
                        botText += chunk;
                        botDiv.innerText = botText;
                        box.scrollTop = box.scrollHeight;
                    }
                }
                history.push({"role": "assistant", "content": botText});
            } catch (e) {
                botDiv.innerText = "Errore: " + e;
            }
            btn.disabled = false;
            input.focus();
        }
    </script>
</body>
</html>
"""


class IncrementalStreamer(TextStreamer):
    def __init__(self, tokenizer, **decode_kwargs):
        # Initialize with skip_prompt=True
        super().__init__(tokenizer, skip_prompt=True, **decode_kwargs)
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text: self.text_queue.put(text)
        if stream_end: self.text_queue.put(self.stop_signal)

    def put(self, value):
        # 1. CRITICAL FIX: Skip the input prompt tokens!
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if len(value.shape) > 1: value = value[0]
        
        # 2. Incremental Decoding
        new_bytes = self.tokenizer.decode(value, skip_special_tokens=True).encode("utf-8")
        new_text = self.decoder.decode(new_bytes, final=False)
        
        if new_text:
            self.text_queue.put(new_text)

    def end(self):
        remaining = self.decoder.decode(b"", final=True)
        if remaining: self.text_queue.put(remaining)
        self.text_queue.put(self.stop_signal)
        
    def __iter__(self): return self
    def __next__(self):
        value = self.text_queue.get()
        if value == self.stop_signal: raise StopIteration()
        return value


print(f"System Detected: {platform.system()}")
print("Scanning hardware...")

DEVICE = "cpu"
QUANT_CONFIG = None
DTYPE = torch.float32 
DEVICE_MAP = None

if torch.cuda.is_available():
    print(f"✅ NVIDIA GPU Detected: {torch.cuda.get_device_name(0)}")
    print("-> Enabling 4-bit Quantization (Fast Mode)")
    DEVICE = "cuda"
    DTYPE = torch.float16
    DEVICE_MAP = {"": 0}
    QUANT_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

elif torch.backends.mps.is_available():
    print(f"🍎 Apple Silicon (Mac) Detected")
    print("-> Using Metal Performance Shaders (Mac Mode)")
    DEVICE = "mps"
    DTYPE = torch.float16
    DEVICE_MAP = None 
    QUANT_CONFIG = None

else:
    print("⚠️ No GPU Detected.")
    print("-> Using CPU Mode (Slow / Compatibility Mode)")
    DEVICE = "cpu"
    DTYPE = torch.float32
    DEVICE_MAP = None
    QUANT_CONFIG = None


print(f"Loading model on {DEVICE.upper()}...")

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        device_map=DEVICE_MAP,
        torch_dtype=DTYPE,
        quantization_config=QUANT_CONFIG, 
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )
    
    if DEVICE_MAP is None:
        model.to(DEVICE)
        
    model.eval()
    
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    print("Possible causes: Corrupted download, or not enough RAM.")
    sys.exit(1)

EOT = tok.convert_tokens_to_ids("<|eot_id|>")
if EOT is None: EOT = tok.eos_token_id
pad_id = tok.pad_token_id if tok.pad_token_id else tok.eos_token_id

class ChatRequest(BaseModel):
    messages: List[dict]
    max_tokens: int = 512
    temperature: float = 0.2

def stream_gen(messages, max_new, temp):
    clean_msgs = []
    for m in messages:
        if 'role' in m and 'content' in m:
            clean_msgs.append(m)

    prompt = tok.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    
    streamer = IncrementalStreamer(tok)
    
    kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=max_new, 
        do_sample=True, 
        temperature=temp, 
        eos_token_id=[tok.eos_token_id, EOT], 
        pad_token_id=pad_id
    )
    
    def run_gen():
        try:
            model.generate(**kwargs)
        finally:
            streamer.end()

    Thread(target=run_gen).start()
    
    for new_text in streamer:
        yield new_text

@app.get("/", response_class=HTMLResponse)
async def root(): return HTML_UI

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    return StreamingResponse(stream_gen(req.messages, req.max_tokens, req.temperature), media_type="text/plain")

if __name__ == "__main__":
    def open_browser():
        print("Opening browser at http://localhost:8000 ...")
        webbrowser.open("http://localhost:8000")

    Timer(1.5, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)