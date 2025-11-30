import os
import sys
import torch
import uvicorn
import json
from threading import Thread
import webbrowser  
from threading import Thread, Timer 
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer


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
        body { font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 15px; border-radius: 15px; line-height: 1.5; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .user { align-self: flex-end; background-color: #0078d4; color: white; border-bottom-right-radius: 2px; }
        .assistant { align-self: flex-start; background-color: white; color: #333; border-bottom-left-radius: 2px; border: 1px solid #e0e0e0; }
        #input-area { padding: 20px; background: white; border-top: 1px solid #ddd; display: flex; gap: 10px; }
        input { flex: 1; padding: 15px; border: 1px solid #ccc; border-radius: 25px; font-size: 16px; outline: none; }
        button { padding: 15px 30px; background-color: #0078d4; color: white; border: none; border-radius: 25px; cursor: pointer; font-weight: bold; }
        button:hover { background-color: #005a9e; }
        button:disabled { background-color: #ccc; }
        .typing { font-style: italic; color: #888; font-size: 12px; margin-left: 10px; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div class="message assistant">Ciao! Sono il tuo Copilot Autotorino. Come posso aiutarti oggi?</div>
    </div>
    <div id="input-area">
        <input type="text" id="user-input" placeholder="Scrivi qui la tua domanda..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()" id="send-btn">Invia</button>
    </div>

    <script>
        let history = [{"role": "system", "content": "Sei un copilot per Autotorino. Rispondi in italiano."}];

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const btn = document.getElementById('send-btn');
            const chat = document.getElementById('chat-container');
            const text = input.value.trim();

            if (!text) return;

            
            addMessage(text, 'user');
            input.value = '';
            btn.disabled = true;
            history.push({"role": "user", "content": text});

            
            const botBubble = addMessage("...", 'assistant');
            let botText = "";

            try {
                
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        messages: history,
                        max_tokens: 512,
                        temperature: 0.2,
                        top_p: 0.9
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                botBubble.innerHTML = ""; 

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, {stream: true});
                    botText += chunk;
                    botBubble.innerText = botText; // Aggiorna testo live
                    chat.scrollTop = chat.scrollHeight; // Auto-scroll
                }
                
                history.push({"role": "assistant", "content": botText});

            } catch (error) {
                botBubble.innerText = "Errore: " + error;
            } finally {
                btn.disabled = false;
                input.focus();
            }
        }

        function addMessage(text, role) {
            const chat = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerText = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }
    </script>
</body>
</html>
"""


import queue
class UniversalStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, timeout=None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.timeout = timeout
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_queue.put(text)
        if stream_end: self.text_queue.put(self.stop_signal)
    def put(self, value):
        if len(value.shape) > 1: value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        text = self.tokenizer.decode(value, **self.decode_kwargs)
        self.text_queue.put(text)
    def end(self): self.text_queue.put(self.stop_signal)
    def __iter__(self): return self
    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal: raise StopIteration()
        return value


print(f"Loading model from: {MODEL_DIR}")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=bnb_config, low_cpu_mem_usage=True,
        attn_implementation="sdpa", trust_remote_code=False
    )
    model.eval()
    model.config.use_cache = True
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

EOT = tok.convert_tokens_to_ids("<|eot_id|>")
if EOT is None: EOT = tok.eos_token_id
pad_id = tok.pad_token_id if tok.pad_token_id else tok.eos_token_id

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

def stream_generation(messages_list, max_new, temp, top_p):
    prompt = tok.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = UniversalStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    
    kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=top_p, eos_token_id=[tok.eos_token_id, EOT], pad_token_id=pad_id)
    Thread(target=model.generate, kwargs=kwargs).start()
    
    for new_text in streamer:
        yield new_text


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return HTML_UI


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    msgs_dicts = [m.model_dump() for m in req.messages]
    return StreamingResponse(stream_generation(msgs_dicts, req.max_tokens, req.temperature, req.top_p), media_type="text/plain")

if __name__ == "__main__":
   
    def open_browser():
        print("Opening browser at http://localhost:8000 ...")
        webbrowser.open("http://localhost:8000")

    Timer(1.5, open_browser).start()
    
    # Start Server
    uvicorn.run(app, host="0.0.0.0", port=8000)