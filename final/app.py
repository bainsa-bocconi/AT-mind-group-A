import os
import sys
import torch
import uvicorn
import platform
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Config Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "autotorino")
BASE_MODEL = "Canonik/Autotorino-Llama-3.1-8B-instruct_v2"

app = FastAPI(title="Autotorino Universal API")


print(f"Operating System: {platform.system()}")
print("Scanning hardware...")

DEVICE = "cpu"
USE_4BIT = False
DTYPE = torch.float32

# 1. Check for NVIDIA (Windows/Linux)
if torch.cuda.is_available():
    print(f"✅ NVIDIA GPU Detected: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
    USE_4BIT = True # 4-bit is great for CUDA
    DTYPE = torch.float16

# 2. Check for Mac (Apple Silicon M1/M2/M3)
elif torch.backends.mps.is_available():
    print(f"🍎 Apple Silicon Detected (Mac Metal Performance Shaders)")
    DEVICE = "mps"
    USE_4BIT = False 
    # bitsandbytes 4-bit is unstable on Mac. We use standard FP16.
    # This requires 16GB+ RAM on the Mac.
    DTYPE = torch.float16 

# 3. Fallback to CPU
else:
    print("⚠️ No GPU detected. Running in slow CPU mode.")
    DEVICE = "cpu"
    USE_4BIT = False
    DTYPE = torch.float32


print(f"Loading engine on: {DEVICE.upper()}...")

quantization_config = None
if USE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map={"": 0} if DEVICE == "cuda" else None, 
        torch_dtype=DTYPE,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )
    
    # Manually move to device if not CUDA 
    if DEVICE != "cuda":
        print(f"Moving model to {DEVICE}...")
        model.to(DEVICE)
        
    model.eval()
    
except Exception as e:
    print(f"CRITICAL LOAD ERROR: {e}")
    if DEVICE == "mps":
        print("Hint: On Mac, ensure you have enough Unified Memory (RAM).")
    sys.exit(1)

# Fix tokens
EOT = tok.convert_tokens_to_ids("<|eot_id|>")
if EOT is None: EOT = tok.eos_token_id
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
pad_id = tok.pad_token_id

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
    return {"status": "ok", "device": DEVICE}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        prompt = tok.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                eos_token_id=[tok.eos_token_id, EOT],
                pad_token_id=pad_id,
            )

        generated_tokens = out[0][inputs.input_ids.shape[1]:]
        text = tok.decode(generated_tokens, skip_special_tokens=True)
        return PlainTextResponse(text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)