# app.py
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = r"C:\models\autotorino"          
BASE_MODEL = "Canonik/Autotorino-Llama-3.1-8B-instruct"       


app = FastAPI(title="Autotorino CPU API")


from transformers import BitsAndBytesConfig

bitsandbytes_loading = BitsAndBytesConfig(
        load_in_4bit = True,                       # we load the model in 4bit
        bnb_4bit_compute_dtype =torch.float16,             # for computations we utilize brain float 16, more stable than float 16
        bnb_4bit_quant_type = "nf4",                 # standard 4bit representation
        bnb_4bit_use_double_quant = True      # use if training VRAM is an issue
    )

tok = AutoTokenizer.from_pretrained("Canonik/Autotorino-Llama-3.1-8B-instruct") # meta-llama/Llama-3.1-8B-instruct  Canonik/Autotorino-Llama-3.1-8B-instruct

model = AutoModelForCausalLM.from_pretrained(
    "Canonik/Autotorino-Llama-3.1-8B-instruct",
        device_map="auto", 
        torch_dtype=torch.float16,                      
        quantization_config = bitsandbytes_loading, # QloRA
        low_cpu_mem_usage=True, 
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True
)
model.eval()
model.config.use_cache = True

# eos id for <|eot_id|>
EOT = tok.convert_tokens_to_ids("<|eot_id|>")


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "autotorino"
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        prompt = tok.apply_chat_template(
            [m.dict() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            eos_token_id=[tok.eos_token_id, EOT],
            pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        return {"choices":[{"index":0,"message":{"role":"assistant","content":text}}], "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))