from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from training.lora_config import Config
from transformers import BitsAndBytesConfig

bitsandbytes_loading = BitsAndBytesConfig(
        load_in_4bit = Config.BIT_4_LOADING,                       # we load the model in 4bit
        bnb_4bit_compute_dtype = Config.BIT_4_COMPUTE,             # for computations we utilize brain float 16, more stable than float 16
        bnb_4bit_quant_type =  Config.BIT_4_DTYPE,                 # standard 4bit representation
        bnb_4bit_use_double_quant = Config.BIT_4_DOUBLE_QUANT      # use if training VRAM is an issue
    )

tok = AutoTokenizer.from_pretrained("Canonik/Autotorino-Llama-3.1-8B-instruct") # meta-llama/Llama-3.1-8B-instruct  Canonik/Autotorino-Llama-3.1-8B-instruct

model = AutoModelForCausalLM.from_pretrained(
    "Canonik/Autotorino-Llama-3.1-8B-instruct",
        device_map={"": 0}, 
        torch_dtype=Config.BIT_4_COMPUTE,                      
        quantization_config = bitsandbytes_loading, # QloRA
        low_cpu_mem_usage=True, 
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True
)
model.eval()
model.config.use_cache = True

eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
if eot_id is None:
    eot_id = tok.eos_token_id
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
pad_id = tok.pad_token_id

messages = [
  {"role":"system","content":"Sei un copilot per Autotorino. Fornisci indicazioni utili. Evita allucinazioni"},
  {"role":"user","content":"Il cliente è insoddisfatto del preventivo, vorrebbe un prezzo meno alto, cosa devo fare per convincerlo?"}
]


prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

out = model.generate(
    **inputs,
    max_new_tokens=400, 
    do_sample=True,               
    temperature=0.2, top_p=0.9,  
    repetition_penalty=1.10,
    no_repeat_ngram_size=3,
    eos_token_id=[eot_id, tok.eos_token_id] if tok.eos_token_id is not None else [eot_id],
    pad_token_id=pad_id,
    streamer = streamer
)
print(tok.decode(out[0], skip_special_tokens=True))
