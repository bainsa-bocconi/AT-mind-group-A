from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

tok = AutoTokenizer.from_pretrained("Canonik/autotorino-Llama-3.2-3B")

model = AutoModelForCausalLM.from_pretrained(
    "Canonik/autotorino-Llama-3.2-3B",
    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    device_map={"": 0},
    attn_implementation = "sdpa" ,                   
    #offload_folder="./offload",          # optional if not full gpu
)
model.eval()
model.config.use_cache = True


eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
messages = [
  {"role":"system","content":"Sei un copilot per Autotorino. Rispondi con una singola frase."},
  {"role":"user","content":"Il cliente è insoddisfatto del preventivo: come lo gestisco senza perdere la vendita?"}
]


prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False,               
    temperature=0.2, top_p=0.9,     
    repetition_penalty=1.52,
    no_repeat_ngram_size=4,
    eos_token_id=[tok.eos_token_id, eot_id],
    pad_token_id=tok.pad_token_id
)
print(tok.decode(out[0], skip_special_tokens=True))
