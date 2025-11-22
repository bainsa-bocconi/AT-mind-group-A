from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# Before merging remove unwanted parameters from adapter-config.json

LORA_ADAPTERS= "lora-outputs/checkpoint-3129"
MODEL_NAME =  "meta-llama/Llama-3.2-3B"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="float32", 
    device_map=None, 
    low_cpu_mem_usage=True
)

if hasattr(base, "hf_device_map"):
    delattr(base, "hf_device_map")

model = PeftModel.from_pretrained(base, LORA_ADAPTERS)
model = model.merge_and_unload()


save_dir = "autotorino-merged-model"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir, safe_serialization=True)
