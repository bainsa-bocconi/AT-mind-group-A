import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #sanity check

"""
we first load the tokenizer specifying [EOS] padding on the right to avoid
problems with attention layers
"""
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

"""
we then load our model, we can reduce
training load by utilizing the bitsandbytes library from hf, to use QLoRA 
"""
bitsandbytes_loading = BitsAndBytesConfig(
    load_in_4bit = True,                       # we load the model in 4bit
    bnb_4bit_compute_dtype = torch.bfloat16,   # for computations we utilize brain float 16, more stable than float 16
    bnb_4bit_quant_type =  "nf4",              # standard 4bit representation
    #bnb_4bit_use_double_quant = True          # use if training VRAM is an issue
)


model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_id,
    device_map="auto",
    dtype=torch.bfloat16,
                       
    quantization_config = bitsandbytes_loading, # QloRA
)

'''
targeting query and attention blocks plus some other meaningful blocks,
if training is too slow, drop to only query and output layers
'''
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] 

"""
creating a LoRA configuration for the model
"""
lora_config = LoraConfig(
r= 16,
target_modules=target_modules,
lora_alpha=32,
lora_dropout=0.1,
bias="none",
task_type="CAUSAL_LM"
)

"""
after loading the correct model from hugginface, we retrieve the correct
configuration for fine-tuning it with our LoRA pipeline and we prepare it
"""
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


print("model retrieved correctly, quantized to 4 bit, LoRa prepared")