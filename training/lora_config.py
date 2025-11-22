import os.path as op
import torch

BASEDIR = op.abspath(op.dirname(__file__))
class LoraConfig(object):
    #Loading model
    PROJECT_ROOT = BASEDIR
    OUTPUT_DIR = op.join(BASEDIR, "..", "lora-outputs")
    MODEL_NAME =  "meta-llama/Llama-3.2-3B"
    
    #LoRA configuration
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj","down_proj" ]
    
    #QLoRA configuration
    BIT_4_LOADING = True
    BIT_4_COMPUTE = torch.float16
    BIT_4_DTYPE = "nf4"
    BIT_4_DOUBLE_QUANT = True

    #fine-tuning configuration
    EPOCHS = 3
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-4
    MAX_OUTPUT_TOKEN = 500
    TOP_P = 0.9
    TEMPERATURE = 0.7

    MAX_LENGTH = 512
    
    SEED = 0000
    
    SAVE_TRAINED_MODEL_DICT = "autotorino-lora-adapters-plus-tokenizer-3epoch"


    # https://jinja.palletsprojects.com/en/stable/templates/
    # syntax used for building llama chat templates
    LLAMA3_CHAT_TEMPLATE =  '''{% if messages and messages[0]['role'] == 'system' -%}
{{ bos_token }}<|start_header_id|>system<|end_header_id|>
{{ messages[0]['content'] }}<|eot_id|>
{% set loop_messages = messages[1:] -%}
{% else -%}
{{ bos_token }}
{% set loop_messages = messages -%}
{% endif -%}
{% for message in loop_messages -%}
{% set role = message['role'] if message['role'] in ['user','assistant','system','tool'] else 'user' -%}
<|start_header_id|>{{ role }}<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% endfor -%}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}
'''

Config = LoraConfig