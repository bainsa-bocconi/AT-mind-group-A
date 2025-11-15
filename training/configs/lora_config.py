import os.path as op
import torch

BASEDIR = op.abspath(op.dirname(__file__))
class LoraConfig(object):
    #Loading model
    PROJECT_ROOT = BASEDIR
    OUTPUT_DIR = op.join(BASEDIR, "..", "lora-outputs")
    MODEL_NAME =  "meta-llama/Llama-3.2-3B"
    
    #LoRA configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    #QLoRA configuration
    BIT_4_LOADING = True
    BIT_4_COMPUTE = torch.float16
    BIT_4_DTYPE = "nf4"
    BIT_4_DOUBLE_QUANT = True

    #fine-tuning configuration
    EPOCHS = 1
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    MAX_OUTPUT_TOKEN = 500
    TOP_P = 0.9
    TEMPERATURE = 0.7

    MAX_LENGTH = 512
    
    SEED = 0000
    
    SAVE_TRAINED_MODEL_DICT = "autotorino-lora-adapters"


    # https://jinja.palletsprojects.com/en/stable/templates/
    # syntax used for building llama chat templates
    LLAMA2_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
{{ bos_token }}[INST] <<SYS>>
{{ messages[0]['content'] }}
<</SYS>>

{% set loop_messages = messages[1:] %}{% else %}{{ bos_token }}[INST]{% set loop_messages = messages %}{% endif %}
{% for message in loop_messages %}
{% if message['role'] == 'user' %}
{{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'tool' %}
Tool: {{ message['content'] }}{{ eos_token }}
{% endif %}
{% if (loop_messages[loop.index0]['role'] == 'assistant') and not loop.last %}[INST]{% endif %}
{% endfor %}"""

Config = LoraConfig