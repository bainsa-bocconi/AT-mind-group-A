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
    BIT_4_DOUBLE_QUANT = False

    #fine-tuning configuration
    EPOCHS = 1
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    MAX_OUTPUT_TOKEN = 500
    TOP_P = 0.9
    TEMPERATURE = 0.7

    MAX_LENGTH = 1024
    
    SEED = 0000

Config = LoraConfig