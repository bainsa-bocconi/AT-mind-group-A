import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from configs.lora_config import Config

"""
we first load the tokenizer specifying [EOS] padding on the right to avoid
problems with attention layers
"""
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return tokenizer, collator
"""
we then load our model, we can reduce
training load by utilizing the bitsandbytes library from hf, to use QLoRA 
"""

def load_quantized_model():
    bitsandbytes_loading = BitsAndBytesConfig(
        load_in_4bit = Config.BIT_4_LOADING,                       # we load the model in 4bit
        bnb_4bit_compute_dtype = Config.BIT_4_COMPUTE,   # for computations we utilize brain float 16, more stable than float 16
        bnb_4bit_quant_type =  Config.BIT_4_DTYPE,              # standard 4bit representation
        bnb_4bit_use_double_quant = Config.BIT_4_DOUBLE_QUANT          # use if training VRAM is an issue
    )


    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = Config.MODEL_NAME,
        device_map="auto",
        dtype=Config.BIT_4_COMPUTE,
                       
        quantization_config = bitsandbytes_loading, # QloRA
    )

    '''
    targeting query and attention blocks plus some other meaningful blocks,
    if training is too slow, drop to only query and output layers
    '''

    """
    creating a LoRA configuration for the model
    """
    lora_config = LoraConfig(
    r= Config.LORA_R,
    target_modules=Config.TARGET_MODULES,
    lora_alpha=Config.LORA_ALPHA,
    lora_dropout=Config.LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
    )

    """
    after loading the correct model from hugginface, we retrieve the correct
    configuration for fine-tuning it with our LoRA pipeline and we prepare it
    """
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("model retrieved correctly, quantized to 4 bit, LoRa prepared")
    return model

def load_trainer():
    _, collator = load_tokenizer()
    model = load_quantized_model()

    trainer = Trainer(
    model=model,
    args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=Config.LEARNING_RATE,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        seed=Config.SEED,
        ),

    train_dataset="",
    eval_dataset="",
    data_collator=collator,
)
    return trainer
