from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from configs.lora_config import Config
import torch

def load_tokenizer():
    """
    we first load the tokenizer specifying [EOS] padding on the right to avoid
    problems with attention layers
    """

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return tokenizer, collator


def load_quantized_model():

    """
    we then load our model, we can reduce
    training load by utilizing the bitsandbytes library from hf, to use QLoRA 
    
    bitsandbytes_loading = BitsAndBytesConfig(
        load_in_4bit = Config.BIT_4_LOADING,                       # we load the model in 4bit
        bnb_4bit_compute_dtype = Config.BIT_4_COMPUTE,             # for computations we utilize brain float 16, more stable than float 16
        bnb_4bit_quant_type =  Config.BIT_4_DTYPE,                 # standard 4bit representation
        bnb_4bit_use_double_quant = Config.BIT_4_DOUBLE_QUANT      # use if training VRAM is an issue
    )
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "aut0")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = Config.MODEL_NAME,
        device_map="auto",
        dtype=Config.BIT_4_COMPUTE,
                       
        #quantization_config = bitsandbytes_loading, # QloRA
    ).to(device)

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
    print("model retrieved correctly\n\n, quantized to 4 bit FALSE\n\n, LoRa prepared\n\n")
    return model

def load_trainer(model, train_dset, test_dset, collator):

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
        bf16=False,
        seed=Config.SEED,

        # Verbose terminal
        logging_strategy="steps",
        log_level="debug",
        report_to="none",
        disable_tqdm=False,
        ),

    train_dataset=train_dset,
    eval_dataset=test_dset,
    data_collator=collator,
)
    return trainer
