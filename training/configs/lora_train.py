from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lora_config import Config
import torch
from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer, SFTConfig


def load_tokenizer():
    """
    we first load the tokenizer specifying [EOS] padding on the right to avoid
    problems with attention layers
    """

    tok = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"
    tok.chat_template = Config.LLAMA3_CHAT_TEMPLATE  # we set template for future uses
    assert "<|start_header_id|>" in tok.chat_template
    assert "<|eot_id|>" in tok.chat_template
    collator = build_collator(tok)
    return tok, collator

def build_collator(tok):
    # derive the exact assistant prefix robustly
    sentinel = "<<<ASSISTANT_START>>>"
    probe = tok.apply_chat_template(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": sentinel},
        ],
        tokenize=False, add_generation_prompt=False
    )
    i = probe.rfind("<|start_header_id|>assistant<|end_header_id|>")
    assert i != -1, "assistant header not found in probe"
    j = probe.index(sentinel, i)
    response_prefix = probe[i:j]  

    response_prefix_ids = tok.encode(response_prefix, add_special_tokens=False)
    assert response_prefix_ids, "empty response_prefix_ids"

    # Safety guardrail for older versions of collator
    try:
        return DataCollatorForCompletionOnlyLM(
            tokenizer=tok, response_template_ids=response_prefix_ids
        )
    except TypeError:
        return DataCollatorForCompletionOnlyLM(
            tokenizer=tok, response_template=response_prefix
        )


def load_quantized_model():

    '''
    we then load our model, we can reduce
    training load by utilizing the bitsandbytes library from hf, to use QLoRA 
    '''
    bitsandbytes_loading = BitsAndBytesConfig(
        load_in_4bit = Config.BIT_4_LOADING,                       # we load the model in 4bit
        bnb_4bit_compute_dtype = Config.BIT_4_COMPUTE,             # for computations we utilize brain float 16, more stable than float 16
        bnb_4bit_quant_type =  Config.BIT_4_DTYPE,                 # standard 4bit representation
        bnb_4bit_use_double_quant = Config.BIT_4_DOUBLE_QUANT      # use if training VRAM is an issue
    )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "auto")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = Config.MODEL_NAME,
        device_map="auto",
        torch_dtype=Config.BIT_4_COMPUTE,
                       
        quantization_config = bitsandbytes_loading, # QloRA
        max_memory={0: "5.5GiB", "cpu": "48GiB"},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
    task_type="CAUSAL_LM",
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

def load_trainer(model, tok, train_ds, eval_ds, collator):
    sft_cfg = SFTConfig(
        dataset_text_field=None,       
        max_seq_length=Config.MAX_LENGTH,
        packing=False,                 
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=16,
        learning_rate=Config.LEARNING_RATE,
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,
        eval_steps=200,
        save_total_limit=4,
        save_safetensors=True,
        optim="paged_adamw_8bit",
        bf16=False,
        seed=Config.SEED,
        logging_strategy="steps",
        log_level="debug",
        report_to="none",
        disable_tqdm=False,
    )

    def format_example(example):
    
        return tok.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False     #we apply the chat template for the model
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        formatting_func=lambda ex: format_example(ex),
        args=sft_cfg,
    )
    return trainer



