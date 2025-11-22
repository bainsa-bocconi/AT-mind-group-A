from lora_config import Config
from configs.lora_train import load_tokenizer, load_quantized_model, load_trainer
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

'''
1) load jsons, merge and shuffle
2) train model with PEFT after loading with 4-bit quantization
3) save results locally --> upload on hf
'''
def load_jsons(paths):
    '''
    Loads json datasets as Dataset hf objects {"train": Dataset(...)}, 
    indexes them at "Train" and concatenates them, 
    '''

    sets = [load_dataset("json", data_files=p, split="train") for p in paths]
    return concatenate_datasets(sets)

def tokenize_row_json(row, tokenizer):
    '''
    DEPRECATED WITH SFFTT
    '''

    template_row = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    toks = tokenizer(template_row, truncation = True, max_length = Config.MAX_LENGTH, padding=False)
    return toks

def tokenize_batch_json(batch, tokenizer):
    '''
    DEPRECATED WITH SFFTT
    '''
    # text is already fully formatted just tokenizes it, for different collator than standard
    texts = batch["text"]
    return tokenizer(
        texts,
        truncation=True,
        max_length=Config.MAX_LENGTH,
        padding=False
    )

def split_shuffle(ds):

    ds = ds.filter(lambda ex: isinstance(ex.get("messages", []), list) and len(ex["messages"]) > 0)
    ds = ds.shuffle(seed=Config.SEED)
    parts = ds.train_test_split(test_size=0.1, seed=Config.SEED)
    return parts["train"], parts["test"]

def build_datasets(jsonl_paths):
    '''
    DEPRECATED WITH SFFTT
    '''
    tokenizer, collator = load_tokenizer()
    tokenizer.chat_template = Config.LLAMA3_CHAT_TEMPLATE
    tokenizer.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)


    raw = load_jsons(jsonl_paths)
    train_raw, eval_raw = split_shuffle(raw)

    train_tok = train_raw.map(lambda batch: tokenize_batch_json(batch, tokenizer),
                              batched=True, remove_columns=train_raw.column_names)
    eval_tok  = eval_raw.map(lambda batch: tokenize_batch_json(batch, tokenizer),
                             batched=True, remove_columns=eval_raw.column_names)
   
    return train_tok, eval_tok, collator


def validate_messages(ex):
    '''
    For safety
    '''
    msgs = ex["messages"]
    assert isinstance(msgs, list) and msgs, "messages must be a non-empty list"
    roles = [m["role"] for m in msgs]
    assert roles[-1] == "assistant", "last message must be assistant"
    assert all(r in {"system","user","assistant","tool"} for r in roles), "unknown role"
    assert all(isinstance(m["content"], str) for m in msgs), "content must be str"

def has_nonempty_answer(ex):
    '''
    for safety
    '''
    last = ex["messages"][-1]
    return bool(last["content"].strip())

def main():
    tok, collator = load_tokenizer()
    raw = load_jsons([
        "data/jsonl/jsonl_llama/Lead_V2_def_labeled_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Assistenza_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Vendite_Nuovo_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Vendite_Usato_llama.jsonl",
    ])
    train_raw, eval_raw = split_shuffle(raw)


    for row in train_raw:
        validate_messages(row)
        has_nonempty_answer(row)
    for row in eval_raw:
        validate_messages(row)
        has_nonempty_answer(row)

    model = load_quantized_model()
    trainer = load_trainer(model, tok, train_raw, eval_raw, collator)
    trainer.train()
    model.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)
    tok.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)

if __name__ == "__main__":
    main()