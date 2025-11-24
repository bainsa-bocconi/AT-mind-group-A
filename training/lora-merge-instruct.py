from lora_config import Config
from configs.lora_train_instruct import load_tokenizer_instruct, load_quantized_model_instruct, load_trainer_instruct
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

def load_jsons(paths):
    '''
    Loads json datasets as Dataset hf objects {"train": Dataset(...)}, 
    indexes them at "Train" and concatenates them, 
    '''

    sets = [load_dataset("json", data_files=p, split="train") for p in paths]
    return concatenate_datasets(sets)

def split_shuffle(ds):

    ds = ds.filter(lambda ex: isinstance(ex.get("messages", []), list) and len(ex["messages"]) > 0)
    ds = ds.shuffle(seed=Config.SEED)
    parts = ds.train_test_split(test_size=0.1, seed=Config.SEED)
    return parts["train"], parts["test"]


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


VALID_ROLES = {"system", "user", "assistant"}  # keep it strict

def is_valid(ex):
    m = ex.get("messages")
    if not isinstance(m, list) or len(m) == 0:
        return False
    for t in m:
        if not isinstance(t, dict):
            return False
        if t.get("role") not in VALID_ROLES:
            return False
        c = t.get("content")
        if not isinstance(c, str) or len(c.strip()) == 0:
            return False
    return True




def main():
    tok, collator = load_tokenizer_instruct()
    raw = load_jsons([
        "data/jsonl/jsonl_llama/Lead_V2_def_labeled_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Assistenza_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Vendite_Nuovo_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Vendite_Usato_llama.jsonl",
        "data/jsonl/jsonl_llama/information_non_provided.jsonl"
    ])
    train_raw, eval_raw = split_shuffle(raw)


    for row in train_raw:
        validate_messages(row)
        has_nonempty_answer(row)
    for row in eval_raw:
        validate_messages(row)
        has_nonempty_answer(row)

    model = load_quantized_model_instruct()
    trainer = load_trainer_instruct(model, tok, train_raw, eval_raw, collator)

    ckpt = get_last_checkpoint(Config.OUTPUT_DIR) 
    trainer.train()
    model.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)
    tok.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)

    
if __name__ == "__main__":
    main()