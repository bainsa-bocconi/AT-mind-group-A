'''
TODO: train ok
      merge lora weights to model  
      upload to hf
'''
from configs.lora_config import Config
from configs.lora_train import load_tokenizer, load_quantized_model, load_trainer
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

'''
1) load jsons, merge and shuffle
2) train model with PEFT after loading with 4-bit quantization
3) save results locally --> upload on hf
'''

def load_jsons(datasets_paths):
    '''
    Loads json datasets as Dataset hf objects {"train": Dataset(...)}, 
    indexes them at "Train" and concatenates them, 
    '''

    jsons = [load_dataset("json", data_files=path, split = "train") for path in datasets_paths]
    dset = concatenate_datasets(dsets=jsons)
    return dset

def tokenize_row_json(row, tokenizer):
    '''
    tokenizer for each row in the json, 
    before we apply template, then we tokenize
    '''

    template_row = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    toks = tokenizer(template_row, truncation = True, max_length = Config.MAX_LENGTH, padding=False)
    return toks

def train_test_hygiene_splitting_and_shuffling(conc_datasets):
    '''
    Eliminates empty strings, or defective jsons
    Then splits the concatenated dataset in a train and test set
    '''
    for row in conc_datasets:
        if "messages" not in row or not isinstance(row["messages"], list):
            raise TypeError("Invalid json formatting1")
        for role in row["messages"]:
            if not isinstance(role, dict): raise TypeError("Invalid json formatting2")
            if "role" not in role.keys() or "content" not in role.keys(): raise TypeError("Invalid Json formatting3")

    
    conc_datasets = conc_datasets.shuffle(seed = Config.SEED)
    splitted_datasets = conc_datasets.train_test_split(test_size=0.1, seed=Config.SEED)
    return splitted_datasets["train"], splitted_datasets["test"]

def build_datasets(jsonl_paths):
    tokenizer, collator = load_tokenizer()
    tokenizer.chat_template = Config.LLAMA2_CHAT_TEMPLATE
    tokenizer.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)


    raw = load_jsons(jsonl_paths)
    train_raw, eval_raw = train_test_hygiene_splitting_and_shuffling(raw)

    train_tok = train_raw.map(lambda row: tokenize_row_json(row, tokenizer),
                              batched=False, remove_columns=train_raw.column_names)
    eval_tok  = eval_raw.map(lambda row: tokenize_row_json(row, tokenizer),
                             batched=False, remove_columns=eval_raw.column_names)
   
    return train_tok, eval_tok, collator

def main():
    jsonl_paths = [
        "data/jsonl/jsonl_llama/Survey_Vendite_Usato_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Vendite_Nuovo_llama.jsonl",
        "data/jsonl/jsonl_llama/Survey_Assistenza_llama.jsonl",
        #"data/jsonl/jsonl_llama/Exit_Poll_-_Aprile_settembre_2025_(1)_llama.jsonl",  #use when sft target is found
        "data/jsonl/jsonl_llama/Lead_V2_def_labeled_llama.jsonl",
    ]
    train_ds, eval_ds, collator = build_datasets(jsonl_paths)
    model = load_quantized_model()

    trainer = load_trainer(model, train_dset=train_ds, test_dset=eval_ds, collator=collator)
    trainer.train()
    model.save_pretrained(Config.SAVE_TRAINED_MODEL_DICT)

if __name__ == "__main__":
    main()