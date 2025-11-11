'''
TODO: train, merge lora weights to model and upload to hf
'''
from configs.lora_config import Config
from configs.lora_train import load_tokenizer, load_quantized_model, load_trainer
from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments

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
    tokenizer for each row in the json, will be mapped to each row of
    the concatenated and shuffled jsons
    '''

    return tokenizer(row["text"], truncation = True, max_length = Config.MAX_LENGTH)

def train_test_hygiene_splitting_and_shuffling(conc_datasets):
    '''
    Eliminates empty strings, or defective jsons, too long/too short.
    Then splits the concatenated dataset in a train and test set
    '''
    conc_datasets = conc_datasets.filter(lambda x: Config.MAX_LENGTH > len(x.get("text", "").strip().lower()) > 0 )

    splitted_datasets = conc_datasets.train_test_split(test_size=0.1, seed=Config.SEED)
    return splitted_datasets["train"], splitted_datasets["test"]

def build_datasets(jsonl_paths):
    tokenizer, collator = load_tokenizer()
    raw = load_jsons(jsonl_paths)
    train_raw, eval_raw = train_test_hygiene_splitting_and_shuffling(raw)

    train_tok = train_raw.map(lambda row: tokenize_row_json(row, tokenizer),
                              batched=True, remove_columns=train_raw.column_names)
    eval_tok  = eval_raw.map(lambda row: tokenize_row_json(row, tokenizer),
                             batched=True, remove_columns=eval_raw.column_names)
    return train_tok, eval_tok, collator

def main():
    jsonl_paths = [
        "data/jsonl_augmented/Survey_Vendite_Usato.jsonl",
        "data/jsonl_augmented/Survey_Vendite_Nuovo.jsonl",
        "data/jsonl_augmented/Survey_Assistenza.jsonl",
        "data/jsonl_augmented/Exit_Poll_-_Aprile_settembre_2025_(1).jsonl",
        "data/jsonl_augmented/Lead_V22_def.jsonl",
    ]
    train_ds, eval_ds, collator = build_datasets(jsonl_paths)
    model = load_quantized_model()

    trainer = load_trainer(model, train_dset=train_ds, test_dset=eval_ds, collator=collator)
    trainer.train()

if __name__ == "__main__":
    main()