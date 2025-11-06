import pandas as pd
from datasets import load_dataset, Dataset # (typehint otherwise pylance complains)
import sys

def excel_data_to_csv(excel_path=None, csv_path=None):
    '''
    Simple excel -> converter
    '''


    if excel_path is None or csv_path is None:
        if len(sys.argv) < 3:
            raise Exception("Usage: python excel_to_csv_jsonl.py <excel_path> <csv_path>")
    
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("Converted to", csv_path)

def json_formating(entry):
        return {
        "instruction": (
            "[TASK=LEAD_PREDICT] You are an assistant for a car dealership. "
            "Analyze the lead details below and return ONLY valid minified JSON "
            "with a single key: recommended_next_step."
        ),
        "input": {
            "brand": entry.get("Marca"),
            "model": entry.get("Modello"),
            "vehicle_state": entry.get("Tipologia auto"),
            "budget_range": entry.get("Prezzo"),
            "Extra" : entry.get("Extra"),
            "text": entry.get("Testo della lead")
        },
        "output": {
            "recommended_next_step": entry.get("next_step"),
        },
        "task": "LEAD_PREDICT"
    }
    
def excel_lead_V2_def_to_jsonl(csv_path=None, json_path=None):
    '''
    Converts augmented (labeled) lead csv to a jsonl format which can be utilized
    within the LoRA fine-tuning framework

    The labeled parameter next step becomes the target output for our model fine tuning
    '''

    if csv_path is None or json_path is None:
        if len(sys.argv) < 3:
            raise Exception("Usage: python excel_to_csv_jsonl.py <csv_path> <jsonl_path>")
        csv_path, json_path = sys.argv[1], sys.argv[2]
    
    data: Dataset = load_dataset("csv", data_files=csv_path)["train"]  # utilizes dataset library from hugginface 
    data.map(json_formating)    # applies json formatting to each entry
    data.to_json(json_path)
    print(f"Converted to", json_path)




if __name__ == "__main__":
    excel_lead_V2_def_to_jsonl("data/csv/labeled/Lead_V2_def_labeled.csv", "data/jsonl_augmented/Lead_V2_def.jsonl")