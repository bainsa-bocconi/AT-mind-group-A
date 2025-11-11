import pandas as pd
from datasets import load_dataset,Features, Value, Dataset # (typehint otherwise pylance complains)
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

def json_formatting_lead(entry):
        '''
        Deprecated json conversion, use json_formatting_lead_v2
        '''

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
            "text": entry.get("Testo della Lead")
        },
        "output": {
            "recommended_next_step": entry.get("next_step"),
        },
        "task": "LEAD_PREDICT"
    }
def json_formatting_lead_v2(entry):
     '''
     json formatting for TASK LEAD_PREDICT
     Easier for finetuning with LoRA
     '''

     return {
          "text": (
                "### System: Sei un assistente per il concessionario AutoTorino.\n"

                "### Task:[TASK=LEAD_PREDICT] "
                "Analizza i dettagli del Lead e fornisci il passo successivo nella trattativa "
                "con una singola chiave: next_step.\n\n"

                f"### Input: Marca={entry.get('Marca')}, Modello={entry.get("Modello")}, Tipologia auto={entry.get('Tipologia auto')}, Prezzo={entry.get('Prezzo')}, Extra={entry.get('Extra')}\n"
                f"### Client text: {entry.get('Testo della Lead')}\n\n"
                f"### Desired output: {entry.get('next_step')}"

          )
     }

def json_formatting_exit_pool(entry):
    '''
    Converts exit pool csv to a jsonl format which can be utilized
    within the LoRA fine-tuning framework

    Only text inputs are retained, since LoRA aim is for the model to become aware of the specific language necessary to the field
    '''

    return {
         "text": (
              "### System: Sei un assistente per il concessionario AutoTorino.\n"

              f"### Task:[TASK=EXIT_POLL]\n"

              f"### Context Feedback cliente: {'Preventivo non ricevuto' if entry.get('Preventivo ricevuto SI/NO') == 0 else 'Preventivo ricevuto'}\n"
              f"Opinione sul prezzo finale: {entry.get('Vuoi darci qualche indicazione per migliorarla?') if entry.get('Vuoi darci qualche indicazione per migliorarla?') else 'Il cliente non ha fornito opinioni sul prezzo finale'}\n\n"

              f"### Comment: Nota vocale del cliente: {entry.get('Note call') if entry.get('Note call') else 'Il cliente non ha fornito ulteriori commenti'}")
    }
 
def json_formatting_assistenza(entry):
    '''
     Converts assistenza csv to a jsonl format which can be utilized
     within the LoRA fine-tuning framework
    '''

    satisfaction = entry.get("Accoglienza") + entry.get("Verifica tempistiche di prenotazione") + entry.get("Spiegazione lavori effettuati")
    if satisfaction > 27:
        satisfaction = "Cliente decisamente soddisfatto"
    elif 23 < satisfaction <= 27:
        satisfaction = "Cliente abbastanza soddisfatto"
    elif 18 <= satisfaction <= 23:
        satisfaction = "Cliente sufficientemente soddisfatto"
    elif 13 < satisfaction < 18:
        satisfaction = "Cliente insoddisfatto"
    else:
        satisfaction = "Cliente molto insoddisfatto"
    
    if entry.get("Manutenzione Ordinaria (Tagliando) (Si/No)") == "Si":
        Tipo_intervento = "Manutenzione auto"
    elif entry.get("Campagna di richiamo/in garanzia (Si/No)") == "Si":
        Tipo_intervento = "Campagna di richiamo/in garanzia"
    elif entry.get("Lavorazione per riparazione (Si/No)") == "Si":
        Tipo_intervento = "Lavorazione per riparazione"
    else:
        Tipo_intervento = "Sconosciuto"

    return {
         "text": (
              "### System: Sei un assistente per il concessionario Autotorino. \n"

              f"### Task:[TASK=ASSISTENZA_FEEDBACK] Analizza la recensione del cliente sul servizio di assistenza post-intervento e identifica il livello di soddisfazione e il motivo principale.\n\n"

              f"### Input: Tipo di intervento = {Tipo_intervento}\n"
              f"### Commento cliente: {entry.get('Note cliente')}\n"

              f"### Desired Output: Livello reale di soddisfazione = {satisfaction}"

         )
    }


def json_formatting_vendite_nuovo(entry):
    '''
    Converts vendite nuovo csv to a jsonl object that can be utilized
    for LoRA fine-tuning
    '''

    satisfaction = entry.get("Raccomandabilità") + entry.get("Accoglienza") + entry.get("Esperienza d'acquisto") + entry.get("Esperienza consegna") + entry.get("Overall complessivo generale")
    if satisfaction > 45:
        satisfaction = "Cliente decisamente soddisfatto"
    elif 40 < satisfaction <= 45:
        satisfaction = "Cliente abbastanza soddisfatto"
    elif 30 <= satisfaction <= 40:
        satisfaction = "Cliente sufficientemente soddisfatto"
    elif 25 < satisfaction < 30:
        satisfaction = "Cliente insoddisfatto"
    else:
        satisfaction = "Cliente molto insoddisfatto"

    return {
         "text": (
              "### System: Sei un assistente per il concessionario Autotorino. \n"

              f"### Task:[TASK=VENDITE_NUOVO_FEEDBACK] Analizza la recensione del cliente sul servizio di assistenza post-vendita e identifica il livello di soddisfazione e il motivo principale.\n\n"

              f"### Commento cliente: {entry.get('Note cliente')}\n"

              f"### Desired Output: Livello reale di soddisfazione = {satisfaction}"

         )
    }

def json_formatting_vendite_usato(entry):
    '''
    Converts vendite nuovo csv to a jsonl object that can be utilized
    for LoRA fine-tuning
    '''

    satisfaction = entry.get("Raccomandabilità") + entry.get("Accoglienza") + entry.get("Esperienza d'acquisto") + entry.get("Esperienza consegna")
    if satisfaction > 36:
        satisfaction = "Cliente decisamente soddisfatto"
    elif 30 < satisfaction <= 36:
        satisfaction = "Cliente abbastanza soddisfatto"
    elif 24 <= satisfaction <= 30:
        satisfaction = "Cliente sufficientemente soddisfatto"
    elif 20 < satisfaction < 24:
        satisfaction = "Cliente insoddisfatto"
    else:
        satisfaction = "Cliente molto insoddisfatto"

    return {
         "text": (
              "### System: Sei un assistente per il concessionario Autotorino. \n"

              f"### Task:[TASK=VENDITE_USATO_FEEDBACK] Analizza la recensione del cliente sul servizio di assistenza post-vendita e identifica il livello di soddisfazione e su quale aspetto il concessionario Autotorino può migliorare.\n\n"

              f"### Commento cliente: {entry.get('Opinione Miglioramento Servizio')}\n"

              f"### Desired Output: Livello reale di soddisfazione = {satisfaction}"

         )
    }

def csv_to_jsonl(csv_path=None, json_path=None):
    '''
    Converts augmented (labeled) lead csv to a jsonl format which can be utilized
    within the LoRA fine-tuning framework

    The labeled parameter next step becomes the target output for our model fine tuning
    '''

    if csv_path is None or json_path is None:
        if len(sys.argv) < 3:
            raise Exception("Usage: python csv_to_csv_jsonl.py <csv_path> <jsonl_path>")
        csv_path, json_path = sys.argv[1], sys.argv[2]

        
    data: Dataset = load_dataset("csv", data_files=csv_path)["train"]  # utilizes dataset library from hugginface 
    data = data.map(
        CSV_TO_JSON_FORMAT,
        remove_columns=data.column_names,
        desc="Formatting LEAD_PREDICT samples"
    )
    data.to_json(json_path)
    print("Converted to", json_path)

CSV_TO_JSON_FORMAT=json_formatting_vendite_usato
INPUT_DIR="data/csv/raw/Survey_Vendite_Usato.csv"
OUTPUT_DIR="data/jsonl_augmented/Survey_Vendite_Usato.jsonl"

if __name__ == "__main__":
    csv_to_jsonl(INPUT_DIR, OUTPUT_DIR)