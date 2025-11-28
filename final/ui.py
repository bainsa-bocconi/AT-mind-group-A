import gradio as gr
import requests
import sys

# Indirizzo del Server API (app.py)
API_URL = "http://localhost:8000/v1/chat/completions"

def ask_api(message, history):
    # 1. Prepara i dati per il server
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un copilot per Autotorino."}
        ],
        "max_tokens": 256,
        "temperature": 0.2
    }
    
    # 2. Aggiungi la cronologia della chat
    for u, a in history:
        payload["messages"].append({"role": "user", "content": u})
        payload["messages"].append({"role": "assistant", "content": a})
    
    payload["messages"].append({"role": "user", "content": message})
    
    # 3. Invia la richiesta al "cervello" (app.py)
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.text
    except requests.exceptions.ConnectionError:
        return "⚠️ ERRORE: Non riesco a contattare il Server API.\nControlla che la finestra nera ridotta a icona sia ancora aperta!"
    except Exception as e:
        return f"⚠️ ERRORE: {str(e)}"

# Configurazione Interfaccia
demo = gr.ChatInterface(
    fn=ask_api,
    title="Autotorino AI Copilot",
    description="Interfaccia Client connessa al Server API locale.",
    examples=["Il cliente è insoddisfatto, cosa faccio?", "Dammi info sulle promozioni."]
)

if __name__ == "__main__":
    print("Avvio interfaccia... Il browser si aprirà automaticamente quando pronto.")
    # MODIFICA: Usiamo inbrowser=True per forzare l'apertura sicura
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)