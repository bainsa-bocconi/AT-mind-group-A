import gradio as gr
import requests

# Configurazione
API_URL = "http://localhost:8000/v1/chat/completions"

def ask_autotorino(message, history):

    payload = {
        "model": "autotorino",
        "messages": [
            {"role": "system", "content": "Sei un assistente utile per Autotorino."},
            {"role": "user", "content": message}
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }

    try:
        
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.text
        
    except Exception as e:
        return f"Errore: {str(e)}"


demo = gr.ChatInterface(
    fn=ask_autotorino, 
    title="Autotorino Copilot",
    description="Chiedi supporto al modello Llama-3 per Autotorino.",
    examples=["Il cliente è insoddisfatto, cosa faccio?", "Quali sono le promozioni attive?"]
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)