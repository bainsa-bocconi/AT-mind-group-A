from fastapi import FastAPI
from datapizza.clients.openai_like import OpenAILikeClient

app = FastAPI()

# Initialize the client for local Llama 3.2B model via Ollama
client = OpenAILikeClient(
    api_key="",  # Ollama doesn't require an API key
    model="llama3.2:3b",  # Assuming the model is available; adjust if needed
    system_prompt="You are a helpful assistant.",
    base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
)


@app.post("/chat")
async def chat(prompt: str):
    response = client.invoke(prompt)
    return {"response": response.content}
