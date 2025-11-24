# AT-mind-group-A

Official repository for project: AT Mind; group A

## Description

This is a FastAPI application that integrates with DataPizza AI to provide an API endpoint for interacting with a local Llama 3.2B language model running via Ollama.

## Installation

python -m uvicorn app:app --host 0.0.0.0 --port 8000

$json = @'
{
  "model": "autotorino",
  "messages": [
    {"role":"system","content":"Sei un copilot per Autotorino."},
    {"role":"user","content":"Cliente insoddisfatto del preventivo: come rispondo?"}
  ],
  "max_tokens": 64
}
'@

Invoke-RestMethod http://localhost:8000/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body $json | ConvertTo-Json -Depth 6
