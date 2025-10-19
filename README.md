# AT-mind-group-A

Official repository for project: AT Mind; group A

## Description

This is a FastAPI application that integrates with DataPizza AI to provide an API endpoint for interacting with a local Llama 3.2B language model running via Ollama.

## Installation

1. Install dependencies:

   ```bash
   make install
   # or: uv sync
   ```

2. Setup Ollama (install, pull model, start):

   ```bash
   make setup-ollama
   # Follow the echoed instructions
   ```

## Running the App

1. Start Ollama (if not already running):

   ```bash
   ollama serve
   ```

2. Run the FastAPI app:

   ```bash
   make run
   # or: uvicorn main:app --reload
   ```

3. Test the endpoint:

   ```bash
   make test
   # or: curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"prompt": "Hello, world!"}'
   ```
