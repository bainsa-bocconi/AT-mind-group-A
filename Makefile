.PHONY: install run test setup-ollama

install:
	uv sync

run:
	uvicorn main:app --reload

test:
	curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"prompt": "Hello, world!"}'

setup-ollama:
	@echo "Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
	@echo "Pull model: ollama pull llama3.2:3b"
	@echo "Start Ollama: ollama serve"
