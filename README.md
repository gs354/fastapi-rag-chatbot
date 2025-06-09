# A RAG chatbot API built with FastAPI, LlamaIndex, Ollama, and Weaviate.

## Project Structure

```
fastapi_rag_chatbot/
├── api/
│   ├── __init__.py
│   ├── routes.py          # API route handlers
│   ├── conversation.py    # Conversation domain models
│   └── vector_store.py    # Document storage and retrieval
│   
├── schemas/
│   ├── __init__.py
│   └── models.py         # Pydantic models for API schemas
│
├── __init__.py
├── main.py              # FastAPI application setup
└── dependencies.py      # Application dependencies and interfaces

tests/
├── __init__.py
└── test_routes.py      # Testing the endpoints

data/
└── ai-agents-arxiv-papers/  # Document storage

config.toml              # Application configuration
docker-compose.yml       # Docker services configuration
.env.local              # Local environment variables
.env.docker              # Docker environment variables
pyproject.toml          # Project dependencies and metadata
```

## Prerequisites

- Docker and Docker Compose
- Ollama running locally with the llama3.1 and all-minilm models present

## Setup

1. Make sure Ollama is running and the required models are downloaded:
```bash
ollama pull llama3.1
ollama pull all-minilm
```

2. Place your documents in the `data/ai-agents-arxiv-papers` directory. The API will automatically index these documents when started.

3. Create a `.env.docker` file in the project root with the following variables:
```bash
WEAVIATE_HOST=weaviate
OLLAMA_HOST=http://host.docker.internal:11434
LOG_LEVEL=INFO
```

## Running the API

1. Build and start the services:
```bash
docker-compose up --build
```

This will:
- Start Weaviate vector database
- Build and start the API service
- Mount the data directory for document access
- Configure the services using environment variables from `.env.docker`

The API will be available at http://localhost:8000

To stop the services:
```bash
docker-compose down
```

## Interacting with the API

### Option 1: Using the Swagger UI
1. Open your browser and go to http://localhost:8000/docs
2. You'll see an interactive API documentation
3. You can:
   - GET /documents - List all available documents
   - POST /conversations - Create a new conversation (optionally specify document_ids)
   - POST /conversations/{conversation_id}/messages - Send a message to a conversation
   - GET /conversations/{conversation_id}/history - Get the chat history for a conversation

### Option 2: Using curl

1. List available documents:
```bash
curl http://localhost:8000/documents
```

2. Create a new conversation (optionally with specific documents):
```bash
# Create conversation with all documents
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{}'

# Create conversation with specific documents
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["document_1.pdf", "document_2.pdf"]}'
```

3. Send a message (replace "1" with your conversation_id):
```bash
curl -X POST http://localhost:8000/conversations/1/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you tell me about AI agents?"}'
```

4. Get conversation history:
```bash
curl http://localhost:8000/conversations/1/history
```

### Option 3: Using Python
```python
import requests

# List available documents
documents = requests.get("http://localhost:8000/documents").json()
print("Available documents:", documents)

# Create a new conversation
response = requests.post("http://localhost:8000/conversations", json={})
conversation_id = response.json()["conversation_id"]

# Send a message
message_response = requests.post(
    f"http://localhost:8000/conversations/{conversation_id}/messages",
    json={"message": "What can you tell me about AI agents?"}
)
print("Response:", message_response.json()["response"])

# Get conversation history
history = requests.get(f"http://localhost:8000/conversations/{conversation_id}/history").json()
print("Chat history:", history)
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc