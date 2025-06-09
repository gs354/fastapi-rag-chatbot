FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install latest Poetry
RUN pip install --no-cache-dir poetry

# Copy all application code first
COPY . .

# Configure Poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install only main dependencies (equivalent to old --no-dev)
RUN poetry install --only main

# Create data directory
RUN mkdir -p data/ai-agents-arxiv-papers

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "fastapi_rag_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
