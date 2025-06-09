import logging
import os

import tomli
from dotenv import load_dotenv
from fastapi import FastAPI

from fastapi_rag_chatbot.api.routes import get_dependencies, router
from fastapi_rag_chatbot.api.vector_store import VectorStoreService
from fastapi_rag_chatbot.dependencies import Dependencies

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Load configuration
with open("config.toml", "rb") as f:
    config = tomli.load(f)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config["paths"]["log_file"]),
        logging.StreamHandler(),  # Also log to console
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version=config["app"]["version"],
    description=config["app"]["description"],
)

# Initialize vector store service
vector_store_service = VectorStoreService(
    config=config,
    weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
    ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
)


def get_app_dependencies() -> Dependencies:
    """Get application dependencies with actual implementations."""
    return Dependencies(
        vector_store_service=vector_store_service,
        system_prompt=config["prompt"]["system_prompt"],
    )


# Override dependencies
app.dependency_overrides[get_dependencies] = get_app_dependencies

# Include router
app.include_router(router)
