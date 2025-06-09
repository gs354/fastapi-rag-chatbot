from dataclasses import dataclass
from typing import Protocol

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore


class VectorStoreServiceProtocol(Protocol):
    """Protocol defining the required interface for vector store services."""

    document_metadata: dict
    index: VectorStoreIndex
    vector_store: WeaviateVectorStore
    documents: list


@dataclass
class Dependencies:
    """Application dependencies container."""

    vector_store_service: VectorStoreServiceProtocol
    system_prompt: str
