import logging
import time

import weaviate
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.weaviate import WeaviateVectorStore

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, config: dict, weaviate_host: str, ollama_host: str):
        # Initialize Weaviate client
        self.client = weaviate.connect_to_local(host=weaviate_host)

        # Initialize Ollama models
        self.llm = Ollama(
            model=config["models"]["llm_model"],
            request_timeout=config["models"]["llm_timeout"],
            base_url=ollama_host,
        )
        self.embed_model = OllamaEmbedding(
            model_name=config["models"]["embedding_model"],
            request_timeout=config["models"]["embedding_timeout"],
            base_url=ollama_host,
        )

        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Initialize vector store
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=config["vector_store"]["index_name"],
            text_key=config["vector_store"]["text_key"],
            by_text=config["vector_store"]["by_text"],
        )

        # Load and index documents
        self.documents = self._load_documents(config["paths"]["data_dir"])
        self.index = self._create_index()
        self.document_metadata = self._create_document_metadata()

    def _load_documents(self, data_dir: str) -> list:
        logger.info("Loading and indexing documents...")
        start_time = time.time()
        documents = SimpleDirectoryReader(data_dir).load_data()
        logger.info(f"Documents loaded in {time.time() - start_time:.2f} seconds")
        return documents

    def _create_index(self) -> VectorStoreIndex:
        start_time = time.time()
        index = VectorStoreIndex.from_documents(
            self.documents, vector_store=self.vector_store
        )
        logger.info(f"Documents indexed in {time.time() - start_time:.2f} seconds")
        return index

    def _create_document_metadata(self) -> dict:
        return {
            doc.metadata.get("file_name", "unknown"): doc.metadata
            for doc in self.documents
        }
