import logging

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore

logger = logging.getLogger(__name__)


class Conversation:
    def __init__(self, conversation_id: str, document_ids: set[str] | None = None):
        self.conversation_id = conversation_id
        self.messages: list[dict] = []
        self.document_ids = document_ids or set()

    def get_chat_engine(
        self,
        index: VectorStoreIndex,
        vector_store: WeaviateVectorStore,
        documents: list,
        system_prompt: str,
    ):
        # Create a filtered index if document_ids are specified
        if self.document_ids:
            filtered_docs = [
                doc
                for doc in documents
                if doc.metadata.get("file_name") in self.document_ids
            ]
            filtered_index = VectorStoreIndex.from_documents(
                filtered_docs, vector_store=vector_store
            )
            return filtered_index.as_chat_engine(
                chat_mode="context",
                system_prompt=system_prompt,
            )
        else:
            # Use the full index if no document_ids specified
            return index.as_chat_engine(
                chat_mode="context",
                system_prompt=system_prompt,
            )


class ConversationManager:
    def __init__(self):
        self.conversations: dict[str, Conversation] = {}

    def create_conversation(
        self, conversation_id: str, document_ids: set[str] | None = None
    ) -> Conversation:
        if conversation_id in self.conversations:
            raise ValueError(f"Conversation {conversation_id} already exists")
        conversation = Conversation(conversation_id, document_ids)
        self.conversations[conversation_id] = conversation
        logger.info(
            f"Created new conversation with ID: {conversation_id} using documents: {document_ids}"
        )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        return self.conversations[conversation_id]
