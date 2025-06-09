import logging
import time

from fastapi import APIRouter, Body, Depends, HTTPException

from fastapi_rag_chatbot.api.conversation import ConversationManager
from fastapi_rag_chatbot.dependencies import Dependencies
from fastapi_rag_chatbot.schemas.models import (
    ChatHistoryResponse,
    ConversationRequest,
    ConversationResponse,
    DocumentInfo,
    MessageRequest,
    MessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()
conversation_manager = ConversationManager()


def get_dependencies() -> Dependencies:
    """Get application dependencies.

    This function will be overridden in main.py with actual dependencies.
    """
    raise NotImplementedError("Dependencies must be overridden in main.py")


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(deps: Dependencies = Depends(get_dependencies)):
    """List all available documents that can be used for context."""
    return [
        DocumentInfo(
            file_name=file_name,
            title=metadata.get("title"),
            author=metadata.get("author"),
            date=metadata.get("date"),
        )
        for file_name, metadata in deps.vector_store_service.document_metadata.items()
    ]


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest = Body(
        default={},
        openapi_examples={
            "all": {"summary": "Create conversation with all documents", "value": {}},
            "specific": {
                "summary": "Create conversation with specific documents",
                "value": {
                    "document_ids": [
                        "Asadi et al. - 2009 - A Framework For Intelligent Multi Agent System Bas.pdf"
                    ]
                },
            },
        },
        description="Create a new conversation. Leave request body empty to use all available documents (default).",
    ),
):
    """Create conversations with an id number and, optionally, specific documents to be indexed"""
    conversation_id = str(len(conversation_manager.conversations) + 1)
    document_ids = set(request.document_ids) if request.document_ids else None
    conversation_manager.create_conversation(conversation_id, document_ids)
    return ConversationResponse(
        conversation_id=conversation_id,
        document_ids=list(document_ids) if document_ids else None,
    )


@router.post(
    "/conversations/{conversation_id}/messages", response_model=MessageResponse
)
async def send_message(
    conversation_id: str,
    request: MessageRequest,
    deps: Dependencies = Depends(get_dependencies),
):
    """Send a message to the LLM for a response. Add both to conversation history."""
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        logger.info(f"Processing message for conversation {conversation_id}")

        # Add user message to conversation history
        conversation.messages.append({"role": "user", "content": request.message})

        try:
            # Get chat engine with appropriate document context
            chat_engine = conversation.get_chat_engine(
                deps.vector_store_service.index,
                deps.vector_store_service.vector_store,
                deps.vector_store_service.documents,
                deps.system_prompt,
            )

            # Get response from LLM
            start_time = time.time()
            response = chat_engine.chat(request.message)
            processing_time = time.time() - start_time

            logger.info(f"Request processed in {processing_time:.2f} seconds")

            # Add assistant response to conversation history
            conversation.messages.append(
                {"role": "assistant", "content": response.response}
            )

            return MessageResponse(
                response=response.response, conversation_id=conversation_id
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing message: {str(e)}"
            )

    except ValueError as e:
        logger.warning(f"Conversation not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/conversations/{conversation_id}/history", response_model=ChatHistoryResponse
)
async def get_chat_history(conversation_id: str):
    """Retrieve the chat history between user and LLM."""
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        logger.info(f"Retrieved chat history for conversation {conversation_id}")
        return ChatHistoryResponse(
            conversation_id=conversation_id,
            messages=conversation.messages,
            document_ids=list(conversation.document_ids)
            if conversation.document_ids
            else None,
        )
    except ValueError as e:
        logger.warning(f"Conversation not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
