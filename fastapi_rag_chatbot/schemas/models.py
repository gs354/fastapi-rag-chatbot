from pydantic import BaseModel


class MessageRequest(BaseModel):
    message: str


class Message(BaseModel):
    role: str
    content: str


class MessageResponse(BaseModel):
    response: str
    conversation_id: str


class ConversationRequest(BaseModel):
    document_ids: list[str] | None = None


class ConversationResponse(BaseModel):
    conversation_id: str
    document_ids: list[str] | None = None


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[Message]
    document_ids: list[str] | None = None


class DocumentInfo(BaseModel):
    file_name: str
    title: str | None = None
    author: str | None = None
    date: str | None = None
