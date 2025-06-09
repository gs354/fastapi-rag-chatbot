import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_rag_chatbot.api import routes


class MockDependencies:
    def __init__(self, mocker):
        self.vector_store_service = mocker.MagicMock()
        self.vector_store_service.document_metadata = {
            "test_doc.pdf": {
                "title": "Test Document",
                "author": "Test Author",
                "date": "2025-01-01",
            }
        }
        self.vector_store_service.index = object()
        self.vector_store_service.vector_store = object()
        self.vector_store_service.documents = object()
        self.system_prompt = "Test system prompt"


@pytest.fixture
def app(mocker):
    app = FastAPI()
    # Patch Conversation.get_chat_engine for all tests that need it
    mock_chat_engine = mocker.MagicMock()
    mock_chat_engine.chat = mocker.MagicMock(
        return_value=type("Resp", (), {"response": "Test AI response"})()
    )
    mocker.patch(
        "fastapi_rag_chatbot.api.conversation.Conversation.get_chat_engine",
        return_value=mock_chat_engine,
    )
    # Dependency override
    app.dependency_overrides[routes.get_dependencies] = lambda: MockDependencies(mocker)
    app.include_router(routes.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_list_documents(client):
    response = client.get("/documents")
    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 1
    assert documents[0]["file_name"] == "test_doc.pdf"
    assert documents[0]["title"] == "Test Document"
    assert documents[0]["author"] == "Test Author"
    assert documents[0]["date"] == "2025-01-01"


def test_create_conversation(client):
    response = client.post("/conversations", json={})
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["document_ids"] is None


def test_create_conversation_with_specific_documents(client):
    response = client.post("/conversations", json={"document_ids": ["test_doc.pdf"]})
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["document_ids"] == ["test_doc.pdf"]


def test_send_message(client):
    conv_response = client.post("/conversations", json={})
    conversation_id = conv_response.json()["conversation_id"]
    response = client.post(
        f"/conversations/{conversation_id}/messages", json={"message": "Hello, AI!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["conversation_id"] == conversation_id
    assert data["response"] == "Test AI response"


def test_get_chat_history(client):
    conv_response = client.post("/conversations", json={})
    conversation_id = conv_response.json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/messages", json={"message": "Hello, AI!"}
    )
    response = client.get(f"/conversations/{conversation_id}/history")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == conversation_id
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Hello, AI!"
    assert data["messages"][1]["role"] == "assistant"
    assert data["messages"][1]["content"] == "Test AI response"


def test_invalid_conversation_id(client):
    response = client.get("/conversations/invalid_id/history")
    assert response.status_code == 404
