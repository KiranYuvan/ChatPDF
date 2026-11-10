"""API tests through FastAPI's in-memory TestClient (like WebApplicationFactory in ASP.NET Core)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.main import app, get_service
from app.config import Settings
from app.llm import LLMError
from app.rag import RAGService
from tests.conftest import FakeLLM, make_pdf

pytestmark = pytest.mark.integration

PDF = make_pdf(["Paid annual leave is 24 days per year.", "Hotels are capped at 220 dollars per night."])


def _client(service: RAGService) -> Iterator[TestClient]:
    app.dependency_overrides[get_service] = lambda: service  # swap the DI registration
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM) -> Iterator[TestClient]:
    yield from _client(RAGService(settings, llm=fake_llm))


def upload(client: TestClient, name: str = "handbook.pdf", data: bytes = PDF, **headers: str):
    return client.post("/documents", files={"file": (name, data, "application/pdf")}, headers=headers)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete_roundtrip(client: TestClient) -> None:
    r = upload(client)
    assert r.status_code == 201 and r.json() == {"source": "handbook.pdf", "pages": 2, "chunks": 2}
    assert client.get("/documents").json() == {"documents": ["handbook.pdf"]}

    r = client.post("/ask", json={"question": "What is the hotel limit?", "source": "handbook.pdf"})
    body = r.json()
    assert r.status_code == 200 and body["answer"] == "fake answer"
    assert body["sources"][0]["page"] == 2

    assert client.delete("/documents/handbook.pdf").status_code == 204
    assert client.delete("/documents/handbook.pdf").status_code == 404
    assert client.get("/documents").json() == {"documents": []}


def test_invalid_pdf_is_400(client: TestClient) -> None:
    r = upload(client, data=b"not a pdf")
    assert r.status_code == 400 and "Could not read" in r.json()["detail"]


def test_too_large_is_413(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings.model_copy(update={"max_upload_mb": 1}), llm=fake_llm)
    for client in _client(service):
        assert upload(client, data=b"0" * (2 * 1024 * 1024)).status_code == 413


def test_empty_question_is_422(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422  # Pydantic validation


def test_namespaces_isolate_tenants(client: TestClient) -> None:
    upload(client, **{"X-Namespace": "alice"})
    assert client.get("/documents", headers={"X-Namespace": "bob"}).json() == {"documents": []}
    assert client.get("/documents", headers={"X-Namespace": "bad ns!"}).status_code == 400


def test_llm_outage_is_503(settings: Settings) -> None:
    class Down:
        name = "down"

        def chat(self, system: str, user: str) -> str:
            raise ConnectionError

    for client in _client(RAGService(settings, llm=Down())):
        upload(client)
        r = client.post("/ask", json={"question": "hotel?"})
        assert r.status_code == 503 and "try again" in r.json()["detail"]
    assert issubclass(LLMError, RuntimeError)


def test_api_key_required_when_configured(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings.model_copy(update={"api_key": SecretStr("s3cret")}), llm=fake_llm)
    for client in _client(service):
        assert client.get("/health").status_code == 200  # probes stay open
        assert client.get("/documents").status_code == 401
        assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
