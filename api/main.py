"""FastAPI service exposing ChatPDF as a REST API.

Run locally:   uvicorn api.main:app --reload            → http://localhost:8000/docs (Swagger UI)

.NET analogy:
    FastAPI()            ~ WebApplication / minimal APIs
    @app.post(...)       ~ app.MapPost(...) / [HttpPost] controller action
    Depends(...)         ~ constructor/parameter injection from the DI container
    Pydantic models      ~ DTOs with DataAnnotations validation (bad input -> automatic 422)
    /docs                ~ Swashbuckle Swagger UI, generated from the code
"""

import re
import secrets
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status

from api.schemas import AskRequest, AskResponse, DocumentIndexed, DocumentList, HealthResponse, SourceChunk
from app.config import get_settings
from app.llm import LLMError
from app.logging_config import configure_logging
from app.rag import RAGService
from app.vector_store import DEFAULT_NAMESPACE

NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(
    title="ChatPDF API",
    version="0.2.0",
    description="Upload PDFs and ask questions answered from their content (RAG).",
)


# ---- Dependencies -----------------------------------------------------------------------------


@lru_cache
def get_service() -> RAGService:
    """Singleton RAGService, created on first request (models load lazily, not at import time)."""
    configure_logging(get_settings().log_level)
    return RAGService()


ServiceDep = Annotated[RAGService, Depends(get_service)]


def require_api_key(service: ServiceDep, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """If CHATPDF_API_KEY is configured, require a matching X-API-Key header (like an auth filter)."""
    expected = service.settings.api_key
    if expected is None:
        return
    # compare_digest takes constant time, so the key can't be guessed by measuring response times.
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key header")


def get_namespace(x_namespace: Annotated[str | None, Header()] = None) -> str:
    """Tenant/workspace id from the X-Namespace header (defaults to a shared 'default')."""
    if x_namespace is None:
        return DEFAULT_NAMESPACE
    if not NAMESPACE_PATTERN.fullmatch(x_namespace):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "X-Namespace must be 1-64 chars: letters, digits, - or _"
        )
    return x_namespace


NamespaceDep = Annotated[str, Depends(get_namespace)]
protected = [Depends(require_api_key)]


# ---- Endpoints --------------------------------------------------------------------------------
# Plain `def` (not `async def`) on purpose: PDF parsing, embedding and LLM calls are blocking,
# CPU-heavy work. FastAPI runs `def` endpoints on a thread pool, so they don't freeze the event loop
# (the equivalent of not calling .Result on a Task inside ASP.NET Core).


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe for Docker / Azure. Cheap: does not load any model."""
    return HealthResponse()


@app.post(
    "/documents",
    response_model=DocumentIndexed,
    status_code=status.HTTP_201_CREATED,
    dependencies=protected,
    tags=["documents"],
)
def upload_document(
    service: ServiceDep, namespace: NamespaceDep, file: Annotated[UploadFile, File(description="PDF file")]
) -> DocumentIndexed:
    """Upload and index a PDF. Uploading the same file name again replaces it."""
    limit = service.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # read at most limit+1 bytes: never buffer a huge upload fully
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Limit is {service.settings.max_upload_mb} MB"
        )
    try:
        result = service.index_pdf(data, file.filename or "document.pdf", namespace)
    except ValueError as exc:  # unreadable / encrypted / scanned / too large
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return DocumentIndexed(source=result.source, pages=result.pages, chunks=result.chunks)


@app.get("/documents", response_model=DocumentList, dependencies=protected, tags=["documents"])
def list_documents(service: ServiceDep, namespace: NamespaceDep) -> DocumentList:
    """List the documents indexed in this namespace."""
    return DocumentList(documents=service.documents(namespace))


@app.delete(
    "/documents/{name}", status_code=status.HTTP_204_NO_CONTENT, dependencies=protected, tags=["documents"]
)
def delete_document(name: str, service: ServiceDep, namespace: NamespaceDep) -> None:
    """Delete one document and all its chunks."""
    if not service.delete_document(name, namespace):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{name}' not found")


@app.post("/ask", response_model=AskResponse, dependencies=protected, tags=["chat"])
def ask(body: AskRequest, service: ServiceDep, namespace: NamespaceDep) -> AskResponse:
    """Answer a question from the indexed documents, with the source chunks used."""
    try:
        answer = service.ask(body.question, source=body.source, namespace=namespace)
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return AskResponse(
        answer=answer.text,
        model=answer.model,
        sources=[
            SourceChunk(source=s.source, page=s.page, score=round(s.score, 4), text=s.text)
            for s in answer.sources
        ],
    )
