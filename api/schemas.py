"""Request/response contracts (DTOs). Kept separate from domain dataclasses so the public API
can evolve without changing app/ internals - same reason you don't return EF entities from controllers."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class DocumentIndexed(BaseModel):
    source: str = Field(examples=["handbook.pdf"])
    pages: int = Field(examples=[5])
    chunks: int = Field(examples=[5])


class DocumentList(BaseModel):
    documents: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What is the hotel cap per night?"])
    source: str | None = Field(default=None, description="Limit the search to one document")


class SourceChunk(BaseModel):
    source: str
    page: int
    score: float = Field(description="Cosine similarity, 1.0 = identical meaning")
    text: str


class AskResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceChunk]
