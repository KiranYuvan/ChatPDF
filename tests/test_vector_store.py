"""Integration tests: real MiniLM embeddings + a real (temporary) Chroma database."""

import pytest

from app.chunker import Chunk
from app.config import Settings
from app.embeddings import Embedder
from app.vector_store import VectorStore

pytestmark = pytest.mark.integration

CHUNKS = [
    Chunk("a::p1::c0", "Employees receive 24 days of paid annual leave per year.", "a.pdf", 1),
    Chunk("a::p2::c0", "Hotel stays are capped at 220 US dollars per night.", "a.pdf", 2),
    Chunk("a::p3::c0", "Lost laptops must be reported to IT within 24 hours.", "a.pdf", 3),
    Chunk("b::p1::c0", "The cafeteria serves vegetarian lunch on Fridays.", "b.pdf", 1),
]


@pytest.fixture
def store(settings: Settings) -> VectorStore:
    vs = VectorStore(settings.chroma_dir, "test", Embedder(settings.embedding_model))
    vs.add_chunks(CHUNKS)
    return vs


def test_empty_store_returns_no_results(settings: Settings) -> None:
    vs = VectorStore(settings.chroma_dir, "empty", Embedder(settings.embedding_model))
    assert vs.search("anything", top_k=3) == []


def test_most_relevant_chunk_ranks_first(store: VectorStore) -> None:
    hits = store.search("How much vacation time do I get?", top_k=3)
    assert len(hits) == 3
    assert hits[0].page == 1 and "annual leave" in hits[0].text
    assert hits[0].score > hits[1].score


def test_source_filter_limits_results(store: VectorStore) -> None:
    hits = store.search("lunch", top_k=4, source="a.pdf")
    assert {h.source for h in hits} == {"a.pdf"}


def test_reindexing_replaces_instead_of_duplicating(store: VectorStore) -> None:
    store.add_chunks([c for c in CHUNKS if c.source == "a.pdf"])
    hits = store.search("hotel", top_k=10, source="a.pdf")
    assert len(hits) == 3
    assert store.sources() == ["a.pdf", "b.pdf"]
