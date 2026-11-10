"""Chroma-backed vector store: persist chunk embeddings and run similarity search.

Think of it as a repository class over a table (id, text, vector, source, page) whose
index answers "nearest neighbours to this vector" instead of "rows where column = value".
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.chunker import Chunk
from app.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit. `score` is cosine similarity: 1.0 = same meaning, ~0 = unrelated."""

    text: str
    source: str
    page: int
    score: float


class VectorStore:
    """Stores chunks with their embeddings and retrieves the most similar ones for a query."""

    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        client = chromadb.PersistentClient(path=str(persist_dir))
        # Cosine distance matches how sentence-transformers vectors are meant to be compared.
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks. Re-indexing a file replaces its old chunks (idempotent, no duplicates)."""
        if not chunks:
            return
        for source in {c.source for c in chunks}:
            self.delete_source(source)
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self._embedder.embed([c.text for c in chunks]),
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )

    def delete_source(self, source: str) -> None:
        """Remove every chunk belonging to one document."""
        self._collection.delete(where={"source": source})

    def search(self, query: str, top_k: int, source: str | None = None) -> list[RetrievedChunk]:
        """Return the `top_k` chunks most similar to `query`, optionally limited to one document."""
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=top_k,
            where={"source": source} if source else None,
        )
        return [
            RetrievedChunk(text=doc, source=meta["source"], page=int(meta["page"]), score=1.0 - dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def sources(self) -> list[str]:
        """Names of all indexed documents."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m["source"]) for m in metadatas})
