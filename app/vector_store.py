"""Chroma-backed vector store: persist chunk embeddings and run similarity search.

Think of it as a repository class over a table (id, text, vector, namespace, source, page) whose
index answers "nearest neighbours to this vector" instead of "rows where column = value".

Namespaces isolate tenants (e.g. one per browser session on the public demo), like a TenantId
column that every query filters on.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.chunker import Chunk
from app.embeddings import Embedder

DEFAULT_NAMESPACE = "default"


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit. `score` is cosine similarity: 1.0 = same meaning, ~0 = unrelated."""

    text: str
    source: str
    page: int
    score: float


def _where(namespace: str, source: str | None = None) -> dict[str, Any]:
    """Build a Chroma metadata filter (Chroma needs an explicit $and for several conditions)."""
    if source is None:
        return {"namespace": namespace}
    return {"$and": [{"namespace": namespace}, {"source": source}]}


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

    def add_chunks(self, chunks: list[Chunk], namespace: str = DEFAULT_NAMESPACE) -> None:
        """Index chunks. Re-indexing a file replaces its old chunks (idempotent, no duplicates)."""
        if not chunks:
            return
        for source in {c.source for c in chunks}:
            self.delete_source(source, namespace)
        self._collection.upsert(
            # Namespace is part of the id so two tenants uploading "report.pdf" never collide.
            ids=[f"{namespace}::{c.id}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self._embedder.embed([c.text for c in chunks]),
            metadatas=[{"namespace": namespace, "source": c.source, "page": c.page} for c in chunks],
        )

    def delete_source(self, source: str, namespace: str = DEFAULT_NAMESPACE) -> None:
        """Remove every chunk of one document in a namespace."""
        self._collection.delete(where=_where(namespace, source))

    def delete_namespace(self, namespace: str) -> None:
        """Remove every chunk in a namespace."""
        self._collection.delete(where=_where(namespace))

    def search(
        self, query: str, top_k: int, source: str | None = None, namespace: str = DEFAULT_NAMESPACE
    ) -> list[RetrievedChunk]:
        """Return the `top_k` chunks most similar to `query` within a namespace (optionally one document)."""
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=top_k,
            where=_where(namespace, source),
        )
        return [
            RetrievedChunk(text=doc, source=meta["source"], page=int(meta["page"]), score=1.0 - dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def sources(self, namespace: str = DEFAULT_NAMESPACE) -> list[str]:
        """Names of all documents indexed in a namespace."""
        metadatas = self._collection.get(where=_where(namespace), include=["metadatas"])["metadatas"] or []
        return sorted({str(m["source"]) for m in metadatas})
