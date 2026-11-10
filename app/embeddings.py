"""Local text embeddings with sentence-transformers (no API key, runs on CPU)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


class Embedder:
    """Turns text into normalized vectors whose cosine similarity reflects semantic similarity.

    Rule: the SAME model must embed documents (at index time) and questions (at query time),
    like a hash function - vectors from different models are not comparable.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Normalized output means dot product == cosine similarity."""
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


@lru_cache
def _load_model(model_name: str) -> SentenceTransformer:
    # ~90 MB, downloaded once to the Hugging Face cache, then loaded once per process.
    return SentenceTransformer(model_name, device="cpu")
