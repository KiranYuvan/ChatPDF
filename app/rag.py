"""RAGService: the application service the UI (and later FastAPI) talks to.

Indexing:  PDF bytes -> pages -> chunks -> embeddings -> vector store
Querying:  question -> top-k chunks -> grounded prompt -> LLM -> answer + sources
"""

from dataclasses import dataclass

from app.chunker import chunk_pages
from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.llm import LLMClient, create_llm
from app.pdf_loader import load_pdf
from app.vector_store import RetrievedChunk, VectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a PDF document. "
    "Use ONLY the context excerpts provided. If the answer is not in the context, reply exactly: "
    '"I couldn\'t find that in the document." Keep answers concise and cite pages like (page 3).'
)

NO_DOCUMENTS = "No documents indexed yet. Upload a PDF first."


@dataclass(frozen=True)
class IndexResult:
    source: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk]
    model: str


def build_prompt(question: str, hits: list[RetrievedChunk]) -> str:
    """Put retrieved chunks (labelled with page numbers) and the question into one user prompt."""
    context = "\n\n".join(f"[page {h.page}]\n{h.text}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class RAGService:
    """Orchestrates indexing and question answering.

    Dependencies can be injected (constructor injection, like in ASP.NET Core) so tests can
    pass a temporary store or a fake LLM; otherwise they are built from settings.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or VectorStore(
            self.settings.chroma_dir, self.settings.collection_name, Embedder(self.settings.embedding_model)
        )
        self._llm = llm  # created lazily: indexing never needs the (slow to load) LLM

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    def index_pdf(self, data: bytes, filename: str) -> IndexResult:
        """Extract, chunk, embed and store a PDF. Re-indexing the same filename replaces it."""
        pages = load_pdf(data)
        chunks = chunk_pages(pages, filename, self.settings.chunk_size, self.settings.chunk_overlap)
        self.store.add_chunks(chunks)
        return IndexResult(source=filename, pages=len(pages), chunks=len(chunks))

    def ask(self, question: str, source: str | None = None) -> Answer:
        """Answer `question` from the indexed content (optionally only from document `source`)."""
        hits = self.store.search(question, self.settings.top_k, source)
        if not hits:
            return Answer(NO_DOCUMENTS, [], "-")
        reply = self.llm.chat(SYSTEM_PROMPT, build_prompt(question, hits))
        return Answer(text=reply, sources=hits, model=self.llm.name)
