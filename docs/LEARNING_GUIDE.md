# ChatPDF: Project Guide

A "chat with your PDF" app built with **Retrieval-Augmented Generation (RAG)** on a fully free stack: no OpenAI, no paid services, everything runs locally or on free tiers.

You upload a PDF, the app indexes it, and you ask questions. Answers come from the PDF's own content and cite the pages they came from.

---

## 1. The big idea: what RAG is and why it exists

An LLM only knows what it was trained on. It doesn't know your PDF. You could paste the whole PDF into the prompt, but:

- context windows are limited, and long prompts are slow (especially on a laptop CPU)
- small models get lost in long inputs
- you can't easily tell which part of the document an answer came from

**RAG fixes this.** It finds the few passages relevant to the question and gives only those to the LLM, with the instruction "answer using only this text."

> Analogy: the vector database is the **source of truth**. The LLM is a **stateless formatting service** that turns retrieved rows into a readable answer. It is not where the facts come from.

---

## 2. The pipeline end to end

There are two flows.

```
INDEXING (once per PDF)                       QUERY (every question)
───────────────────────                       ──────────────────────
PDF ──► extract text ──► chunk ──► embed ──►  [Chroma] ◄── embed question
                                     │            │
                                     └─ store ────┘
                                                  ▼
                                      top-k most similar chunks
                                                  ▼
                                   prompt = rules + chunks + question
                                                  ▼
                                            LLM ──► grounded answer
```

| Stage | What it does | Engineering analogy you already know |
|---|---|---|
| **Ingest** | Extract raw text from each PDF page, keeping page numbers | The "Extract" in ETL |
| **Chunk** | Split text into small passages (~800 chars) that overlap a little | Paging or sharding: retrieve passages, not whole books |
| **Embed** | Turn each chunk into a vector (384 numbers) that captures its meaning | A hash function where *similar meaning gives nearby values* |
| **Store** | Save vector + text + metadata (file, page) | A DB table with a nearest-neighbour index instead of a B-tree |
| **Retrieve** | Embed the question and fetch the k closest chunks | `SELECT TOP k ... ORDER BY similarity DESC`, matching on meaning rather than keywords |
| **Generate** | The LLM answers from the retrieved chunks only | A stateless service call whose input is the query result |

### Key concepts

- **Embedding:** a fixed-length list of numbers representing a text's meaning. "How do I reset my password?" and "steps to change login credentials" share almost no words, but their vectors are close. This is **semantic search**, compared with keyword search (`LIKE '%password%'`).
- **Similarity (cosine):** measures the angle between two vectors. 1 means the same meaning; around 0 means unrelated.
- **Vector database:** stores vectors and answers "give me the k nearest vectors to this one" quickly, using an approximate index (HNSW), the way SQL Server uses B-tree indexes.
- **Chunking:** embedding models have an input limit, and smaller passages make retrieval more precise. **Overlap** repeats the tail of one chunk at the start of the next, so a sentence cut at a boundary isn't lost.
- **Grounding:** the prompt tells the LLM to answer only from the provided context and to say "I don't know" otherwise. This reduces hallucination.

---

## 3. Tech stack (all free)

| Concern | Choice | Why | Alternative |
|---|---|---|---|
| Language | Python 3.12 | Best AI/ML ecosystem; 3.12 has reliable wheels for torch/chromadb | 3.11 |
| PDF parsing | `pypdf` | Pure Python, no native deps | PyMuPDF (better extraction, AGPL licence) |
| Chunking | `langchain-text-splitters` | Proven recursive splitter, without pulling in all of LangChain | A custom splitter |
| Embeddings | `sentence-transformers` + `all-MiniLM-L6-v2` | ~90 MB, fast on CPU, 384 dims, no key | `BAAI/bge-small-en-v1.5` (better quality, 512 tokens) |
| Vector DB | `chromadb` (embedded, persistent) | Runs in-process like SQLite, saves to `data/chroma/` | FAISS, Qdrant, pgvector |
| LLM (default) | **Ollama + `llama3.2:3b`** | Free, offline, private, no rate limits | `qwen2.5:1.5b` / `llama3.2:1b` (lighter) |
| LLM (optional) | **Groq free tier** (`llama-3.1-8b-instant`) | Very fast, better quality | HF Inference (least reliable free tier) |
| Config | `pydantic-settings` + `.env` | Typed, validated settings | n/a |
| UI | Streamlit | Python-only web UI in a few lines | Gradio |
| Later | FastAPI + Docker + Azure | Production API and deployment | n/a |

### LLM trade-offs for this laptop (16 GB RAM, Intel Iris Xe, no NVIDIA GPU)

| | Ollama (local) | Groq (free hosted) |
|---|---|---|
| Key needed | No | Free key |
| Works offline / data stays private | Yes | No |
| Speed | ~5–15 tokens/s on CPU (an answer takes ~10–30 s) | Hundreds of tokens/s |
| Quality | Fine for grounded Q&A; follows instructions less reliably | Noticeably better |
| Risks | RAM use (~3–4 GB for 3B) | Rate limits; model names get retired |

Both will sit behind one interface (like `ILlmClient` in .NET), so switching is a config change: `CHATPDF_LLM_PROVIDER=ollama|groq`.

---

## 4. Folder structure

```
ChatPDF/
├── app/                  # Core logic, no UI or web code (like a .NET Core class library)
│   ├── config.py         # Typed settings
│   ├── pdf_loader.py     # PDF -> pages
│   ├── chunker.py        # pages -> chunks
│   ├── embeddings.py     # text -> vectors (MiniLM)
│   ├── vector_store.py   # Chroma repository: add / search
│   ├── llm.py            # LLMClient interface + Ollama / Groq / HF implementations
│   └── rag.py            # RAGService: index_pdf() / ask()
├── ui/streamlit_app.py   # Streamlit front end
├── scripts/              # check_setup, demo, make_sample_pdf
├── tests/                # pytest unit + integration tests
├── docs/LEARNING_GUIDE.md  # This file
├── data/                 # pdfs/ + chroma/ (git-ignored, created at runtime)
├── pyproject.toml        # Project metadata + pytest/ruff config (~ .csproj)
├── requirements.txt      # Pinned runtime deps
├── requirements-dev.txt  # + pytest, ruff, fpdf2
├── .env.example          # Config template (committed)
├── .env                  # Your local config (never committed)
└── README.md
```

**Why `app/` has no UI code:** Streamlit (Step 7) and FastAPI (Step 8) are just two front doors calling the same core. This is the Core / Web project split from .NET, and it's why Step 8 will be a thin wrapper rather than a rewrite.

---

## 5. Configuration pattern (`app/config.py`)

| Python | .NET equivalent |
|---|---|
| `class Settings(BaseSettings)` | Options POCO bound from `appsettings.json` |
| `.env` file | `appsettings.Development.json` / user-secrets |
| Environment variables (`CHATPDF_*`) | Azure App Service "App Settings" |
| `@lru_cache get_settings()` | `services.Configure<T>()` + `IOptions<T>` singleton |
| `SecretStr` | Hides secrets in logs |
| `@model_validator` | `ValidateOnStart()` on options |

**Precedence:** real environment variables, then `.env`, then code defaults. Locally you use `.env`; in Azure you set App Settings and the code doesn't change.

**Current settings:**

| Setting | Default | Meaning |
|---|---|---|
| `CHATPDF_CHUNK_SIZE` | 800 | Characters per chunk (~150–200 tokens) |
| `CHATPDF_CHUNK_OVERLAP` | 150 | Characters shared between neighbouring chunks |
| `CHATPDF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHATPDF_TOP_K` | 4 | Chunks retrieved per question |
| `CHATPDF_LLM_PROVIDER` | `ollama` | `ollama` or `groq` |
| `CHATPDF_OLLAMA_MODEL` | `llama3.2:3b` | Local model name |
| `CHATPDF_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `CHATPDF_GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model (check console for current names) |
| `CHATPDF_GROQ_API_KEY` | none | Required only when provider is `groq` |
| `CHATPDF_TEMPERATURE` | 0.1 | Low means factual and repeatable answers |

---

## 6. Design decisions and trade-offs

| Decision | Choice | Trade-off |
|---|---|---|
| **Chunk size** | 800 chars | Smaller chunks are more precise but lose context; bigger chunks carry more context but are noisier, and anything past 256 tokens gets **silently truncated** by MiniLM |
| **Overlap** | 150 chars (~20%) | More overlap means fewer broken sentences but more duplicate storage and duplicate retrieval hits |
| **Top-k** | 4 | More chunks raise the chance the answer is in the context, but give a longer, slower prompt and more distraction for a small model |
| **Embedding model** | MiniLM | Fast and small, compared with bge-small (better retrieval, about the same cost) |
| **LLM** | Local 3B | Free, private and no limits, but slower and lower quality than hosted 8B+ |
| **Temperature** | 0.1 | Grounded and repeatable, but less "creative" (fine for Q&A) |

> Golden rule: **the embedding model used at indexing time must be the same one used at query time.** If you change models, re-index everything, the same way changing a hash function invalidates the old hash index.

---

## 7. Build roadmap

| Step | What | Status |
|---|---|---|
| 1 | Project setup: venv, dependencies, config, smoke test | **Done** |
| 2 | Load PDF and extract text page by page (`app/pdf_loader.py`) | **Done** |
| 3 | Chunk text with overlap, keep metadata (`app/chunker.py`) | **Done** |
| 4 | Embed locally and store in Chroma (`app/vector_store.py`) | **Done** |
| 5 | Retrieval: top-k similar chunks (`VectorStore.search`) | **Done** |
| 6 | Generation: grounded prompt to an LLM behind one interface (`app/llm.py`, `app/rag.py`) | **Done** |
| 7 | Streamlit UI: upload PDF and chat (`ui/streamlit_app.py`) | **Done** |

**LLM providers (`CHATPDF_LLM_PROVIDER`):** `auto` (default) uses Ollama if it's running, else Groq if a key is set, else a local Hugging Face model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB, runs in-process on CPU, ~20–30 s per answer).

**Run it:**
```powershell
pip install -r requirements-dev.txt; pip install -e .   # once: deps + make `app` importable
python -m scripts.make_sample_pdf          # creates data/pdfs/sample_handbook.pdf
python -m scripts.demo                     # CLI: index + 4 sample questions
streamlit run streamlit_app.py          # UI at http://localhost:8501
```
| 8 | FastAPI + Docker, deploy to Azure | Later |

---

## 8. Common pitfalls

- **Wrong Python version:** 3.13/3.14 may lack torch/chromadb wheels. Use the `.venv` built from 3.12.
- **Running scripts from the wrong folder:** always run from the project root with `python -m scripts.<name>`, or `import app` fails.
- **PowerShell won't activate the venv:** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **Scanned PDFs:** they contain images, not text, so pypdf returns empty strings and OCR is needed. We'll detect this in Step 2.
- **Silent truncation:** chunks longer than the embedding model's token limit lose their tail without any error.
- **Changing the embedding model without re-indexing:** retrieval quietly returns garbage.
- **Committing `.env`:** it's git-ignored; keep it that way.
- **First run is slow:** the embedding model (~90 MB) and the Ollama model (~2 GB) download once, then are cached.

---

## 9. How to run

From the `ChatPDF` folder in PowerShell:

```powershell
# one-time setup
py -V:Astral/CPython3.12.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env

# Step 1 check
python -m scripts.check_setup     # expect "=== ALL GOOD ==="

# for Step 6 (local LLM): install Ollama from https://ollama.com, then
ollama pull llama3.2:3b
```

---

## 10. Hardware notes (this laptop)

- **16 GB RAM** is enough for embeddings plus a 3B model at the same time.
- **No NVIDIA GPU**, so everything runs on the CPU. Embedding a 100-page PDF takes a few seconds to a minute; answers from a local LLM take ~10–30 s.
- **If it's too slow:** use `llama3.2:1b` / `qwen2.5:1.5b`, lower `TOP_K` to 3, or switch to Groq for generation (embeddings stay local and free).
