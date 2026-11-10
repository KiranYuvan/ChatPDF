"""Central, typed configuration.

.NET analogy:
    Settings       ~ an options class bound from appsettings.json
    .env           ~ user-secrets / appsettings.Development.json (git-ignored, never committed)
    get_settings() ~ services.Configure<T>() + IOptions<T> (one cached instance)

Precedence (highest wins): real environment variables > .env file > defaults below.
In Azure you set App Settings (env vars) and no .env file exists at all.
Secrets have NO default and are never written to code or committed files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProvider = Literal["auto", "ollama", "groq", "huggingface"]


class Settings(BaseSettings):
    """All runtime settings. Every field can be overridden with a CHATPDF_<NAME> env var."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="CHATPDF_",
        extra="ignore",
    )

    # --- Storage ---------------------------------------------------------------
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "pdf_chunks"

    # --- Chunking --------------------------------------------------------------
    # Characters. ~800 chars ≈ 150-200 tokens: under all-MiniLM-L6-v2's 256-token limit,
    # beyond which input is silently truncated.
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Embeddings ------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval -------------------------------------------------------------
    top_k: int = Field(default=4, ge=1, le=20)

    # --- Generation ------------------------------------------------------------
    # auto = Ollama if reachable, else Groq if a key is set, else the local Hugging Face model.
    llm_provider: LLMProvider = "auto"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=256, gt=0)

    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

    groq_model: str = "llama-3.1-8b-instant"
    # Secret: read from CHATPDF_GROQ_API_KEY or the conventional GROQ_API_KEY. SecretStr masks it in logs.
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("CHATPDF_GROQ_API_KEY", "GROQ_API_KEY")
    )

    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB, runs in-process on CPU

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        """Fail fast at startup (like ValidateOnStart) instead of misbehaving at runtime."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )
        if self.llm_provider == "groq" and self.groq_api_key is None:
            raise ValueError(
                "llm_provider=groq requires GROQ_API_KEY (free key: https://console.groq.com/keys)"
            )
        return self

    def ensure_dirs(self) -> None:
        """Create the data folders if they don't exist yet."""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the single shared Settings instance (read from env/.env once per process)."""
    return Settings()
