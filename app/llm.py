"""One interface, several free LLM backends (like ILlmClient with multiple implementations).

Providers:
    ollama      - local Ollama server (best local option; needs Ollama installed)
    groq        - free hosted tier (needs a free API key, data leaves the machine)
    huggingface - small transformers model running in-process on CPU (zero extra setup, slowest)
    auto        - ollama if reachable, else groq if a key is set, else huggingface
"""

import urllib.request
from typing import Protocol

from app.config import Settings


class LLMClient(Protocol):
    """Structural interface: any class with `name` and `chat()` qualifies (no inheritance needed)."""

    name: str

    def chat(self, system: str, user: str) -> str:
        """Send a system + user message and return the model's reply text."""
        ...


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OllamaClient:
    """Talks to a local Ollama server over HTTP."""

    def __init__(self, settings: Settings) -> None:
        import ollama  # imported lazily so unused providers cost nothing at startup

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.ollama_model
        self._temperature = settings.temperature
        self.name = f"ollama:{settings.ollama_model}"

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self._model, messages=_messages(system, user), options={"temperature": self._temperature}
        )
        return resp["message"]["content"]


class GroqClient:
    """Groq's hosted free tier (OpenAI-style chat completions API)."""

    def __init__(self, settings: Settings) -> None:
        from groq import Groq

        if settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set")
        self._client = Groq(api_key=settings.groq_api_key.get_secret_value())
        self._model = settings.groq_model
        self._temperature = settings.temperature
        self.name = f"groq:{settings.groq_model}"

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=_messages(system, user), temperature=self._temperature
        )
        return resp.choices[0].message.content or ""


class HuggingFaceLocalClient:
    """Runs a small instruct model on CPU inside this Python process."""

    def __init__(self, settings: Settings) -> None:
        from transformers import pipeline

        self._pipe = pipeline("text-generation", model=settings.hf_model, device="cpu", dtype="auto")
        self._max_new_tokens = settings.max_new_tokens
        self.name = f"huggingface:{settings.hf_model}"

    def chat(self, system: str, user: str) -> str:
        out = self._pipe(_messages(system, user), max_new_tokens=self._max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


def _ollama_reachable(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2):
            return True
    except OSError:
        return False


def resolve_provider(settings: Settings) -> str:
    """Turn `auto` into a concrete provider name based on what is available."""
    if settings.llm_provider != "auto":
        return settings.llm_provider
    if _ollama_reachable(settings.ollama_host):
        return "ollama"
    if settings.groq_api_key is not None:
        return "groq"
    return "huggingface"


_PROVIDERS: dict[str, type] = {
    "ollama": OllamaClient,
    "groq": GroqClient,
    "huggingface": HuggingFaceLocalClient,
}


def create_llm(settings: Settings) -> LLMClient:
    """Factory: build the configured LLM client (like resolving an interface from DI)."""
    return _PROVIDERS[resolve_provider(settings)](settings)
