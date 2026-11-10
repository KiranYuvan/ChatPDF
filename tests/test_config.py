import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_are_valid() -> None:
    s = Settings(_env_file=None)
    assert s.chunk_overlap < s.chunk_size
    assert s.groq_api_key is None  # no secret has a default


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_groq_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CHATPDF_GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        Settings(_env_file=None, llm_provider="groq")


def test_secret_is_read_from_env_and_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_value")
    s = Settings(_env_file=None)
    assert s.groq_api_key is not None
    assert s.groq_api_key.get_secret_value() == "gsk_test_value"
    assert "gsk_test_value" not in repr(s)  # never leaks into logs
